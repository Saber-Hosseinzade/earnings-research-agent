"""The MCP side of the tools: an AgentCore Gateway with two Lambda targets.

    search_passages  (target "passages")     search_lambda/handler.py -> Bedrock Knowledge Base Retrieve
    get_transcript   (target "transcripts")  tool_lambda/handler.py (EarningsResearchAgentTool) -> transcript API

Inbound auth is AWS IAM (SigV4): only principals granted ``InvokeGateway`` on this Gateway can
list or call its tools; the runtime role is the one granted (runtime_stack.py). Outbound, the
Gateway invokes the two functions with its own role, which may invoke those two and nothing else.

Billed per MCP operation, with no hourly charge, so it stays deployed like the tool and KB stacks.
Semantic tool search is off: two tools don't need it, and the index is charged.

MCP is the transport, not the control point: the agent's search limit, pinning and citation
ledger (agent/retrieval.py) and the approval gate (agent/approvals.py) run in the agent before
any call reaches this Gateway.
"""

from pathlib import Path

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import Duration
from constructs import Construct

SEARCH_DIR = Path(__file__).resolve().parents[1] / "search_lambda"

S = agentcore.SchemaDefinitionType

SEARCH_TOOL = agentcore.ToolDefinition(
    name="search_passages",
    description=("Search the earnings-call transcript corpus. Returns up to k passages (one speaker turn each) "
                 "with passage_id, ticker, quarter, speaker, role, start_seconds, score and text."),
    input_schema=agentcore.SchemaDefinition(
        type=S.OBJECT,
        properties={
            "query": agentcore.SchemaDefinition(type=S.STRING, description="What to search for, in plain words"),
            "ticker": agentcore.SchemaDefinition(type=S.STRING, description="Limit to one company, e.g. AMD"),
            "quarter": agentcore.SchemaDefinition(type=S.STRING, description="Limit to one fiscal quarter, e.g. 2026Q2"),
            "k": agentcore.SchemaDefinition(type=S.INTEGER, description="Number of passages, 1 to 8 (default 5)"),
        },
        required=["query"],
    ),
)

FETCH_TOOL = agentcore.ToolDefinition(
    name="get_transcript",
    description=("Fetch one earnings-call transcript from the external transcript API, by ticker and the "
                 "company's fiscal quarter. Counts against a daily API quota."),
    input_schema=agentcore.SchemaDefinition(
        type=S.OBJECT,
        properties={
            "ticker": agentcore.SchemaDefinition(type=S.STRING, description="Ticker symbol, e.g. NVDA"),
            "quarter": agentcore.SchemaDefinition(type=S.STRING, description="Fiscal quarter, e.g. 2026Q3"),
        },
        required=["ticker", "quarter"],
    ),
)


class EarningsGatewayStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        tool: lambda_.IFunction,
        knowledge_base: bedrock.CfnKnowledgeBase,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        search_logs = logs.LogGroup(
            self, "SearchLogs", retention=logs.RetentionDays.ONE_WEEK, removal_policy=RemovalPolicy.DESTROY
        )
        self.search_function = lambda_.Function(
            self,
            "SearchPassages",
            runtime=lambda_.Runtime.PYTHON_3_13,
            architecture=lambda_.Architecture.ARM_64,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(SEARCH_DIR)),
            timeout=Duration.seconds(15),
            memory_size=256,
            environment={"KNOWLEDGE_BASE_ID": knowledge_base.attr_knowledge_base_id},
            log_group=search_logs,
        )
        # This one Knowledge Base. The KB's own role embeds the query, so no embedding-model permission here.
        self.search_function.add_to_role_policy(
            iam.PolicyStatement(actions=["bedrock:Retrieve"], resources=[knowledge_base.attr_knowledge_base_arn])
        )

        self.gateway = agentcore.Gateway(
            self,
            "Gateway",
            gateway_name="earnings-research-agent",
            description="Earnings-call tools over MCP: corpus search and one-call transcript fetch",
            authorizer_configuration=agentcore.GatewayAuthorizer.using_aws_iam(),
            # Without an explicit configuration the construct turns on semantic tool search (searchType
            # SEMANTIC), which indexes the tools at a charge and adds a third, built-in search tool.
            protocol_configuration=agentcore.McpProtocolConfiguration(
                instructions="Earnings-call research tools: search the fixed transcript corpus, or fetch one call.",
            ),
        )
        # Default credential provider: the Gateway's own role, granted invoke on each target function.
        self.gateway.add_lambda_target(
            "Passages",
            gateway_target_name="passages",
            description="Knowledge Base search over the fixed transcript corpus",
            lambda_function=self.search_function,
            tool_schema=agentcore.ToolSchema.from_inline([SEARCH_TOOL]),
        )
        self.gateway.add_lambda_target(
            "Transcripts",
            gateway_target_name="transcripts",
            description="One transcript from the external API (the agent calls it only after a reviewer approves)",
            lambda_function=tool,
            tool_schema=agentcore.ToolSchema.from_inline([FETCH_TOOL]),
        )

        CfnOutput(self, "GatewayUrl", value=self.gateway.gateway_url)
        CfnOutput(self, "GatewayId", value=self.gateway.gateway_id)
        CfnOutput(self, "SearchFunctionName", value=self.search_function.function_name)
