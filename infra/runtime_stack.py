"""The agent side of the system: the AgentCore Runtime and what it may touch.

A separate stack from the tool, because it has a different lifetime: the runtime
is torn down after testing, while the Lambda and its secret stay.

    scripts/build_agent.sh && cdk deploy EarningsResearchAgentRuntime
    cdk destroy EarningsResearchAgentRuntime

It reads the Gateway, guardrail, prompt and audit log IDs from their stacks (cross-stack exports),
so destroy the runtime before replacing any of them. Search and fetch go through the Gateway
(gateway_stack.py): the role may invoke that Gateway, and neither the Knowledge Base nor the
transcript Lambda directly.
"""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import custom_resources as cr
from constructs import Construct

MEMORY_EXPIRY_DAYS = 7
# The runtime's log group holds its spans (full prompts, questions and answers) and stdout. AgentCore creates
# it together with the runtime, with no expiry and outside the stack; the stack sets this retention right after
# and deletes the group when the runtime goes.
RUNTIME_LOG_RETENTION_DAYS = 7
# Spend caps (agent/caps.py), across all callers. A turn runs up to 40K tokens (budget.py).
TURNS_PER_SESSION = 30
TURNS_PER_DAY = 100


class EarningsRuntimeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        code_path: Path,
        model_id: str,
        gateway: agentcore.IGateway,
        guardrail: bedrock.CfnGuardrail,
        guardrail_version: bedrock.CfnGuardrailVersion,
        prompts: dict[str, tuple[bedrock.CfnPrompt, bedrock.CfnPromptVersion]],
        audit_log: logs.ILogGroup,
        helper_log_group: logs.ILogGroup,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Session memory (agent/session.py): short-term events only, no extraction strategies, so nothing
        # outlives the session's events. Kept for the minimum period AgentCore allows. Part of this stack,
        # so destroying the runtime also deletes every stored conversation.
        memory = agentcore.Memory(
            self,
            "SessionMemory",
            memory_name="earnings_research_agent_sessions",
            description="Earlier turns of a chat session: question, answer, ticker, quarter, cited IDs",
            expiration_duration=Duration.days(MEMORY_EXPIRY_DAYS),
        )

        # Fetch requests waiting for a reviewer (agent/approvals.py). Requests expire after 24 hours (TTL), and the
        # table goes with the runtime, like the session memory.
        requests_table = dynamodb.Table(
            self,
            "FetchRequests",
            partition_key=dynamodb.Attribute(name="request_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="expires_ttl",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Spend caps (agent/caps.py): one counter per session and per UTC day, expired by TTL. Goes with the
        # runtime: a fresh runtime starts with fresh counters, which is fine for caps measured in days.
        usage_table = dynamodb.Table(
            self,
            "UsageCounters",
            partition_key=dynamodb.Attribute(name="counter", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="expires_ttl",
            removal_policy=RemovalPolicy.DESTROY,
        )

        runtime = agentcore.Runtime(
            self,
            "AgentRuntime",
            runtime_name="earnings_research_agent",
            description="Answers questions about earnings calls with cited passages. Research aid, not investment advice.",
            agent_runtime_artifact=agentcore.AgentRuntimeArtifact.from_code_asset(
                path=str(code_path),
                runtime=agentcore.AgentCoreRuntime.PYTHON_3_12,
                # serve.py: v2, one question per invocation (main.py, v1, stays in the bundle).
                # opentelemetry-instrument wraps it so that runs produce traces.
                entrypoint=["opentelemetry-instrument", "serve.py"],
            ),
            environment_variables={
                "MODEL_ID": model_id,
                # agent/gateway.py: search and fetch go over MCP to this Gateway.
                "GATEWAY_URL": gateway.gateway_url,
                "GUARDRAIL_ID": guardrail.attr_guardrail_id,
                "GUARDRAIL_VERSION": guardrail_version.attr_version,
                # agent/prompts.py loads these pinned versions and fails rather than fall back to a file.
                "AUDIT_LOG_GROUP": audit_log.log_group_name,
                "MEMORY_ID": memory.memory_id,
                "REQUESTS_TABLE": requests_table.table_name,
                "USAGE_TABLE": usage_table.table_name,
                "TURNS_PER_SESSION": str(TURNS_PER_SESSION),
                "TURNS_PER_DAY": str(TURNS_PER_DAY),
                "PROMPT_SOURCE": "bedrock",
                **{
                    f"{name.upper()}_PROMPT_{key}": value
                    for name, (prompt, version) in prompts.items()
                    for key, value in (("ID", prompt.attr_id), ("VERSION", version.attr_version))
                },
            },
            tracing_enabled=True,
        )

        # AgentCore creates the log group when it creates the runtime (a LogGroup resource here fails: "already
        # exists", tried 19 Nov), named <runtime id>-<endpoint>, endpoint DEFAULT. So: set its retention once the
        # runtime exists, and delete the group with the stack instead of by hand.
        log_group_name = f"/aws/bedrock-agentcore/runtimes/{runtime.agent_runtime_id}-DEFAULT"
        retention = cr.AwsCustomResource(
            self,
            "RuntimeLogRetention",
            on_create=cr.AwsSdkCall(
                service="CloudWatchLogs",
                action="putRetentionPolicy",
                parameters={"logGroupName": log_group_name, "retentionInDays": RUNTIME_LOG_RETENTION_DAYS},
                physical_resource_id=cr.PhysicalResourceId.of(log_group_name),
            ),
            on_update=cr.AwsSdkCall(
                service="CloudWatchLogs",
                action="putRetentionPolicy",
                parameters={"logGroupName": log_group_name, "retentionInDays": RUNTIME_LOG_RETENTION_DAYS},
                physical_resource_id=cr.PhysicalResourceId.of(log_group_name),
            ),
            on_delete=cr.AwsSdkCall(
                service="CloudWatchLogs",
                action="deleteLogGroup",
                parameters={"logGroupName": log_group_name},
                ignore_error_codes_matching="ResourceNotFoundException",
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=["logs:PutRetentionPolicy", "logs:DeleteLogGroup"],
                    resources=[f"arn:{self.partition}:logs:{self.region}:{self.account}:log-group:{log_group_name}",
                               f"arn:{self.partition}:logs:{self.region}:{self.account}:log-group:{log_group_name}:*"],
                )
            ]),
            install_latest_aws_sdk=False,
            # The custom resource's own logs go to a group in the audit stack, which outlives this one (audit_stack.py).
            log_group=helper_log_group,
        )
        retention.node.add_dependency(runtime)
        self.runtime = runtime  # the UI stack grants invoke on it (ui_stack.py)

        # Least privilege: this one Gateway, this one model and (below) this one guardrail, set of prompts and
        # audit log. The Gateway is the only way to the Knowledge Base and the transcript Lambda.
        # An inference profile routes between EU regions, so the role needs the
        # profile and the model behind it in each of them ("eu.amazon.x" -> "amazon.x").
        gateway.grant_invoke(runtime)
        foundation_model = model_id.split(".", 1)[1]
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    f"arn:{self.partition}:bedrock:{self.region}:{self.account}:inference-profile/{model_id}",
                    f"arn:{self.partition}:bedrock:eu-*::foundation-model/{foundation_model}",
                ],
            )
        )

        # agent/guard.py checks the question and the answer with this one guardrail.
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:ApplyGuardrail"],
                resources=[guardrail.attr_guardrail_arn, f"{guardrail.attr_guardrail_arn}:*"],
            )
        )

        # agent/prompts.py reads the agent's prompts, these and no others.
        prompt_arns = [prompt.attr_arn for prompt, _ in prompts.values()]
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:GetPrompt"],
                resources=prompt_arns + [f"{arn}:*" for arn in prompt_arns],
            )
        )

        # agent/audit.py writes one record per turn to this group: new streams and events, no reads.
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                # Built from the name: the group's Arn attribute already ends in ":*".
                resources=[
                    f"arn:{self.partition}:logs:{self.region}:{self.account}"
                    f":log-group:{audit_log.log_group_name}:log-stream:*"
                ],
            )
        )

        # agent/session.py reads and appends this memory's short-term events, and nothing else on it.
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock-agentcore:CreateEvent", "bedrock-agentcore:ListEvents"],
                resources=[memory.memory_arn],
            )
        )

        # agent/approvals.py: create, read, decide (conditional update) and list fetch requests. No deletes:
        # requests expire by TTL.
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem", "dynamodb:Scan"],
                resources=[requests_table.table_arn],
            )
        )

        # agent/caps.py: count turns (conditional updates in one transaction). Nothing else on the table.
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["dynamodb:UpdateItem"],
                resources=[usage_table.table_arn],
            )
        )

        # AgentCore uses this to let X-Ray deliver spans to the runtime's log group.
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["logs:PutResourcePolicy"],
                resources=[
                    f"arn:{self.partition}:logs:{self.region}:{self.account}"
                    ":log-group:/aws/bedrock-agentcore/runtimes/earnings_research_agent-*"
                ],
            )
        )

        CfnOutput(self, "RuntimeArn", value=runtime.agent_runtime_arn)
        CfnOutput(self, "MemoryId", value=memory.memory_id)
