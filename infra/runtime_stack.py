"""The agent side of the system: the AgentCore Runtime and what it may touch.

A separate stack from the tool, because it has a different lifetime: the runtime
is torn down after testing, while the Lambda and its secret stay.

    scripts/build_agent.sh && cdk deploy EarningsResearchAgentRuntime
    cdk destroy EarningsResearchAgentRuntime

It reads the Knowledge Base, guardrail, prompt and audit log IDs from their stacks (cross-stack
exports), so destroy the runtime before replacing any of them, e.g. switching the KB's vector store.
"""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

MEMORY_EXPIRY_DAYS = 7


class EarningsRuntimeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        code_path: Path,
        model_id: str,
        tool: lambda_.IFunction,
        knowledge_base: bedrock.CfnKnowledgeBase,
        guardrail: bedrock.CfnGuardrail,
        guardrail_version: bedrock.CfnGuardrailVersion,
        prompts: dict[str, tuple[bedrock.CfnPrompt, bedrock.CfnPromptVersion]],
        audit_log: logs.ILogGroup,
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
                "TOOL_FUNCTION_NAME": tool.function_name,
                "KNOWLEDGE_BASE_ID": knowledge_base.attr_knowledge_base_id,
                "GUARDRAIL_ID": guardrail.attr_guardrail_id,
                "GUARDRAIL_VERSION": guardrail_version.attr_version,
                # agent/prompts.py loads these pinned versions and fails rather than fall back to a file.
                "AUDIT_LOG_GROUP": audit_log.log_group_name,
                "MEMORY_ID": memory.memory_id,
                "PROMPT_SOURCE": "bedrock",
                **{
                    f"{name.upper()}_PROMPT_{key}": value
                    for name, (prompt, version) in prompts.items()
                    for key, value in (("ID", prompt.attr_id), ("VERSION", version.attr_version))
                },
            },
            tracing_enabled=True,
        )

        # Least privilege: this one tool, this one model and (below) this one KB, guardrail, set of prompts and audit log.
        # An inference profile routes between EU regions, so the role needs the
        # profile and the model behind it in each of them ("eu.amazon.x" -> "amazon.x").
        tool.grant_invoke(runtime)
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

        # retrieve_passages searches this one Knowledge Base. The KB's own role
        # embeds the query, so the runtime needs no embedding-model permission.
        runtime.add_to_role_policy(
            iam.PolicyStatement(actions=["bedrock:Retrieve"], resources=[knowledge_base.attr_knowledge_base_arn])
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
