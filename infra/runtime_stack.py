"""The agent side of the system: the AgentCore Runtime and what it may touch.

A separate stack from the tool, because it has a different lifetime: the runtime
is torn down after testing, while the Lambda and its secret stay.

    scripts/build_agent.sh && cdk deploy EarningsResearchAgentRuntime
    cdk destroy EarningsResearchAgentRuntime
"""

from pathlib import Path

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from constructs import Construct


class EarningsRuntimeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        code_path: Path,
        model_id: str,
        tool: lambda_.IFunction,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        runtime = agentcore.Runtime(
            self,
            "AgentRuntime",
            runtime_name="earnings_research_agent",
            description="Summarises earnings calls. Research aid, not investment advice.",
            agent_runtime_artifact=agentcore.AgentRuntimeArtifact.from_code_asset(
                path=str(code_path),
                runtime=agentcore.AgentCoreRuntime.PYTHON_3_12,
                # opentelemetry-instrument wraps main.py so that runs produce traces.
                entrypoint=["opentelemetry-instrument", "main.py"],
            ),
            environment_variables={"MODEL_ID": model_id, "TOOL_FUNCTION_NAME": tool.function_name},
            tracing_enabled=True,
        )

        # Least privilege: this one tool and this one model. An inference profile
        # routes between EU regions, so the role needs the profile and the model
        # behind it in each of them ("eu.amazon.x" -> "amazon.x").
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
