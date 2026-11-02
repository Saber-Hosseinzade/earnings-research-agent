"""The tool side of the system: the Lambda and its secret.

These outlive the agent runtime (see ``runtime_stack.py``), which is granted
permission to invoke this one function and nothing else.
"""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool_lambda"


class EarningsToolStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # The value is set out of band (`aws secretsmanager put-secret-value`),
        # so the API key never appears in code, in the template or in the repo.
        secret = secretsmanager.Secret(
            self,
            "AlphaVantageKey",
            description="Alpha Vantage API key for the earnings-research-agent tool",
            removal_policy=RemovalPolicy.DESTROY,
        )

        log_group = logs.LogGroup(
            self, "ToolLogs", retention=logs.RetentionDays.ONE_WEEK, removal_policy=RemovalPolicy.DESTROY
        )

        self.function = lambda_.Function(
            self,
            "EarningsTranscriptTool",
            runtime=lambda_.Runtime.PYTHON_3_13,
            architecture=lambda_.Architecture.ARM_64,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(TOOL_DIR)),
            # The handler's HTTP timeout is derived from the remaining time minus a
            # margin, so the function can always return a typed error before this.
            timeout=Duration.seconds(15),
            memory_size=256,
            environment={
                "SECRET_ARN": secret.secret_arn,
                "MAX_TRANSCRIPT_CHARS": "100000",
                "MAX_HTTP_TIMEOUT_S": "10",
                "SAFETY_MARGIN_S": "2",
            },
            log_group=log_group,
        )
        secret.grant_read(self.function)

        CfnOutput(self, "ToolFunctionName", value=self.function.function_name)
        CfnOutput(self, "SecretArn", value=secret.secret_arn)
