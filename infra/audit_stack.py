"""Audit log group: one JSON record per turn (agent/audit.py lists the fields).

Own stack, kept deployed: it costs nothing without records, and its lifetime is not
the runtime's — records must outlive a torn-down runtime.

    cdk deploy EarningsResearchAgentAudit

Records expire after RETENTION. The group is retained if the stack is destroyed, so
`cdk destroy` cannot delete an audit trail; its records still expire on schedule.
"""

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_logs as logs
from constructs import Construct

LOG_GROUP_NAME = "/earnings-research-agent/audit"
RETENTION = logs.RetentionDays.ONE_MONTH


class EarningsAuditStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.log_group = logs.LogGroup(
            self,
            "AuditLog",
            log_group_name=LOG_GROUP_NAME,
            retention=RETENTION,
            removal_policy=RemovalPolicy.RETAIN,
        )
        CfnOutput(self, "AuditLogGroupName", value=self.log_group.log_group_name)
