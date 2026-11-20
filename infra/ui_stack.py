"""The hosted UI: Streamlit on ECS Express Mode (Fargate, ARM64) behind an AWS-managed HTTPS endpoint.

Own stack with the runtime's lifetime: deploy for a demo window, destroy after. Destroy it before the
runtime stack (it reads the runtime's ARN).

    scripts/deploy_ui.sh
    cdk destroy EarningsResearchAgentUi

The task is defined here (ARM64, secrets, log retention, a role that may only invoke the runtime) and
handed to Express Mode, which adds the load balancer, HTTPS on an AWS-managed domain, and scaling.

Spend: one task, never more (Streamlit keeps each session in the process's memory, and the ALB has no
sticky sessions configured). Turn and daily caps run in the runtime (agent/caps.py).
"""

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

SERVICE_NAME = "earnings-research-ui"
CONTAINER_PORT = 8501
CPU, MEMORY_MIB = 512, 1024  # 0.5 vCPU, 1 GB, ARM64: ~$0.065/h with the load balancer and public IPs
LOG_GROUP_NAME = "/earnings-research-agent/ui"


class EarningsUiStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, repository: ecr.IRepository, image_tag: str,
                 runtime: agentcore.IBedrockAgentRuntime, client_id: str, metadata_url: str,
                 client_secret: secretsmanager.ISecret, cookie_secret: secretsmanager.ISecret,
                 hostname: str = "", **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Container logs: who used the UI and any errors. The audit log (runtime side) keeps the turns.
        log_group = logs.LogGroup(self, "Logs", log_group_name=LOG_GROUP_NAME,
                                  retention=logs.RetentionDays.ONE_WEEK, removal_policy=RemovalPolicy.DESTROY)

        task = ecs.FargateTaskDefinition(
            self,
            "Task",
            cpu=CPU,
            memory_limit_mib=MEMORY_MIB,
            runtime_platform=ecs.RuntimePlatform(cpu_architecture=ecs.CpuArchitecture.ARM64,
                                                 operating_system_family=ecs.OperatingSystemFamily.LINUX),
        )
        # The task role may invoke this one runtime and nothing else. No AWS keys anywhere.
        runtime.grant_invoke_runtime(task.task_role)

        task.add_container(
            "ui",
            image=ecs.ContainerImage.from_ecr_repository(repository, image_tag),
            port_mappings=[ecs.PortMapping(container_port=CONTAINER_PORT)],
            environment={
                "AGENT_RUNTIME_ARN": runtime.agent_runtime_arn,
                "AWS_REGION": self.region,
                "AUTH_CLIENT_ID": client_id,
                "AUTH_METADATA_URL": metadata_url,
                # Empty until the endpoint is known (first deploy): the app then says sign-in is not configured.
                "AUTH_REDIRECT_URI": f"https://{hostname}/oauth2callback" if hostname else "",
            },
            # Read by the execution role at task start; ui/entrypoint.py writes them to secrets.toml.
            secrets={
                "AUTH_CLIENT_SECRET": ecs.Secret.from_secrets_manager(client_secret),
                "AUTH_COOKIE_SECRET": ecs.Secret.from_secrets_manager(cookie_secret),
            },
            logging=ecs.LogDrivers.aws_logs(stream_prefix="ui", log_group=log_group),
        )

        # Express Mode manages the load balancer, listener, certificate, security groups and scaling with this role.
        infrastructure_role = iam.Role(
            self,
            "InfrastructureRole",
            assumed_by=iam.ServicePrincipal("ecs.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonECSInfrastructureRoleforExpressGatewayServices")],
        )

        service = ecs.CfnExpressGatewayService(
            self,
            "Service",
            service_name=SERVICE_NAME,
            infrastructure_role_arn=infrastructure_role.role_arn,
            task_definition_arn=task.task_definition_arn,
            health_check_path="/_stcore/health",
            scaling_target=ecs.CfnExpressGatewayService.ExpressGatewayScalingTargetProperty(
                min_task_count=1, max_task_count=1),
        )
        CfnOutput(self, "Endpoint", value=service.attr_endpoint)
        CfnOutput(self, "ServiceArn", value=service.attr_service_arn)
