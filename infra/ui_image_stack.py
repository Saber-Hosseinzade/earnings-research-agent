"""The UI's container image: an ECR repository and a CodeBuild project that builds ui/ for ARM64.

Own stack, kept deployed (an idle project costs nothing; the repository keeps at most MAX_IMAGES images).
No Docker is needed locally: CDK uploads ui/ as a zip, CodeBuild builds and pushes the image, tagged with
the zip's hash, so the same source is never built twice and the UI stack knows the tag at synth time.

    scripts/deploy_ui.sh        # deploys this stack, builds the image if its tag is missing, deploys the UI

To destroy this stack, delete the repository's images first (no auto-delete: its helper Lambda would leave
a log group that never expires).
"""

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3_assets as s3_assets
from constructs import Construct

REPOSITORY_NAME = "earnings-research-agent-ui"
MAX_IMAGES = 5
EXCLUDE = [".venv", ".streamlit", "__pycache__", "*.pyc"]

BUILD_SPEC = {
    "version": "0.2",
    "phases": {
        "pre_build": {"commands": [
            'aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | docker login --username AWS --password-stdin "${REPO_URI%%/*}"',
        ]},
        "build": {"commands": [
            'docker build --tag "$REPO_URI:$IMAGE_TAG" .',
            'docker push "$REPO_URI:$IMAGE_TAG"',
        ]},
    },
}


class EarningsUiImageStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, source_dir: Path, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.repository = ecr.Repository(
            self,
            "Repository",
            repository_name=REPOSITORY_NAME,
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,  # a tag is a source hash: it never moves
            lifecycle_rules=[ecr.LifecycleRule(max_image_count=MAX_IMAGES)],
            removal_policy=RemovalPolicy.DESTROY,
        )

        source = s3_assets.Asset(self, "Source", path=str(source_dir), exclude=EXCLUDE)
        self.image_tag = source.asset_hash

        log_group = logs.LogGroup(self, "BuildLogs", log_group_name="/earnings-research-agent/ui-build",
                                  retention=logs.RetentionDays.ONE_WEEK, removal_policy=RemovalPolicy.DESTROY)
        project = codebuild.Project(
            self,
            "Build",
            project_name="earnings-research-agent-ui",
            description="Builds the Streamlit UI image (ui/) for ARM64 and pushes it to ECR",
            source=codebuild.Source.s3(bucket=source.bucket, path=source.s3_object_key),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxArmBuildImage.AMAZON_LINUX_2023_STANDARD_3_0,
                compute_type=codebuild.ComputeType.SMALL,
                privileged=True,  # docker build needs it; the build runs in AWS, not on the laptop
            ),
            environment_variables={
                "REPO_URI": codebuild.BuildEnvironmentVariable(value=self.repository.repository_uri),
                "IMAGE_TAG": codebuild.BuildEnvironmentVariable(value=self.image_tag),
            },
            build_spec=codebuild.BuildSpec.from_object(BUILD_SPEC),
            logging=codebuild.LoggingOptions(cloud_watch=codebuild.CloudWatchLoggingOptions(log_group=log_group)),
            timeout=Duration.minutes(20),
        )
        self.repository.grant_pull_push(project)

        CfnOutput(self, "ProjectName", value=project.project_name)
        CfnOutput(self, "RepositoryName", value=self.repository.repository_name)
        CfnOutput(self, "ImageTag", value=self.image_tag)
