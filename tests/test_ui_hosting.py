"""Hosted UI, offline: the container entrypoint, the secret copy, and the image, UI and auth stacks."""

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "infra")]

from auth_stack import EarningsAuthStack  # noqa: E402
from test_infra import MODEL_ID  # noqa: E402
from ui_image_stack import EarningsUiImageStack  # noqa: E402
from ui_stack import EarningsUiStack  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


entry = _load("ui_entrypoint", ROOT / "ui" / "entrypoint.py")
cfg = _load("ui_auth_config_h", ROOT / "scripts" / "ui_auth_config.py")

ENV = {"AUTH_REDIRECT_URI": "https://ui.example/oauth2callback", "AUTH_COOKIE_SECRET": "k",
       "AUTH_CLIENT_ID": "c", "AUTH_CLIENT_SECRET": 's"x', "AUTH_METADATA_URL": "https://idp/.well-known/openid-configuration"}


# --- container entrypoint ------------------------------------------------------------------------------------------

def test_entrypoint_writes_the_auth_section_streamlit_reads():
    auth = tomllib.loads(entry.secrets_toml(ENV))["auth"]
    assert auth == {"redirect_uri": "https://ui.example/oauth2callback", "cookie_secret": "k", "client_id": "c",
                    "client_secret": 's"x', "server_metadata_url": "https://idp/.well-known/openid-configuration"}


@pytest.mark.parametrize("missing", list(ENV))
def test_no_file_without_every_value(missing):
    assert entry.secrets_toml({**ENV, missing: ""}) is None


def test_streamlit_listens_on_the_container_port_without_telemetry():
    cmd = entry.streamlit_command()
    assert cmd[cmd.index("--server.port") + 1] == "8501" and cmd[cmd.index("--server.address") + 1] == "0.0.0.0"
    assert cmd[cmd.index("--browser.gatherUsageStats") + 1] == "false"


def test_secrets_are_removed_from_streamlits_environment(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(entry, "__file__", str(tmp_path / "entrypoint.py"))
    monkeypatch.setattr(entry.os, "execvpe", lambda f, args, env: seen.update(env=env))
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    entry.main()
    assert not {"AUTH_CLIENT_SECRET", "AUTH_COOKIE_SECRET"} & set(seen["env"])
    written = tmp_path / ".streamlit" / "secrets.toml"
    assert written.stat().st_mode & 0o777 == 0o600


# --- client secret copy --------------------------------------------------------------------------------------------

class FakeSM:
    class exceptions:
        class ResourceNotFoundException(Exception):
            pass

    def __init__(self, current=None):
        self.current, self.puts = current, []

    def get_secret_value(self, SecretId):
        if self.current is None:
            raise RuntimeError("no current version")
        return {"SecretString": self.current}

    def put_secret_value(self, SecretId, SecretString):
        self.puts.append(SecretString)


def test_secret_copied_only_when_it_differs():
    sm = FakeSM(current="placeholder")
    assert cfg.copy_to_secrets_manager(sm, "n", "real") and sm.puts == ["real"]
    sm = FakeSM(current="real")
    assert not cfg.copy_to_secrets_manager(sm, "n", "real") and sm.puts == []
    sm = FakeSM(current=None)
    assert cfg.copy_to_secrets_manager(sm, "n", "real") and sm.puts == ["real"]


# --- stacks --------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def templates(tmp_path_factory):
    from test_infra import stacks
    app_runtime, _ = stacks(tmp_path_factory.mktemp("rt"))
    app = app_runtime.node.root
    auth = EarningsAuthStack(app, "Auth", domain_prefix="p", callback_urls=["http://localhost:8501/oauth2callback"],
                             logout_urls=["http://localhost:8501/"], ui_hostname="ui.example")
    image = EarningsUiImageStack(app, "Image", source_dir=ROOT / "ui")
    ui = EarningsUiStack(app, "Ui", repository=image.repository, image_tag=image.image_tag,
                         runtime=app_runtime.runtime, client_id=auth.client.user_pool_client_id,
                         metadata_url=auth.metadata_url, client_secret=auth.client_secret,
                         cookie_secret=auth.cookie_secret, hostname="ui.example")
    return {name: Template.from_stack(s) for name, s in (("auth", auth), ("image", image), ("ui", ui))}


def _one(t, kind):
    (r,) = t.find_resources(kind).values()
    return r["Properties"]


def test_service_runs_one_task_with_streamlits_health_check(templates):
    p = _one(templates["ui"], "AWS::ECS::ExpressGatewayService")
    assert p["ScalingTarget"] == {"MinTaskCount": 1, "MaxTaskCount": 1}
    assert p["HealthCheckPath"] == "/_stcore/health"


def test_task_is_arm64_half_vcpu_one_gb(templates):
    p = _one(templates["ui"], "AWS::ECS::TaskDefinition")
    assert p["RuntimePlatform"]["CpuArchitecture"] == "ARM64"
    assert (p["Cpu"], p["Memory"]) == ("512", "1024")
    (c,) = p["ContainerDefinitions"]
    assert c["PortMappings"][0]["ContainerPort"] == 8501
    assert sorted(s["Name"] for s in c["Secrets"]) == ["AUTH_CLIENT_SECRET", "AUTH_COOKIE_SECRET"]
    env = {e["Name"]: e["Value"] for e in c["Environment"]}
    assert env["AUTH_REDIRECT_URI"] == "https://ui.example/oauth2callback"
    assert "AUTH_CLIENT_SECRET" not in env and "AUTH_COOKIE_SECRET" not in env


def test_task_role_may_only_invoke_the_runtime(templates):
    t = templates["ui"]
    roles = t.find_resources("AWS::IAM::Role")
    task_role = next(k for k, r in roles.items() if k.startswith("TaskTaskRole"))
    policies = [p["Properties"] for p in t.find_resources("AWS::IAM::Policy").values()
                if any(r.get("Ref") == task_role for r in p["Properties"]["Roles"])]
    actions = {s["Action"] for p in policies for s in p["PolicyDocument"]["Statement"]}
    assert actions == {"bedrock-agentcore:InvokeAgentRuntime"}


def test_container_logs_expire(templates):
    assert _one(templates["ui"], "AWS::Logs::LogGroup")["RetentionInDays"] == 7
    assert _one(templates["image"], "AWS::Logs::LogGroup")["RetentionInDays"] == 7


def test_image_repository_and_build(templates):
    t = templates["image"]
    repo = _one(t, "AWS::ECR::Repository")
    assert repo["ImageTagMutability"] == "IMMUTABLE"
    assert json.loads(repo["LifecyclePolicy"]["LifecyclePolicyText"])["rules"][0]["selection"]["countNumber"] == 5
    env = _one(t, "AWS::CodeBuild::Project")["Environment"]
    assert env["Type"] == "ARM_CONTAINER" and env["PrivilegedMode"] is True
    assert not t.find_resources("Custom::ECRAutoDeleteImages")  # its Lambda's log group would never expire


def test_hosted_url_is_a_callback_and_logout_url(templates):
    p = _one(templates["auth"], "AWS::Cognito::UserPoolClient")
    assert "https://ui.example/oauth2callback" in p["CallbackURLs"] and "https://ui.example/" in p["LogoutURLs"]


def test_auth_stack_runs_no_helper_lambda(templates):
    t = templates["auth"]
    assert not t.find_resources("AWS::Lambda::Function") and not t.find_resources("Custom::AWS")
    assert len(t.find_resources("AWS::SecretsManager::Secret")) == 2


def test_dockerignore_keeps_local_secrets_and_venv_out():
    ignored = (ROOT / "ui" / ".dockerignore").read_text().split()
    assert ".streamlit" in ignored and ".venv" in ignored


def test_hostname_file_is_git_ignored():
    assert "infra/.ui_hostname" in (ROOT / ".gitignore").read_text().splitlines()
