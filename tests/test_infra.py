"""The runtime stack, offline: its role reaches the one tool and the one model."""

import json
import sys
from pathlib import Path

import aws_cdk as cdk
from aws_cdk.assertions import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))

from runtime_stack import EarningsRuntimeStack  # noqa: E402
from stack import EarningsToolStack  # noqa: E402

MODEL_ID = "eu.amazon.nova-2-lite-v1:0"


def statements(tmp_path):
    app = cdk.App()
    tool = EarningsToolStack(app, "Tool")
    stack = EarningsRuntimeStack(app, "Runtime", code_path=tmp_path, model_id=MODEL_ID, tool=tool.function)
    policies = Template.from_stack(stack).find_resources("AWS::IAM::Policy")
    return [s for policy in policies.values() for s in policy["Properties"]["PolicyDocument"]["Statement"]]


def by_action(found, action):
    return [s for s in found if action in str(s["Action"])]


def test_role_may_invoke_only_the_tool(tmp_path):
    found = by_action(statements(tmp_path), "lambda:InvokeFunction")
    assert found
    assert '"*"' not in json.dumps([s["Resource"] for s in found])


def test_role_may_call_only_the_configured_model(tmp_path):
    found = by_action(statements(tmp_path), "bedrock:InvokeModel")
    resources = json.dumps([s["Resource"] for s in found])
    assert f"inference-profile/{MODEL_ID}" in resources
    assert "foundation-model/amazon.nova-2-lite-v1:0" in resources
    assert '"*"' not in resources
