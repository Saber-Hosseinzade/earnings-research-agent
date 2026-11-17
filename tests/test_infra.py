"""The runtime and guardrail stacks, offline: the role reaches one tool, model, KB and guardrail."""

import json
import sys
from pathlib import Path

import aws_cdk as cdk
from aws_cdk.assertions import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))

from guardrail_stack import PII_TYPES, EarningsGuardrailStack  # noqa: E402
from kb_stack import EarningsKnowledgeBaseStack  # noqa: E402
from runtime_stack import EarningsRuntimeStack  # noqa: E402
from stack import EarningsToolStack  # noqa: E402

MODEL_ID = "eu.amazon.nova-2-lite-v1:0"


def statements(tmp_path):
    app = cdk.App()
    tool = EarningsToolStack(app, "Tool")
    kb = EarningsKnowledgeBaseStack(app, "KB", vector_store="s3vectors")
    guard = EarningsGuardrailStack(app, "Guard")
    stack = EarningsRuntimeStack(app, "Runtime", code_path=tmp_path, model_id=MODEL_ID, tool=tool.function,
                                 knowledge_base=kb.knowledge_base, guardrail=guard.guardrail,
                                 guardrail_version=guard.version)
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


def test_role_may_retrieve_only_from_the_knowledge_base(tmp_path):
    found = by_action(statements(tmp_path), "bedrock:Retrieve")
    assert len(found) == 1
    assert found[0]["Action"] == "bedrock:Retrieve"
    resource = json.dumps(found[0]["Resource"])
    assert "KnowledgeBaseArn" in resource  # imported from the KB stack, not a wildcard
    assert '"*"' not in resource


def test_role_may_apply_only_the_guardrail(tmp_path):
    found = by_action(statements(tmp_path), "bedrock:ApplyGuardrail")
    assert len(found) == 1
    resource = json.dumps(found[0]["Resource"])
    assert "GuardrailArn" in resource
    assert '"*"' not in resource


def guardrail_props():
    app = cdk.App()
    props = Template.from_stack(EarningsGuardrailStack(app, "Guard")).find_resources("AWS::Bedrock::Guardrail")
    (resource,) = props.values()
    return resource["Properties"]


def test_guardrail_denies_advice_and_prompt_attacks():
    p = guardrail_props()
    (topic,) = p["TopicPolicyConfig"]["TopicsConfig"]
    assert topic["Type"] == "DENY" and topic["Name"] == "InvestmentAdvice"
    (attack,) = p["ContentPolicyConfig"]["FiltersConfig"]
    assert attack == {"Type": "PROMPT_ATTACK", "InputStrength": "HIGH", "OutputStrength": "NONE"}


def test_guardrail_masks_pii_but_not_speaker_names():
    entities = guardrail_props()["SensitiveInformationPolicyConfig"]["PiiEntitiesConfig"]
    types = {e["Type"] for e in entities}
    assert types == set(PII_TYPES)
    assert "NAME" not in types  # executives' names are the speakers the agent cites
    assert {e["Action"] for e in entities} == {"ANONYMIZE"}


def test_grounding_is_detect_only():
    filters = guardrail_props()["ContextualGroundingPolicyConfig"]["FiltersConfig"]
    assert {f["Type"] for f in filters} == {"GROUNDING", "RELEVANCE"}
    assert {f["Action"] for f in filters} == {"NONE"}  # flagged in code, answer kept


def test_version_changes_when_the_policy_changes(monkeypatch):
    import guardrail_stack

    def version_description():
        res = Template.from_stack(EarningsGuardrailStack(cdk.App(), "G")).find_resources("AWS::Bedrock::GuardrailVersion")
        return next(iter(res.values()))["Properties"]["Description"]

    before = version_description()
    monkeypatch.setattr(guardrail_stack, "GROUNDING_THRESHOLD", 0.8)
    assert version_description() != before
