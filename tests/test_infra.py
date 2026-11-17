"""The runtime and guardrail stacks, offline: the role reaches one tool, model, KB and guardrail."""

import json
import sys
from pathlib import Path

import aws_cdk as cdk
from aws_cdk.assertions import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))

from audit_stack import LOG_GROUP_NAME, EarningsAuditStack  # noqa: E402
from guardrail_stack import PII_TYPES, EarningsGuardrailStack  # noqa: E402
from kb_stack import EarningsKnowledgeBaseStack  # noqa: E402
from prompt_stack import PROMPT_NAMES, EarningsPromptStack  # noqa: E402
from runtime_stack import EarningsRuntimeStack  # noqa: E402
from stack import EarningsToolStack  # noqa: E402

MODEL_ID = "eu.amazon.nova-2-lite-v1:0"


def runtime(tmp_path):
    app = cdk.App()
    tool = EarningsToolStack(app, "Tool")
    kb = EarningsKnowledgeBaseStack(app, "KB", vector_store="s3vectors")
    guard = EarningsGuardrailStack(app, "Guard")
    prompts = EarningsPromptStack(app, "Prompts")
    audit = EarningsAuditStack(app, "Audit")
    stack = EarningsRuntimeStack(app, "Runtime", code_path=tmp_path, model_id=MODEL_ID, tool=tool.function,
                                 knowledge_base=kb.knowledge_base, guardrail=guard.guardrail,
                                 guardrail_version=guard.version, prompts=prompts.prompts,
                                 audit_log=audit.log_group)
    return stack


def statements(tmp_path):
    policies = Template.from_stack(runtime(tmp_path)).find_resources("AWS::IAM::Policy")
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
    for e in entities:  # input set explicitly: "Action" alone masked only the output (live check)
        assert e["InputAction"] == e["OutputAction"] == "ANONYMIZE" and e["InputEnabled"] and e["OutputEnabled"]


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


# --- prompt management -----------------------------------------------------------------------------------------

def test_role_may_read_only_the_agents_prompts(tmp_path):
    found = by_action(statements(tmp_path), "bedrock:GetPrompt")
    assert len(found) == 1
    resource = json.dumps(found[0]["Resource"])
    assert resource.count("PromptArn") == 2 * len(PROMPT_NAMES)  # each prompt and its versions
    assert '"*"' not in resource


def test_runtime_loads_pinned_versions_and_never_a_file(tmp_path):
    res = Template.from_stack(runtime(tmp_path)).find_resources("AWS::BedrockAgentCore::Runtime")
    env = next(iter(res.values()))["Properties"]["EnvironmentVariables"]
    assert env["PROMPT_SOURCE"] == "bedrock"
    for name in PROMPT_NAMES:
        assert f"{name.upper()}_PROMPT_ID" in env and f"{name.upper()}_PROMPT_VERSION" in env


def _prompt_template(prompts_dir=None):
    kw = {"prompts_dir": prompts_dir} if prompts_dir else {}
    return Template.from_stack(EarningsPromptStack(cdk.App(), "P", **kw))


def test_prompts_are_the_repo_files_with_no_model():
    agent_prompts = Path(__file__).resolve().parents[1] / "agent" / "prompts"
    t = _prompt_template()
    prompts = {r["Properties"]["Name"]: r["Properties"] for r in t.find_resources("AWS::Bedrock::Prompt").values()}
    for name in PROMPT_NAMES:
        (variant,) = prompts[f"earnings-{name.replace('_', '-')}"]["Variants"]
        assert variant["TemplateConfiguration"]["Text"]["Text"] == (agent_prompts / f"{name}.txt").read_text()
        assert "ModelId" not in variant  # the model is config (MODEL_ID), not part of the prompt


def test_versions_are_retained_and_follow_the_text(tmp_path):
    for name in PROMPT_NAMES:
        (tmp_path / f"{name}.txt").write_text(f"{name} text one")
    versions = _prompt_template(tmp_path).find_resources("AWS::Bedrock::PromptVersion")
    assert {v["DeletionPolicy"] for v in versions.values()} == {"Retain"}
    before = {k: v["Properties"]["Description"] for k, v in versions.items()}
    (tmp_path / "answer_system.txt").write_text("answer_system text two")
    after = {k: v["Properties"]["Description"]
             for k, v in _prompt_template(tmp_path).find_resources("AWS::Bedrock::PromptVersion").items()}
    changed = [k for k in before if before[k] != after[k]]
    assert len(changed) == 1 and changed[0].startswith("AnswerSystem")


# --- audit log -------------------------------------------------------------------------------------------------

def test_audit_log_has_retention_and_survives_stack_deletion():
    groups = Template.from_stack(EarningsAuditStack(cdk.App(), "A")).find_resources("AWS::Logs::LogGroup")
    (group,) = groups.values()
    assert group["Properties"]["LogGroupName"] == LOG_GROUP_NAME
    assert group["Properties"]["RetentionInDays"] == 30
    assert group["DeletionPolicy"] == "Retain"


def test_role_may_only_append_to_the_audit_log(tmp_path):
    # The Runtime construct grants its own log group separately; this is the audit statement.
    found = [st for st in by_action(statements(tmp_path), "logs:PutLogEvents") if "AuditLog" in json.dumps(st)]
    assert len(found) == 1
    assert sorted(found[0]["Action"]) == ["logs:CreateLogStream", "logs:PutLogEvents"]  # no reads, no deletes
    resource = json.dumps(found[0]["Resource"])
    assert ":log-stream:*" in resource and "AuditLog" in resource
    assert ":*:log-stream" not in resource  # not built from the Arn attribute, which ends in ":*"


def test_runtime_gets_the_audit_log_group(tmp_path):
    res = Template.from_stack(runtime(tmp_path)).find_resources("AWS::BedrockAgentCore::Runtime")
    assert "AUDIT_LOG_GROUP" in next(iter(res.values()))["Properties"]["EnvironmentVariables"]
