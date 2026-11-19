"""The runtime, Gateway and guardrail stacks, offline: the role reaches one Gateway, model and guardrail."""

import json
import sys
from pathlib import Path

import aws_cdk as cdk
from aws_cdk.assertions import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))

from audit_stack import LOG_GROUP_NAME, EarningsAuditStack  # noqa: E402
from gateway_stack import EarningsGatewayStack  # noqa: E402
from guardrail_stack import PII_TYPES, EarningsGuardrailStack  # noqa: E402
from kb_stack import EarningsKnowledgeBaseStack  # noqa: E402
from prompt_stack import PROMPT_NAMES, EarningsPromptStack  # noqa: E402
from runtime_stack import EarningsRuntimeStack  # noqa: E402
from stack import EarningsToolStack  # noqa: E402

MODEL_ID = "eu.amazon.nova-2-lite-v1:0"


def stacks(tmp_path):
    app = cdk.App()
    tool = EarningsToolStack(app, "Tool")
    kb = EarningsKnowledgeBaseStack(app, "KB", vector_store="s3vectors")
    gateway = EarningsGatewayStack(app, "Gateway", tool=tool.function, knowledge_base=kb.knowledge_base)
    guard = EarningsGuardrailStack(app, "Guard")
    prompts = EarningsPromptStack(app, "Prompts")
    audit = EarningsAuditStack(app, "Audit")
    stack = EarningsRuntimeStack(app, "Runtime", code_path=tmp_path, model_id=MODEL_ID, gateway=gateway.gateway,
                                 guardrail=guard.guardrail, guardrail_version=guard.version,
                                 prompts=prompts.prompts, audit_log=audit.log_group,
                                 helper_log_group=audit.helper_log_group)
    return stack, gateway


def runtime(tmp_path):
    return stacks(tmp_path)[0]


def statements(tmp_path):
    policies = Template.from_stack(runtime(tmp_path)).find_resources("AWS::IAM::Policy")
    return [s for policy in policies.values() for s in policy["Properties"]["PolicyDocument"]["Statement"]]


def by_action(found, action):
    return [s for s in found if action in str(s["Action"])]


def test_role_reaches_tools_only_through_the_gateway(tmp_path):
    found = statements(tmp_path)
    invoke = by_action(found, "bedrock-agentcore:InvokeGateway")
    assert len(invoke) == 1
    resource = json.dumps(invoke[0]["Resource"])
    assert "GatewayArn" in resource and '"*"' not in resource
    # Neither the transcript Lambda nor the Knowledge Base directly.
    assert not by_action(found, "lambda:InvokeFunction")
    assert not by_action(found, "bedrock:Retrieve")


def test_role_may_call_only_the_configured_model(tmp_path):
    found = by_action(statements(tmp_path), "bedrock:InvokeModel")
    resources = json.dumps([s["Resource"] for s in found])
    assert f"inference-profile/{MODEL_ID}" in resources
    assert "foundation-model/amazon.nova-2-lite-v1:0" in resources
    assert '"*"' not in resources


def gateway_template(tmp_path):
    return Template.from_stack(stacks(tmp_path)[1])


def test_gateway_is_mcp_with_iam_auth(tmp_path):
    gw = next(iter(gateway_template(tmp_path).find_resources("AWS::BedrockAgentCore::Gateway").values()))
    assert gw["Properties"]["AuthorizerType"] == "AWS_IAM"
    assert gw["Properties"]["ProtocolType"] == "MCP"
    # Semantic tool search off: charged per indexed tool, and two tools don't need it.
    assert "SearchType" not in json.dumps(gw["Properties"].get("ProtocolConfiguration", {}))


def test_gateway_exposes_exactly_the_two_tools(tmp_path):
    targets = gateway_template(tmp_path).find_resources("AWS::BedrockAgentCore::GatewayTarget").values()
    tools = {t["Properties"]["Name"]: [d["Name"] for d in t["Properties"]["TargetConfiguration"]["Mcp"]["Lambda"]
                                        ["ToolSchema"]["InlinePayload"]] for t in targets}
    assert tools == {"passages": ["search_passages"], "transcripts": ["get_transcript"]}


def test_gateway_role_may_invoke_only_its_two_functions(tmp_path):
    policies = gateway_template(tmp_path).find_resources("AWS::IAM::Policy")
    gw = [s for k, p in policies.items() if k.startswith("GatewayServiceRole")
          for s in p["Properties"]["PolicyDocument"]["Statement"]]
    assert gw and all(s["Action"] == "lambda:InvokeFunction" for s in gw)
    resources = json.dumps([s["Resource"] for s in gw])
    assert "SearchPassages" in resources and "EarningsTranscriptTool" in resources and '"*"' not in resources


def test_search_function_may_retrieve_only_from_the_knowledge_base(tmp_path):
    policies = gateway_template(tmp_path).find_resources("AWS::IAM::Policy")
    found = [s for k, p in policies.items() if k.startswith("SearchPassages")
             for s in p["Properties"]["PolicyDocument"]["Statement"]]
    assert len(found) == 1 and found[0]["Action"] == "bedrock:Retrieve"
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


def test_guardrail_versions_are_retained():
    res = Template.from_stack(EarningsGuardrailStack(cdk.App(), "G")).find_resources("AWS::Bedrock::GuardrailVersion")
    assert {v["DeletionPolicy"] for v in res.values()} == {"Retain"}


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
    (group,) = [g for g in groups.values() if g["Properties"]["LogGroupName"] == LOG_GROUP_NAME]
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


def test_runtime_runs_the_v2_entrypoint_with_tracing(tmp_path):
    res = Template.from_stack(runtime(tmp_path)).find_resources("AWS::BedrockAgentCore::Runtime")
    code = next(iter(res.values()))["Properties"]["AgentRuntimeArtifact"]["CodeConfiguration"]
    assert code["EntryPoint"] == ["opentelemetry-instrument", "serve.py"]


# --- session memory ------------------------------------------------------------------------------------------

def test_session_memory_is_short_term_and_expires_after_a_week(tmp_path):
    (memory,) = Template.from_stack(runtime(tmp_path)).find_resources("AWS::BedrockAgentCore::Memory").values()
    assert memory["Properties"]["EventExpiryDuration"] == 7
    assert not memory["Properties"].get("MemoryStrategies")  # no long-term extraction


def test_role_may_only_read_and_append_session_events(tmp_path):
    found = by_action(statements(tmp_path), "bedrock-agentcore:CreateEvent")
    assert len(found) == 1
    assert sorted(found[0]["Action"]) == ["bedrock-agentcore:CreateEvent", "bedrock-agentcore:ListEvents"]
    assert "SessionMemory" in json.dumps(found[0]["Resource"]) and '"*"' not in json.dumps(found[0]["Resource"])


def test_runtime_gets_the_memory_id(tmp_path):
    res = Template.from_stack(runtime(tmp_path)).find_resources("AWS::BedrockAgentCore::Runtime")
    assert "MEMORY_ID" in next(iter(res.values()))["Properties"]["EnvironmentVariables"]


# --- approval gate ---------------------------------------------------------------------------------------------

def test_fetch_requests_expire_and_go_with_the_runtime(tmp_path):
    (table,) = Template.from_stack(runtime(tmp_path)).find_resources("AWS::DynamoDB::Table").values()
    assert table["Properties"]["TimeToLiveSpecification"] == {"AttributeName": "expires_ttl", "Enabled": True}
    assert table["DeletionPolicy"] == "Delete"


def test_role_may_use_the_request_table_but_not_delete_from_it(tmp_path):
    found = by_action(statements(tmp_path), "dynamodb:UpdateItem")
    assert len(found) == 1
    assert sorted(found[0]["Action"]) == ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Scan", "dynamodb:UpdateItem"]
    assert "FetchRequests" in json.dumps(found[0]["Resource"])
    assert not by_action(statements(tmp_path), "dynamodb:DeleteItem")


def test_runtime_gets_the_request_table_and_the_gateway(tmp_path):
    res = Template.from_stack(runtime(tmp_path)).find_resources("AWS::BedrockAgentCore::Runtime")
    env = next(iter(res.values()))["Properties"]["EnvironmentVariables"]
    assert "REQUESTS_TABLE" in env and "GATEWAY_URL" in env
    assert "TOOL_FUNCTION_NAME" not in env and "KNOWLEDGE_BASE_ID" not in env


def _sdk_call(props, phase):
    """A Custom::AWS phase (Create/Update/Delete) as its JSON text, with the runtime id token left as a marker."""
    parts = props[phase]["Fn::Join"][1]
    return json.loads("".join(p if isinstance(p, str) else "RUNTIME_ID" for p in parts))


def test_runtime_log_group_gets_a_retention_and_is_deleted_with_the_stack(tmp_path):
    """Spans hold full prompts and answers: the runtime's log group (created by AgentCore) must not keep them forever."""
    found = Template.from_stack(runtime(tmp_path)).find_resources("Custom::AWS")
    assert len(found) == 1
    props = next(iter(found.values()))["Properties"]
    name = "/aws/bedrock-agentcore/runtimes/RUNTIME_ID-DEFAULT"
    for phase in ("Create", "Update"):
        call = _sdk_call(props, phase)
        assert call["action"] == "putRetentionPolicy"
        assert call["parameters"] == {"logGroupName": name, "retentionInDays": 7}
    delete = _sdk_call(props, "Delete")
    assert delete["action"] == "deleteLogGroup" and delete["parameters"] == {"logGroupName": name}


def test_runtime_stack_owns_no_log_group_that_could_outlive_it(tmp_path):
    """Its helper Lambda logs into the audit stack's group, which expires and is never deleted under it."""
    assert not Template.from_stack(runtime(tmp_path)).find_resources("AWS::Logs::LogGroup")
    helper = Template.from_stack(EarningsAuditStack(cdk.App(), "Audit")).find_resources("AWS::Logs::LogGroup")
    named = [g for g in helper.values() if g["Properties"].get("LogGroupName") == "/earnings-research-agent/runtime-helpers"]
    assert len(named) == 1 and named[0]["Properties"]["RetentionInDays"] == 7
