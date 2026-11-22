#!/usr/bin/env python3
import sys
from pathlib import Path

import aws_cdk as cdk

from audit_stack import EarningsAuditStack
from auth_stack import EarningsAuthStack
from gateway_stack import EarningsGatewayStack
from guardrail_stack import EarningsGuardrailStack
from kb_stack import EarningsKnowledgeBaseStack
from prompt_stack import EarningsPromptStack
from runtime_stack import EarningsRuntimeStack
from stack import EarningsToolStack
from ui_image_stack import EarningsUiImageStack
from ui_stack import EarningsUiStack

CODE_PATH = Path(__file__).resolve().parents[1] / "build" / "agent"
UI_PATH = Path(__file__).resolve().parents[1] / "ui"
# The hosted UI's endpoint hostname, known after the UI stack's first deploy. Written by scripts/deploy_ui.sh
# to a git-ignored file, so the demo URL stays out of the public repo; `-c ui_hostname=...` overrides it.
HOSTNAME_FILE = Path(__file__).resolve().parent / ".ui_hostname"

TAGS = {"project": "earnings-research-agent"}

app = cdk.App()
ui_hostname = app.node.try_get_context("ui_hostname") or (
    HOSTNAME_FILE.read_text().strip() if HOSTNAME_FILE.exists() else "")
tool = EarningsToolStack(app, "EarningsResearchAgentTool", tags=TAGS)
# Vector store: "s3vectors" (default, stays deployed) or "opensearch" (destroy after use).
# Override per command: cdk deploy EarningsResearchAgentKB -c vector_store=opensearch
kb = EarningsKnowledgeBaseStack(
    app,
    "EarningsResearchAgentKB",
    vector_store=app.node.get_context("vector_store"),
    tags=TAGS,
)
# MCP: search and fetch tools behind an AgentCore Gateway (IAM auth). No hourly charge; stays deployed.
gateway = EarningsGatewayStack(
    app, "EarningsResearchAgentGateway", tool=tool.function, knowledge_base=kb.knowledge_base, tags=TAGS
)
guard = EarningsGuardrailStack(app, "EarningsResearchAgentGuardrail", tags=TAGS)
prompts = EarningsPromptStack(app, "EarningsResearchAgentPrompts", tags=TAGS)
audit = EarningsAuditStack(app, "EarningsResearchAgentAudit", tags=TAGS)
# UI login: Cognito, groups viewer and reviewer. Stays deployed. The hosted UI's callback URLs come from context
# (localhost for local runs; the hosted UI's URL is added when the UI is deployed).
auth = EarningsAuthStack(
    app,
    "EarningsResearchAgentAuth",
    domain_prefix=app.node.get_context("auth_domain_prefix"),
    callback_urls=app.node.get_context("ui_callback_urls"),
    logout_urls=app.node.get_context("ui_logout_urls"),
    ui_hostname=ui_hostname,
    tags=TAGS,
)
# The UI image: ECR + CodeBuild (no local Docker). Stays deployed.
ui_image = EarningsUiImageStack(app, "EarningsResearchAgentUiImage", source_dir=UI_PATH, tags=TAGS)
# The runtime stack uploads build/agent as its code. Without a build it is left
# out, so the other stacks can still be deployed or destroyed.
if CODE_PATH.is_dir() and any(CODE_PATH.iterdir()):
    runtime = EarningsRuntimeStack(
        app,
        "EarningsResearchAgentRuntime",
        code_path=CODE_PATH,
        model_id=app.node.get_context("model_id"),
        gateway=gateway.gateway,
        guardrail=guard.guardrail,
        guardrail_version=guard.version,
        prompts=prompts.prompts,
        audit_log=audit.log_group,
        helper_log_group=audit.helper_log_group,
        tags=TAGS,
    )
    # The hosted UI needs the runtime: deploy after it, destroy before it.
    EarningsUiStack(
        app,
        "EarningsResearchAgentUi",
        repository=ui_image.repository,
        image_tag=ui_image.image_tag,
        runtime=runtime.runtime,
        client_id=auth.client.user_pool_client_id,
        metadata_url=auth.metadata_url,
        client_secret=auth.client_secret,
        cookie_secret=auth.cookie_secret,
        hostname=ui_hostname,
        tags=TAGS,
    )
else:
    print("note: build/agent is missing or empty, EarningsResearchAgentRuntime and EarningsResearchAgentUi skipped "
          "(run scripts/build_agent.sh to include it)", file=sys.stderr)
app.synth()
