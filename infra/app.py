#!/usr/bin/env python3
import sys
from pathlib import Path

import aws_cdk as cdk

from audit_stack import EarningsAuditStack
from gateway_stack import EarningsGatewayStack
from guardrail_stack import EarningsGuardrailStack
from kb_stack import EarningsKnowledgeBaseStack
from prompt_stack import EarningsPromptStack
from runtime_stack import EarningsRuntimeStack
from stack import EarningsToolStack

CODE_PATH = Path(__file__).resolve().parents[1] / "build" / "agent"

TAGS = {"project": "earnings-research-agent"}

app = cdk.App()
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
# The runtime stack uploads build/agent as its code. Without a build it is left
# out, so the other stacks can still be deployed or destroyed.
if CODE_PATH.is_dir() and any(CODE_PATH.iterdir()):
    EarningsRuntimeStack(
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
else:
    print("note: build/agent is missing or empty, EarningsResearchAgentRuntime skipped "
          "(run scripts/build_agent.sh to include it)", file=sys.stderr)
app.synth()
