#!/usr/bin/env python3
from pathlib import Path

import aws_cdk as cdk

from runtime_stack import EarningsRuntimeStack
from stack import EarningsToolStack

CODE_PATH = Path(__file__).resolve().parents[1] / "build" / "agent"
if not CODE_PATH.is_dir():
    raise SystemExit("build/agent is missing: run scripts/build_agent.sh first")

TAGS = {"project": "earnings-research-agent"}

app = cdk.App()
tool = EarningsToolStack(app, "EarningsResearchAgentTool", tags=TAGS)
EarningsRuntimeStack(
    app,
    "EarningsResearchAgentRuntime",
    code_path=CODE_PATH,
    model_id=app.node.get_context("model_id"),
    tool=tool.function,
    tags=TAGS,
)
app.synth()
