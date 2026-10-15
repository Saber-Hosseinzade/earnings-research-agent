#!/usr/bin/env python3
import aws_cdk as cdk

from stack import EarningsToolStack

app = cdk.App()
EarningsToolStack(
    app,
    "EarningsResearchAgentTool",
    tags={"project": "earnings-research-agent"},
)
app.synth()
