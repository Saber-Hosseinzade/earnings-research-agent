"""Bedrock Prompt Management: the agent's prompts as numbered, immutable versions.

Own stack, kept deployed (no cost at rest; the model calls are what is billed).

    cdk deploy EarningsResearchAgentPrompts

Source of truth is agent/prompts/<name>.txt, reviewed in git. Each file becomes a
prompt with one TEXT variant and no model: the model stays config (MODEL_ID), not
part of the prompt. The runtime loads the pinned version (agent/prompts.py).

A version is a snapshot of the prompt at the time it is created. Its description
carries a hash of the text, so changed text publishes the next version on deploy.
Replaced versions are retained, not deleted, so the numbered history stays in AWS:
version 1 of answer_system is the v2.0 baseline prompt, version 2 is prompt v2.
"""

import hashlib
from pathlib import Path

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
from constructs import Construct

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "agent" / "prompts"
PROMPT_NAMES = ("answer_system", "answer_final")
VARIANT = "default"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class EarningsPromptStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, prompts_dir: Path = PROMPTS_DIR, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # name -> (prompt, pinned version)
        self.prompts: dict[str, tuple[bedrock.CfnPrompt, bedrock.CfnPromptVersion]] = {}
        for name in PROMPT_NAMES:
            text = (prompts_dir / f"{name}.txt").read_text()
            logical = "".join(part.title() for part in name.split("_"))
            prompt = bedrock.CfnPrompt(
                self,
                f"{logical}Prompt",
                name=f"earnings-{name.replace('_', '-')}",
                description=f"agent/prompts/{name}.txt. Research aid, not investment advice.",
                default_variant=VARIANT,
                variants=[
                    bedrock.CfnPrompt.PromptVariantProperty(
                        name=VARIANT,
                        template_type="TEXT",
                        template_configuration=bedrock.CfnPrompt.PromptTemplateConfigurationProperty(
                            text=bedrock.CfnPrompt.TextPromptTemplateConfigurationProperty(text=text)
                        ),
                    )
                ],
            )
            version = bedrock.CfnPromptVersion(
                self,
                f"{logical}Version",
                prompt_arn=prompt.attr_arn,
                description=f"sha256 {text_hash(text)}",
            )
            version.apply_removal_policy(RemovalPolicy.RETAIN)  # keep the history when a new version replaces it
            self.prompts[name] = (prompt, version)
            CfnOutput(self, f"{logical}PromptId", value=prompt.attr_id)
            CfnOutput(self, f"{logical}PromptVersion", value=version.attr_version)
