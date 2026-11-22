"""Bedrock Guardrail for the agent: the hard rule as a control outside the prompt.

Own stack, kept deployed: a guardrail costs nothing at rest (charged per text checked).

    cdk deploy EarningsResearchAgentGuardrail

The agent calls ApplyGuardrail itself (agent/guard.py), once on the question and once
on the answer, with the retrieved passages as the grounding source. Policies:

- denied topic "investment advice" (input and output) -> blocked
- prompt-attack filter (input only; Bedrock allows no output strength for it) -> blocked
- PII: contact and payment identifiers masked, in the question and the answer. Not NAME: executives' names are the speakers we cite.
- contextual grounding and relevance: detect only (action NONE). The agent flags a low score,
  like a failed citation check, and keeps the answer.

The version resource is immutable, so its description carries a hash of the policy:
change the policy and CDK publishes a new version. Replaced versions are retained
(version 1 was deleted when version 2 was published, 17 Nov, before this was set).
"""

import hashlib
import json

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
from constructs import Construct

BLOCKED_INPUT = (
    "I only report what management said on earnings calls. I can't give buy, sell or hold "
    "recommendations, price targets or investment advice. Try asking what management said about a topic."
)
BLOCKED_OUTPUT = (
    "I can't give that answer: it read as investment advice. I only report what management said "
    "on earnings calls."
)

ADVICE_TOPIC = {
    "name": "InvestmentAdvice",
    "type": "DENY",
    "definition": (
        "Recommendations on whether to buy, sell or hold a security, price targets, opinions on "
        "whether a stock is cheap or a good investment, or how to allocate a portfolio."
    ),
    "examples": [
        "Should I buy AMD stock after this call?",
        "Is Meta a good investment right now?",
        "What's a fair price target for RIVN?",
        "Should I sell my Nebius shares?",
        "How much of my portfolio should go into CRDO?",
    ],
}

PII_TYPES = [
    "EMAIL", "PHONE", "CREDIT_DEBIT_CARD_NUMBER", "CREDIT_DEBIT_CARD_CVV",
    "INTERNATIONAL_BANK_ACCOUNT_NUMBER", "US_BANK_ACCOUNT_NUMBER", "US_SOCIAL_SECURITY_NUMBER",
]

GROUNDING_THRESHOLD = 0.7
RELEVANCE_THRESHOLD = 0.5


def policy() -> dict:
    """The guardrail's policies as plain data: one source for the resource and the version hash."""
    return {
        "topic_policy_config": {"topics_config": [ADVICE_TOPIC]},
        "content_policy_config": {
            "filters_config": [{"type": "PROMPT_ATTACK", "input_strength": "HIGH", "output_strength": "NONE"}]
        },
        "sensitive_information_policy_config": {
            # input_* explicitly: with only "action" set, the deployed guardrail masked
            # output but let an email in the question through (live check, 17 Nov).
            "pii_entities_config": [
                {"type": t, "action": "ANONYMIZE", "input_action": "ANONYMIZE", "input_enabled": True,
                 "output_action": "ANONYMIZE", "output_enabled": True}
                for t in PII_TYPES
            ]
        },
        "contextual_grounding_policy_config": {
            "filters_config": [
                {"type": "GROUNDING", "threshold": GROUNDING_THRESHOLD, "action": "NONE"},
                {"type": "RELEVANCE", "threshold": RELEVANCE_THRESHOLD, "action": "NONE"},
            ]
        },
        "blocked_input_messaging": BLOCKED_INPUT,
        "blocked_outputs_messaging": BLOCKED_OUTPUT,
    }


class EarningsGuardrailStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        p = policy()
        self.guardrail = bedrock.CfnGuardrail(
            self,
            "Guardrail",
            name="earnings-research-agent",
            description="Research aid, not investment advice: blocks advice and prompt attacks, masks PII, scores grounding.",
            topic_policy_config=bedrock.CfnGuardrail.TopicPolicyConfigProperty(
                topics_config=[bedrock.CfnGuardrail.TopicConfigProperty(**t) for t in p["topic_policy_config"]["topics_config"]]
            ),
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=[
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(**f)
                    for f in p["content_policy_config"]["filters_config"]
                ]
            ),
            sensitive_information_policy_config=bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                pii_entities_config=[
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(**e)
                    for e in p["sensitive_information_policy_config"]["pii_entities_config"]
                ]
            ),
            contextual_grounding_policy_config=bedrock.CfnGuardrail.ContextualGroundingPolicyConfigProperty(
                filters_config=[
                    bedrock.CfnGuardrail.ContextualGroundingFilterConfigProperty(**f)
                    for f in p["contextual_grounding_policy_config"]["filters_config"]
                ]
            ),
            blocked_input_messaging=p["blocked_input_messaging"],
            blocked_outputs_messaging=p["blocked_outputs_messaging"],
        )

        digest = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]
        self.version = bedrock.CfnGuardrailVersion(
            self,
            "GuardrailVersion",
            guardrail_identifier=self.guardrail.attr_guardrail_id,
            description=f"policy {digest}",
        )
        self.version.apply_removal_policy(RemovalPolicy.RETAIN)  # keep earlier versions when a new one replaces it

        CfnOutput(self, "GuardrailId", value=self.guardrail.attr_guardrail_id)
        CfnOutput(self, "GuardrailVersionNumber", value=self.version.attr_version)
