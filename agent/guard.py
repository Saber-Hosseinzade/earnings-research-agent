"""Bedrock Guardrail checks on one turn: the question before the agent, the answer after it.

Called in code through ApplyGuardrail, not attached to the model: the agent loop makes
several model calls per turn, and only here can the retrieved passages be passed as the
grounding source. The policies live in infra/guardrail_stack.py.

What each outcome does (answer.py):
- question blocked (investment advice, prompt attack) -> no agent run, status "blocked"
- question has PII -> the agent sees the masked question
- answer blocked (reads as investment advice) -> answer replaced, status "blocked"
- answer has PII -> masked
- answer poorly grounded in the cited passages -> flagged, answer kept (like the citation check)

Fail closed on the question (an unchecked question is not answered), open on the
answer (it is returned with the error recorded), so a guardrail outage is visible.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

# A guardrail has one blocked-input message (the advice one); a prompt attack gets its own.
PROMPT_ATTACK_MESSAGE = ("I can't follow instructions to change how I work. "
                         "Ask a question about what management said on an earnings call.")
GUARD_ERROR_MESSAGE = "The safety check could not run, so this question was not answered. Please try again."


@dataclass
class Check:
    """Outcome of one ApplyGuardrail call, reduced to what the caller and the audit log need."""

    blocked: bool = False
    blocked_by: list[str] = field(default_factory=list)  # e.g. "topic:InvestmentAdvice", "content:PROMPT_ATTACK"
    masked: list[str] = field(default_factory=list)  # PII types masked
    text: str | None = None  # replacement text: the blocked message, or the masked input/answer
    grounding: dict[str, Any] | None = None  # {"GROUNDING": {"score", "threshold", "flagged"}, "RELEVANCE": {...}}
    error: str | None = None
    usage: dict[str, int] | None = None  # ApplyGuardrail text units per policy (what is billed)
    latency_ms: int | None = None

    @property
    def grounding_flagged(self) -> bool:
        return bool(self.grounding) and any(f["flagged"] for f in self.grounding.values())

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "grounding_flagged": self.grounding_flagged}


class Guard(Protocol):
    def check_input(self, question: str) -> Check: ...

    def check_output(self, question: str, answer: str, sources: list[str]) -> Check: ...


def parse(resp: dict[str, Any]) -> Check:
    """Reduce an ApplyGuardrail response. Decides from the assessments, not the overall action:
    detect-only filters (grounding) report without intervening, PII masking intervenes without blocking."""
    check = Check()
    grounding: dict[str, Any] = {}
    for a in resp.get("assessments", []):
        for t in a.get("topicPolicy", {}).get("topics", []):
            if t.get("action") == "BLOCKED":
                check.blocked_by.append(f"topic:{t.get('name')}")
        for f in a.get("contentPolicy", {}).get("filters", []):
            if f.get("action") == "BLOCKED":
                check.blocked_by.append(f"content:{f.get('type')}")
        sip = a.get("sensitiveInformationPolicy", {})
        for e in sip.get("piiEntities", []):
            if e.get("action") == "BLOCKED":
                check.blocked_by.append(f"pii:{e.get('type')}")
            elif e.get("action") == "ANONYMIZED":
                check.masked.append(e.get("type"))
        for f in a.get("contextualGroundingPolicy", {}).get("filters", []):
            score, threshold = f.get("score"), f.get("threshold")
            flagged = f.get("detected") if f.get("detected") is not None else (
                score is not None and threshold is not None and score < threshold)
            grounding[f.get("type")] = {"score": score, "threshold": threshold, "flagged": bool(flagged)}
    check.blocked = bool(check.blocked_by)
    check.usage = {k: v for k, v in (resp.get("usage") or {}).items() if isinstance(v, int) and v} or None
    check.grounding = grounding or None
    outputs = resp.get("outputs") or []
    if resp.get("action") == "GUARDRAIL_INTERVENED" and outputs and (check.blocked or check.masked):
        check.text = outputs[0].get("text")
    return check


class BedrockGuard:
    """Guard backed by ApplyGuardrail on one guardrail version."""

    def __init__(self, guardrail_id: str | None = None, version: str | None = None, client: Any = None) -> None:
        self.guardrail_id = guardrail_id or os.environ["GUARDRAIL_ID"]
        self.version = version or os.environ["GUARDRAIL_VERSION"]
        if client is None:
            import boto3

            client = boto3.client("bedrock-runtime")
        self.client = client

    def _apply(self, source: str, content: list[dict[str, Any]]) -> Check:
        started = time.perf_counter()
        try:
            resp = self.client.apply_guardrail(guardrailIdentifier=self.guardrail_id, guardrailVersion=self.version,
                                               source=source, content=content)
            check = parse(resp)
        except Exception as exc:  # noqa: BLE001 - any failure is recorded; the caller decides open or closed
            check = Check(error=f"{type(exc).__name__}: {str(exc)[:200]}")
        check.latency_ms = round((time.perf_counter() - started) * 1000)
        return check

    def check_input(self, question: str) -> Check:
        return self._apply("INPUT", [{"text": {"text": question}}])

    def check_output(self, question: str, answer: str, sources: list[str]) -> Check:
        content: list[dict[str, Any]] = [{"text": {"text": answer, "qualifiers": ["guard_content"]}}]
        if sources:  # grounding needs a source; a "not discussed" answer has none, so only the topic/PII checks run
            content = [{"text": {"text": "\n\n".join(sources), "qualifiers": ["grounding_source"]}},
                       {"text": {"text": question, "qualifiers": ["query"]}}] + content
        return self._apply("OUTPUT", content)


def guard_from_env() -> Guard | None:
    """The deployed guardrail if GUARDRAIL_ID is set, else None (offline runs and the pre-guardrail baselines)."""
    return BedrockGuard() if os.environ.get("GUARDRAIL_ID") else None
