"""AgentCore Runtime entrypoint for v2: one question per invocation, answered by answer.run.

    payload: {"question": "...", "ticker": "AMD", "quarter": "2026Q2", "user_id": ..., "role": ...}
             ticker, quarter, user_id and role are optional (see answer.py)
    returns: answer.run's result, with the audit outcome

Deployed, the guardrail and the audit log are required, not optional: the process refuses to
start without GUARDRAIL_ID, GUARDRAIL_VERSION and AUDIT_LOG_GROUP, so a misconfigured runtime
cannot answer unchecked or unlogged. Prompts come from Bedrock Prompt Management (prompts.py
fails rather than fall back to a file when PROMPT_SOURCE=bedrock).

The session id in the audit record is the runtime's session id (runtimeSessionId), set by
AgentCore, not one the caller puts in the payload.

main.py (v1, one brief per call) stays as it was, so the v1.0 baseline remains reproducible.
This is a research aid: answers can be wrong, and they are never investment advice.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp

import answer
import audit
from guard import BedrockGuard

REQUIRED_ENV = ("MODEL_ID", "KNOWLEDGE_BASE_ID", "GUARDRAIL_ID", "GUARDRAIL_VERSION", "AUDIT_LOG_GROUP")


def check_env(environ: dict[str, str] | None = None) -> None:
    environ = os.environ if environ is None else environ
    missing = [name for name in REQUIRED_ENV if not environ.get(name)]
    if missing:
        raise RuntimeError(f"serve.py needs {', '.join(missing)}; refusing to start without them")


@lru_cache(maxsize=1)
def _deployed() -> dict[str, Any]:
    """Built once per process: the audit writer keeps one log stream per process (audit.py)."""
    return {"guard": BedrockGuard(), "audit_writer": audit.CloudWatchWriter()}


def handle(payload: dict[str, Any], session_id: str | None, **deps: Any) -> dict[str, Any]:
    """One turn. ``deps`` (model, retriever, guard, audit_writer) are for tests; deployed, they come from the env."""
    payload = {**payload, "session_id": session_id}  # the runtime's session id, never the caller's
    if "guard" not in deps or "audit_writer" not in deps:
        deps = {**_deployed(), **deps}
    return answer.run(payload, **deps)


app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict[str, Any], context: Any) -> dict[str, Any]:
    return handle(payload, getattr(context, "session_id", None))


if __name__ == "__main__":
    check_env()
    app.run()
