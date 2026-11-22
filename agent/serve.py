"""AgentCore Runtime entrypoint for v2: one question per invocation, answered by answer.run.

    payload: {"question": "...", "ticker": "AMD", "quarter": "2026Q2", "user_id": ..., "role": ...}
             ticker, quarter, user_id and role are optional (see answer.py); follow-ups in the same
             runtime session resolve against earlier turns (session.py)
    returns: answer.run's result, with context (how the call was resolved), memory and the audit outcome

Deployed, the guardrail and the audit log are required, not optional: the process refuses to
start without GUARDRAIL_ID, GUARDRAIL_VERSION, AUDIT_LOG_GROUP and MEMORY_ID, so a misconfigured runtime
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

import approvals
import audit
import caps
import session
from guard import BedrockGuard

# GATEWAY_URL: deployed, search and fetch go through the AgentCore Gateway (gateway.py), never directly.
# USAGE_TABLE: the spend caps' counters (caps.py).
REQUIRED_ENV = ("MODEL_ID", "GATEWAY_URL", "GUARDRAIL_ID", "GUARDRAIL_VERSION", "AUDIT_LOG_GROUP", "MEMORY_ID",
                "REQUESTS_TABLE", "USAGE_TABLE")
REQUEST_FIELDS = ("request_id", "status", "ticker", "quarter", "question", "requested_by", "created_at", "expires_at",
                  "decided_by", "decided_at")  # what a reviewer's list shows: no answers, no session ids


def check_env(environ: dict[str, str] | None = None) -> None:
    environ = os.environ if environ is None else environ
    missing = [name for name in REQUIRED_ENV if not environ.get(name)]
    if missing:
        raise RuntimeError(f"serve.py needs {', '.join(missing)}; refusing to start without them")


@lru_cache(maxsize=1)
def _deployed() -> dict[str, Any]:
    """Built once per process: the audit writer keeps one log stream per process (audit.py)."""
    return {"guard": BedrockGuard(), "audit_writer": audit.CloudWatchWriter(),
            "store": session.AgentCoreMemoryStore(), "corpus": session.Corpus.load(),
            "requests": approvals.DynamoRequests(), "fetch": approvals.default_fetcher(),
            "counter": caps.DynamoCounter(), "limits": caps.Limits.from_env()}


DEPS = ("guard", "audit_writer", "store", "corpus", "requests", "fetch")


def handle(payload: dict[str, Any], session_id: str | None, **deps: Any) -> dict[str, Any]:
    """One request. ``deps`` (model, retriever, guard, audit_writer, store, corpus, requests, fetch, counter,
    limits) are for tests; deployed, they come from the env. Without a counter, nothing is capped (tests only:
    deployed, the counter always comes with the other deps).

    payload.action:
      (none)          a chat turn (session.run_turn); earlier turns come from AgentCore Memory
      list_requests   reviewer only: pending fetch requests (approvals.py)
      decide          reviewer only: {"request_id", "decision": "approve" | "decline"}; approval fetches and answers
      request_status  the requester or a reviewer: status of one request, with the answer once there is one
    A chat turn and a decision count against the spend caps (caps.py) before anything runs; a capped one
    returns status "capped" and gets an audit record.
    user_id and role come from the caller; behind the UI they will come from the login, checked server-side.
    """
    payload = {**payload, "session_id": session_id}  # the runtime's session id, never the caller's
    if any(k not in deps for k in DEPS):
        deps = {**_deployed(), **deps}
    store, corpus, requests, fetch = (deps.pop(k) for k in ("store", "corpus", "requests", "fetch"))
    counter, limits = deps.pop("counter", None), deps.pop("limits", None)
    action = payload.get("action")
    if action in (None, "", "decide") and counter is not None:
        verdict = caps.take(counter, session_id, limits)
        if not verdict.allowed:
            return _capped(payload, verdict, deps)
    if not action:
        return session.run_turn(payload, session_id, store, corpus, requests=requests, **deps)
    if action == "list_requests":
        if payload.get("role") != approvals.REVIEWER:
            return {"status": "refused", "error": "only a reviewer can list fetch requests"}
        items = sorted(requests.pending(), key=lambda i: i["created_at"])
        return {"status": "ok", "requests": [{k: i.get(k) for k in REQUEST_FIELDS} for i in items]}
    if action == "decide":
        return _decide(payload, store, requests, fetch, deps)
    if action == "request_status":
        item = requests.get(str(payload.get("request_id") or ""))
        if not item or (payload.get("role") != approvals.REVIEWER
                        and item["requested_by"] != (payload.get("user_id") or "anonymous")):
            return {"status": "not_found"}  # the same reply whether it doesn't exist or isn't yours
        return {"status": "ok", "request": {**{k: item.get(k) for k in REQUEST_FIELDS}, "result": item.get("result")}}
    return {"status": "bad_request", "error": f"unknown action {action!r}"}


def _capped(payload: dict[str, Any], verdict: caps.Verdict, deps: dict[str, Any]) -> dict[str, Any]:
    out = {"status": "capped", "question": str(payload.get("question", "")).strip(),
           "answer": {"covered": False, "answer": verdict.message, "citations": []},
           "cap": verdict.to_dict(), "citation_check": None, "retrieval": None, "run": {}, "guardrail": None,
           "error": verdict.error}
    if deps.get("audit_writer") is not None:
        out["audit"] = audit.write(deps["audit_writer"], out, payload)
    return out


def _decide(payload: dict[str, Any], store: Any, requests: Any, fetch: Any, deps: dict[str, Any]) -> dict[str, Any]:
    request_id = str(payload.get("request_id") or "")
    d = approvals.decide(requests, request_id, str(payload.get("decision") or ""), user_id=payload.get("user_id"),
                         role=payload.get("role"), fetch=fetch, model=deps.get("model"), guard=deps.get("guard"))
    item = d.request or {}
    result = item.get("result") or {}
    approval = {"request_id": request_id, "status": d.status, "decided_by": item.get("decided_by"),
                "request_status": item.get("status"), "ticker": item.get("ticker"), "quarter": item.get("quarter")}
    out = {"status": d.status, "message": d.message, "approval": approval,
           "request": {k: item.get(k) for k in REQUEST_FIELDS} if item else None, "result": result or None}
    # The requester's session gets the answer, so their follow-ups can build on it.
    if store and result.get("answer") and item.get("session_id"):
        try:
            store.save(item["requested_by"], item["session_id"],
                       {"question": item["question"], "answer": result["answer"], "covered": result.get("covered"),
                        "ticker": item["ticker"], "quarter": item["quarter"], "citations": []})
        except Exception as exc:  # noqa: BLE001
            out["memory_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if deps.get("audit_writer") is not None:
        record = {"status": f"decision:{d.status}", "approval": approval, "question": item.get("question"),
                  "answer": {"covered": result.get("covered"), "answer": result.get("answer"), "citations": []},
                  "run": result.get("run") or {}, "error": None if d.status in ("approved", "declined") else d.message}
        out["audit"] = audit.write(deps["audit_writer"], record,
                                   {**payload, "question": item.get("question"), "ticker": item.get("ticker"),
                                    "quarter": item.get("quarter")})
    return out


app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict[str, Any], context: Any) -> dict[str, Any]:
    return handle(payload, getattr(context, "session_id", None))


if __name__ == "__main__":
    check_env()
    app.run()
