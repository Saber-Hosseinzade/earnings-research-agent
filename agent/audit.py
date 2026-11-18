"""Audit log: one structured JSON record per turn — who asked what, what ran, what was decided.

Written by answer.run when an audit writer is passed. Deployed, records go to the CloudWatch
log group in AUDIT_LOG_GROUP (infra/audit_stack.py: own group, 30-day retention). Locally,
JsonlWriter appends to a file. Without a writer, nothing is recorded.

A record (schema_version 1) holds exactly these fields:

    schema_version, turn_id, ts                  record id and UTC time
    session_id, user_id, role                    who; session_id is the runtime's session id when deployed
                                                 (serve.py); user_id and role null until a login exists
    question                                     as the agent saw it: masked if the guardrail masked PII
    ticker, quarter                              the search filter pinned by the caller, if any
    status                                       ok | stopped | invalid_output | bad_request | blocked | guardrail_error
    tool_calls                                   each search: query, filter, passage IDs returned
    searches_refused                             searches the search limit refused
    cited, citation_check                        passage IDs the answer cited; {ok, problems, unreturned}
    guardrail                                    per check (input, output): blocked_by, masked types,
                                                 grounding and relevance scores, error, text units, latency
    approval                                     null: there is no approval gate yet (PROJECT_CONTEXT 8.11 item 2)
    covered, answer                              what the user was shown
    model_id, prompts                            model, and each prompt's source, id, version, sha256
    tokens_used, model_calls, latency_ms, final_answer, stopped_reason, error

Never in a record: raw PII the guardrail masked, passage text, the system prompt, credentials.

Writing never changes the answer. A failed write is reported in the turn's result
(audit.written=false, audit.error) so it is visible, not raised.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = 1


class Writer(Protocol):
    def write(self, record: dict[str, Any]) -> None: ...


def _guard_part(check: dict[str, Any] | None) -> dict[str, Any] | None:
    if not check:
        return None
    ground = check.get("grounding") or {}
    return {
        "blocked_by": check.get("blocked_by", []),
        "masked": check.get("masked", []),
        "grounding": (ground.get("GROUNDING") or {}).get("score"),
        "relevance": (ground.get("RELEVANCE") or {}).get("score"),
        "grounding_flagged": bool(check.get("grounding_flagged")),
        "error": check.get("error"),
        "text_units": check.get("usage"),
        "latency_ms": check.get("latency_ms"),
    }


def build(result: dict[str, Any], payload: dict[str, Any], *, turn_id: str | None = None,
          now: datetime | None = None) -> dict[str, Any]:
    """One audit record from answer.run's payload and result. Pure: no I/O."""
    g = result.get("guardrail") or {}
    gin = g.get("input") or {}
    question = str(payload.get("question", "")).strip()
    if gin.get("masked") and gin.get("text"):
        question = gin["text"]  # the raw question held PII; keep only the masked form
    run = result.get("run") or {}
    retrieval = result.get("retrieval") or {}
    check = result.get("citation_check") or None
    answer = result.get("answer") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "turn_id": turn_id or uuid.uuid4().hex,
        "ts": (now or datetime.now(timezone.utc)).isoformat(timespec="milliseconds"),
        "session_id": payload.get("session_id"),
        "user_id": payload.get("user_id"),
        "role": payload.get("role"),
        "question": question,
        "ticker": (str(payload.get("ticker") or "").strip().upper() or None),
        "quarter": (str(payload.get("quarter") or "").strip().upper() or None),
        "status": result.get("status"),
        "tool_calls": [{"query": q["query"], "ticker": q.get("ticker"), "quarter": q.get("quarter"),
                        "returned": q.get("returned", [])} for q in retrieval.get("queries", [])],
        "searches_refused": run.get("searches_refused", 0),
        "cited": answer.get("citations", []),
        "citation_check": ({k: check.get(k) for k in ("ok", "problems", "unreturned")} if check else None),
        "guardrail": ({"input": _guard_part(g.get("input")), "output": _guard_part(g.get("output"))} if g else None),
        "approval": None,
        "covered": answer.get("covered"),
        "answer": answer.get("answer"),
        "model_id": run.get("model_id"),
        "prompts": run.get("prompts"),
        "tokens_used": run.get("tokens_used"),
        "model_calls": run.get("model_calls"),
        "latency_ms": run.get("latency_ms"),
        "final_answer": bool(run.get("final_answer")),
        "stopped_reason": run.get("stopped_reason"),
        "error": result.get("error"),
    }


class JsonlWriter:
    """Append records to a local file, one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class CloudWatchWriter:
    """Put each record as one event in a CloudWatch log group, one stream per process."""

    def __init__(self, log_group: str | None = None, client: Any = None) -> None:
        self.log_group = log_group or os.environ["AUDIT_LOG_GROUP"]
        if client is None:
            import boto3

            client = boto3.client("logs")
        self.client = client
        self.stream: str | None = None

    def write(self, record: dict[str, Any]) -> None:
        if self.stream is None:
            stream = f"{datetime.now(timezone.utc):%Y/%m/%d}/{uuid.uuid4().hex[:12]}"
            self.client.create_log_stream(logGroupName=self.log_group, logStreamName=stream)
            self.stream = stream
        self.client.put_log_events(logGroupName=self.log_group, logStreamName=self.stream,
                                   logEvents=[{"timestamp": int(time.time() * 1000),
                                               "message": json.dumps(record, ensure_ascii=False)}])


def write(writer: Writer, result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Build and write the record. Returns {turn_id, written, error}; never raises."""
    record = build(result, payload)
    try:
        writer.write(record)
        return {"turn_id": record["turn_id"], "written": True, "error": None}
    except Exception as exc:  # noqa: BLE001 - an audit failure is reported on the turn, not raised
        return {"turn_id": record["turn_id"], "written": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def writer_from_env() -> Writer | None:
    """CloudWatch if AUDIT_LOG_GROUP is set, a local file if AUDIT_LOG_FILE is set, else None."""
    if os.environ.get("AUDIT_LOG_GROUP"):
        return CloudWatchWriter()
    if os.environ.get("AUDIT_LOG_FILE"):
        return JsonlWriter(os.environ["AUDIT_LOG_FILE"])
    return None
