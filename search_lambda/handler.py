"""Lambda tool behind the AgentCore Gateway: search the transcript Knowledge Base.

    event:   {"query": "...", "ticker": "AMD", "quarter": "2026Q2", "k": 5}   ticker, quarter, k optional
    returns: {"ok": true, "passages": [{passage_id, ticker, quarter, speaker, role, start_seconds, score, text}]}
             or {"ok": false, "error": {"type", "message", "retryable"}}

Only the search. The rules around it stay in the agent (agent/retrieval.py), which calls this
through the Gateway: pinned ticker/quarter, the search limit, abbreviation expansion, and the
ledger of returned passage IDs that the citation check compares against. So any MCP client of
the Gateway gets plain, filtered search; the agent's safeguards don't depend on this function.

The result mapping is the same as ``agent/retrieval.bedrock_retriever`` (the direct path for
local runs); tests/test_gateway.py feeds both the same response and requires the same passages.

Standard library + boto3 only (boto3 ships with the Lambda Python runtime).
"""

from __future__ import annotations

import os
import re
from typing import Any

import boto3

MAX_K = 8
MAX_QUERY_CHARS = 1000
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
QUARTER_RE = re.compile(r"^\d{4}Q[1-4]$")

_client = None


def error(kind: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {"ok": False, "error": {"type": kind, "message": message[:300], "retryable": retryable}}


def to_passages(resp: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in resp.get("retrievalResults", []):
        m = r.get("metadata", {})
        out.append({
            "passage_id": m.get("passage_id"), "ticker": m.get("ticker"), "quarter": m.get("quarter"),
            "speaker": m.get("speaker"), "role": m.get("role"),
            "start_seconds": None if m.get("start_seconds") is None else int(m["start_seconds"]),
            "score": round(float(r.get("score", 0.0)), 4),
            "text": (r.get("content") or {}).get("text", ""),
        })
    return out


def search(query: str, ticker: str | None, quarter: str | None, k: int, kb_id: str, client: Any) -> dict[str, Any]:
    config: dict[str, Any] = {"numberOfResults": k}
    conds = [{"equals": {"key": key, "value": val}} for key, val in (("ticker", ticker), ("quarter", quarter)) if val]
    if len(conds) == 1:
        config["filter"] = conds[0]
    elif conds:
        config["filter"] = {"andAll": conds}
    resp = client.retrieve(knowledgeBaseId=kb_id, retrievalQuery={"text": query},
                           retrievalConfiguration={"vectorSearchConfiguration": config})
    return {"ok": True, "passages": to_passages(resp)}


def lambda_handler(event: dict[str, Any], context: Any = None, client: Any = None) -> dict[str, Any]:
    global _client
    event = event or {}
    query = str(event.get("query") or "").strip()
    ticker = str(event.get("ticker") or "").strip().upper() or None
    quarter = str(event.get("quarter") or "").strip().upper() or None
    if not query:
        return error("bad_request", "query is empty")
    if len(query) > MAX_QUERY_CHARS:
        return error("bad_request", f"query is longer than {MAX_QUERY_CHARS} characters")
    if ticker and not TICKER_RE.match(ticker):
        return error("bad_request", f"ticker {ticker!r} is not a ticker symbol")
    if quarter and not QUARTER_RE.match(quarter):
        return error("bad_request", f"quarter must look like 2026Q2, got {quarter!r}")
    try:
        k = max(1, min(int(event.get("k") or 5), MAX_K))
    except (TypeError, ValueError):
        return error("bad_request", "k must be a whole number from 1 to 8")
    if client is None:
        _client = _client or boto3.client("bedrock-agent-runtime")
        client = _client
    try:
        return search(query, ticker, quarter, k, os.environ["KNOWLEDGE_BASE_ID"], client)
    except Exception as exc:  # throttling, permissions, a missing KB
        name = type(exc).__name__
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", name) if hasattr(exc, "response") else name
        retryable = "Throttl" in code or "Throttl" in str(exc)
        return error("rate_limited" if retryable else "search_failed", f"{code}: {exc}", retryable)
