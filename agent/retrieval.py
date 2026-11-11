"""The retrieve_passages tool: search the transcript Knowledge Base, remember what came back.

One passage is one speaker turn of one call (or part of a long turn), with an ID
like ``AMD_2026Q2_t003_p1`` that is fixed before ingestion. The tool returns the
passages with their IDs, and records every ID it returned in a ``Ledger`` for
the run. The citation check (``citations.py``) compares the answer's cited IDs
with that ledger: an ID the retriever never returned in this run is flagged,
not trusted.

Filtering: the retrieval baseline (9 Nov) found the right call at rank 1 for
only 15 of 30 questions without a filter, because the same speaker says similar
things every quarter. So when the caller already knows the ticker or quarter,
the entrypoint *pins* them: the pinned values are applied in code and the model
cannot widen the search. When nothing is pinned, the model may pass ticker and
quarter itself.

Result shape matches the Lambda tool: ``{"ok": true, ...}`` or
``{"ok": false, "error": {"type", "message", "retryable"}}``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from strands import tool

DEFAULT_K = 5
MAX_K = 8
QUARTER = re.compile(r"^\d{4}Q[1-4]$")

# retrieve(query, ticker_or_None, quarter_or_None, k) -> [{"passage_id", "ticker", "quarter", "speaker",
#                                                          "role", "start_seconds", "score", "text"}]
Retriever = Callable[[str, str | None, str | None, int], list[dict[str, Any]]]


@dataclass
class Ledger:
    """What the retriever returned during one run, in order."""

    queries: list[dict[str, Any]] = field(default_factory=list)
    returned: dict[str, dict[str, Any]] = field(default_factory=dict)  # passage_id -> metadata (no text)

    def record(self, query: str, ticker: str | None, quarter: str | None, passages: list[dict[str, Any]]) -> None:
        self.queries.append({"query": query, "ticker": ticker, "quarter": quarter,
                             "returned": [p["passage_id"] for p in passages]})
        for p in passages:
            self.returned.setdefault(p["passage_id"], {k: v for k, v in p.items() if k != "text"})


def _error(kind: str, message: str, retryable: bool = False) -> str:
    return json.dumps({"ok": False, "error": {"type": kind, "message": message[:300], "retryable": retryable}})


def make_retrieve_tool(retriever: Retriever, ledger: Ledger,
                       pinned_ticker: str | None = None, pinned_quarter: str | None = None):
    """Build the tool for one run. Pinned values override whatever the model passes."""

    @tool
    def retrieve_passages(query: str, ticker: str | None = None, quarter: str | None = None,
                          k: int = DEFAULT_K) -> str:
        """Search the earnings-call transcripts for passages relevant to a query.

        Returns JSON. On success: ok=true and passages, each with passage_id, ticker, quarter, speaker,
        role and text. Cite passages by their passage_id. On failure: ok=false with error.type and
        error.retryable. Do not retry an error that is not retryable.

        Args:
            query: What to search for, in plain words, e.g. "reasons for gross margin change".
            ticker: Limit the search to one company, e.g. "AMD". Leave empty to search all companies.
            quarter: Limit the search to one quarter, in the form YYYYQn, e.g. "2026Q2". Leave empty for all.
            k: How many passages to return, 1 to 8.
        """
        query = (query or "").strip()
        t = pinned_ticker or ((ticker or "").strip().upper() or None)
        q = pinned_quarter or ((quarter or "").strip().upper() or None)
        if not query:
            return _error("bad_request", "query is empty")
        if q and not QUARTER.match(q):
            return _error("bad_request", f"quarter must look like 2026Q2, got {q!r}")
        k = max(1, min(int(k or DEFAULT_K), MAX_K))
        try:
            passages = retriever(query, t, q, k)
        except Exception as exc:  # throttling, permissions, network
            retryable = type(exc).__name__ in {"ThrottlingException"} or "Throttl" in str(exc)
            return _error("retrieve_failed", str(exc), retryable)
        ledger.record(query, t, q, passages)
        return json.dumps({"ok": True, "filter": {"ticker": t, "quarter": q}, "passages": passages})

    return retrieve_passages


def bedrock_retriever(kb_id: str | None = None, region: str | None = None) -> Retriever:
    """Retriever backed by the Knowledge Base's Retrieve API (filter on the passage metadata)."""
    import boto3

    kb_id = kb_id or os.environ["KNOWLEDGE_BASE_ID"]
    client = boto3.client("bedrock-agent-runtime", region_name=region)

    def retrieve(query: str, ticker: str | None, quarter: str | None, k: int) -> list[dict[str, Any]]:
        search: dict[str, Any] = {"numberOfResults": k}
        conds = [{"equals": {"key": key, "value": val}} for key, val in (("ticker", ticker), ("quarter", quarter)) if val]
        if len(conds) == 1:
            search["filter"] = conds[0]
        elif conds:
            search["filter"] = {"andAll": conds}
        resp = client.retrieve(knowledgeBaseId=kb_id, retrievalQuery={"text": query},
                               retrievalConfiguration={"vectorSearchConfiguration": search})
        out = []
        for r in resp["retrievalResults"]:
            m = r.get("metadata", {})
            out.append({
                "passage_id": m.get("passage_id"), "ticker": m.get("ticker"), "quarter": m.get("quarter"),
                "speaker": m.get("speaker"), "role": m.get("role"),
                "start_seconds": None if m.get("start_seconds") is None else int(m["start_seconds"]),
                "score": round(float(r.get("score", 0.0)), 4),
                "text": (r.get("content") or {}).get("text", ""),
            })
        return out

    return retrieve
