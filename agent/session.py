"""Session memory: earlier turns of a conversation, and which call a follow-up is about.

Two parts:

resolve()   Decides the ticker and quarter a message is about, in code, from the message, the caller's
            payload and the previous turn (docs/conversation_design.md, context rules):
              1. a ticker or quarter the caller passes wins ("payload");
              2. a corpus company named in the message wins ("named"); two or more named, a
                 cross-company phrase ("which companies"), or a comparison with a company other than
                 the previous one pins no company ("cross_company"); a comparison about one company
                 pins the company and no quarter, so all its calls are searched ("cross_quarter");
              3. otherwise the previous turn's ticker and quarter carry over ("carried");
              4. otherwise nothing is pinned ("none").
            A quarter in the message ("Q2 2026", "2026Q2", "Q1 FY2026") overrides a carried one.
            "Latest quarter" means the latest quarter in the corpus for that company; "previous
            quarter" the one before the quarter in context. A carried quarter is dropped when the
            company changes and the new company has no call for it.
            Companies outside the corpus are not recognised here; the corpus check handles them.

Store       Keeps one compact record per answered turn: question, answer, covered, ticker, quarter,
            citations. Never passage text, so a follow-up costs a few hundred tokens of history, not
            the passages again. AgentCoreMemoryStore writes them as short-term events (expiry set on
            the memory resource, infra/runtime_stack.py); InMemoryStore is for tests and local runs.

This is a research aid: answers can be wrong, and they are never investment advice.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

HISTORY_TURNS = 3          # turns shown to the model
ANSWER_CHARS = 400         # each earlier answer is cut to this length in the history block
_HERE = Path(__file__).resolve().parent
# Deployed, build_agent.sh puts calls.json next to this module; in the repo it is corpus/calls.json.
CALLS_FILE = Path(os.environ.get("CALLS_FILE") or
                  (_HERE / "calls.json" if (_HERE / "calls.json").exists() else _HERE.parent / "corpus" / "calls.json"))

CROSS_COMPANY = re.compile(
    r"\b(which|what|any|all|each|every|other|both)\s+(of\s+the\s+)?(companies|company's|firms|tickers)\b"
    r"|\bacross\s+(the\s+)?(companies|corpus)\b",
    re.IGNORECASE)
COMPARE = re.compile(r"\bcompar(e|ed|es|ing|ison)\b|\bversus\b|\bvs\.?(?![A-Za-z])|\bacross\s+(the\s+)?(quarters|calls)\b"
                     r"|\bover\s+the\s+(year|quarters)\b", re.IGNORECASE)
LATEST = re.compile(r"\b(latest|last|most\s+recent)\s+(quarter|call)\b", re.IGNORECASE)
PREVIOUS = re.compile(r"\b(previous|prior|preceding)\s+(quarter|call)\b|\bquarter\s+before\b", re.IGNORECASE)
_Q_FORMS = [
    re.compile(r"\b(20\d\d)\s*Q([1-4])\b", re.IGNORECASE),                       # 2026Q2
    re.compile(r"\bQ([1-4])\s*(?:FY\s*)?'?(20\d\d|\d\d)\b", re.IGNORECASE),       # Q2 2026, Q1 FY2026, Q2 '26
    re.compile(r"\bFY\s*(20\d\d|\d\d)\s*Q([1-4])\b", re.IGNORECASE),             # FY2026 Q1
]


@dataclass
class Corpus:
    quarters: dict[str, list[str]]          # ticker -> labels, oldest first
    names: dict[str, list[str]]             # ticker -> names a user may type

    @classmethod
    def load(cls, path: Path = CALLS_FILE) -> "Corpus":
        data = json.loads(Path(path).read_text())
        quarters: dict[str, list[str]] = defaultdict(list)
        for call in data["calls"]:
            quarters[call["ticker"]].append(call["label"])
        names = {t: v for t, v in data.get("companies", {}).items() if not t.startswith("_")}
        return cls({t: sorted(qs) for t, qs in quarters.items()}, names)

    def named_tickers(self, text: str) -> list[str]:
        """Corpus tickers named in the text, in order of first mention."""
        hits: list[tuple[int, str]] = []
        for ticker in self.quarters:
            for name in [ticker, *self.names.get(ticker, [])]:
                m = re.search(r"(?<![A-Za-z])" + re.escape(name) + r"(?:'s)?(?![A-Za-z])", text)
                if m:
                    hits.append((m.start(), ticker))
                    break
        return [t for _, t in sorted(hits)]


def quarter_in(text: str) -> str | None:
    for i, pat in enumerate(_Q_FORMS):
        m = pat.search(text)
        if m:
            year, q = (m.group(1), m.group(2)) if i != 1 else (m.group(2), m.group(1))
            year = year if len(year) == 4 else f"20{year}"
            return f"{year}Q{q}"
    return None


@dataclass
class Context:
    ticker: str | None
    quarter: str | None
    how: str                                  # payload | named | carried | cross_company | cross_quarter | none
    history_turns: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve(question: str, payload: dict[str, Any], history: list[dict[str, Any]], corpus: Corpus) -> Context:
    p_ticker = str(payload.get("ticker") or "").strip().upper() or None
    p_quarter = str(payload.get("quarter") or "").strip().upper() or None
    prev = history[-1] if history else {}
    notes: list[str] = []

    if p_ticker:
        ticker, quarter, how = p_ticker, p_quarter, "payload"
    else:
        named = corpus.named_tickers(question)
        compare = bool(COMPARE.search(question))
        other = prev.get("ticker") if named and prev.get("ticker") not in (None, *named) else None
        if len(named) > 1 or (not named and CROSS_COMPANY.search(question)) or (compare and other):
            return Context(None, None, "cross_company", len(history), [f"named: {named}"] if named else [])
        if compare and (named or prev.get("ticker")) and not quarter_in(question):
            # "How did that compare with the quarter before?": one company, every quarter of it.
            return Context(named[0] if named else prev["ticker"], None, "cross_quarter", len(history))
        if named:
            ticker, how = named[0], "named"
            quarter = prev.get("quarter") if prev.get("ticker") else None   # "and META?" keeps the quarter
        elif prev.get("ticker"):
            ticker, quarter, how = prev["ticker"], prev.get("quarter"), "carried"
        else:
            ticker, quarter, how = None, None, "none"

        said = quarter_in(question)
        if said:
            quarter = said
        elif ticker and LATEST.search(question):
            quarter = corpus.quarters.get(ticker, [None])[-1]
            notes.append("latest quarter in the corpus")
        elif ticker and PREVIOUS.search(question):
            labels = corpus.quarters.get(ticker, [])
            base = quarter if quarter in labels else (labels[-1] if labels else None)
            i = labels.index(base) if base in labels else 0
            quarter = labels[i - 1] if i > 0 else None
            notes.append("quarter before the one in context" if quarter else "no earlier quarter in the corpus")
        if ticker and quarter and how == "named" and not said and quarter not in corpus.quarters.get(ticker, []):
            notes.append(f"{ticker} has no {quarter} call; quarter dropped")
            quarter = None

    return Context(ticker, quarter, how, len(history), notes)


def history_block(history: list[dict[str, Any]], turns: int = HISTORY_TURNS) -> str:
    """Earlier turns as plain text for the model, most recent last. Empty when there are none."""
    if not history:
        return ""
    lines = ["Earlier in this conversation (most recent last):"]
    for t in history[-turns:]:
        where = " ".join(x for x in (t.get("ticker"), t.get("quarter")) if x) or "all calls"
        answer = (t.get("answer") or "").strip()
        if len(answer) > ANSWER_CHARS:
            answer = answer[:ANSWER_CHARS].rsplit(" ", 1)[0] + " …"
        lines += [f"- Q ({where}): {t.get('question', '').strip()}", f"  A: {answer}"]
    return "\n".join(lines) + "\n\n"


def follow_up_text(question: str, history: list[dict[str, Any]]) -> str:
    """The question with the previous one in front, for the guardrail's second look at a short follow-up."""
    return f"Previous question: {history[-1].get('question', '').strip()}\nFollow-up question: {question}"


def turn_record(question: str, result: dict[str, Any], context: Context) -> dict[str, Any] | None:
    """What is remembered from one turn, or None for a turn that should not shape the next ones
    (blocked, rejected, failed). The question is stored as the agent saw it: masked if the guardrail
    masked personal data, never the raw text."""
    answer = result.get("answer") or {}
    if result.get("status") != "ok" or not answer:
        return None
    return {"question": result.get("question") or question, "answer": answer.get("answer", ""), "covered": bool(answer.get("covered")),
            "ticker": context.ticker, "quarter": context.quarter, "citations": list(answer.get("citations") or [])}


class Store(Protocol):
    def load(self, actor_id: str, session_id: str, k: int) -> list[dict[str, Any]]: ...
    def save(self, actor_id: str, session_id: str, turn: dict[str, Any]) -> None: ...


class InMemoryStore:
    def __init__(self) -> None:
        self.turns: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    def load(self, actor_id: str, session_id: str, k: int) -> list[dict[str, Any]]:
        return [dict(t) for t in self.turns[(actor_id, session_id)][-k:]]

    def save(self, actor_id: str, session_id: str, turn: dict[str, Any]) -> None:
        self.turns[(actor_id, session_id)].append(dict(turn))


class AgentCoreMemoryStore:
    """Short-term events in AgentCore Memory: one event per turn, the question as the USER message
    and the rest of the record as JSON in the ASSISTANT message."""

    def __init__(self, memory_id: str | None = None, client: Any = None, region: str | None = None):
        self.memory_id = memory_id or os.environ["MEMORY_ID"]
        if client is None:
            from bedrock_agentcore.memory import MemoryClient
            client = MemoryClient(region_name=region or os.environ.get("AWS_REGION"))
        self.client = client

    def load(self, actor_id: str, session_id: str, k: int) -> list[dict[str, Any]]:
        # Sorted here by event time: the API's page order is not documented as oldest or newest first.
        events = self.client.list_events(memory_id=self.memory_id, actor_id=actor_id, session_id=session_id,
                                         max_results=100)
        out = []
        for event in sorted(events, key=lambda e: str(e.get("eventTimestamp", ""))):
            for item in event.get("payload", []):
                conv = item.get("conversational") or {}
                if conv.get("role") == "ASSISTANT":
                    try:
                        out.append(json.loads(conv["content"]["text"]))
                    except (KeyError, TypeError, ValueError):
                        continue
        return out[-k:]

    def save(self, actor_id: str, session_id: str, turn: dict[str, Any]) -> None:
        self.client.create_event(memory_id=self.memory_id, actor_id=actor_id, session_id=session_id,
                                 messages=[(turn["question"], "USER"), (json.dumps(turn), "ASSISTANT")])


def run_turn(payload: dict[str, Any], session_id: str | None, store: Store | None, corpus: Corpus,
             **deps: Any) -> dict[str, Any]:
    """One conversational turn: load the session, resolve the call, answer, remember the turn.

    Without a session id or a store, the turn stands alone (no history, nothing saved), but the
    message is still resolved, so a company named in it pins the search. ``deps`` go to answer.run.
    """
    import answer  # here, not at the top: answer imports nothing from this module, tests patch it

    question = str(payload.get("question", "")).strip()
    actor_id = str(payload.get("user_id") or "anonymous")
    history = store.load(actor_id, session_id, HISTORY_TURNS) if (store and session_id) else []
    context = resolve(question, payload, history, corpus)
    result = answer.run({**payload, "ticker": context.ticker, "quarter": context.quarter,
                         "history": history, "context": context.to_dict()}, **deps)
    record = turn_record(question, result, context)
    result["memory"] = {"loaded": len(history), "saved": False, "error": None}
    if store and session_id and record:
        try:
            store.save(actor_id, session_id, record)
            result["memory"]["saved"] = True
        except Exception as exc:  # noqa: BLE001 - a memory failure costs the next turn its context, not this answer
            result["memory"]["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return result
