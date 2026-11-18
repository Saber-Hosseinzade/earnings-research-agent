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
            A quarter in the message ("Q2 2026", "2026Q2", "Q1 FY2026") overrides a carried one. A company
            named with no quarter in the message or the context gets its latest quarter in the corpus.
            "Latest quarter" means the latest quarter in the corpus for that company; "previous
            quarter" the one before the quarter in context. A carried quarter is dropped when the
            company changes and the new company has no call for it.
            A ticker outside the corpus (a cashtag, or a capitalised word that is not a common
            abbreviation) resolves as "unknown_ticker" and never falls back to the previous company.

clarify_reply() The clarifying question (flow 2), in code: no company is known (none in the message or the
            session, not a cross-company question) and the question points at one without naming it
            ("they", "the company"). One fixed question listing the corpus companies, no search.
            Decided 18 Nov: the model (Nova 2 Lite) ignored this rule in the prompt.

corpus_reply()  The corpus check (flow 5), in code, before the agent runs: an unknown ticker, or a quarter
            the corpus lacks for a known ticker, gets a fixed reply instead of a search ("not discussed"
            would be wrong: the call was never in the corpus). The guardrail still runs first.

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
    terms: set[str] = field(default_factory=set)   # capitalised words the transcripts use (corpus/terms.json)

    @classmethod
    def load(cls, path: Path = CALLS_FILE) -> "Corpus":
        data = json.loads(Path(path).read_text())
        quarters: dict[str, list[str]] = defaultdict(list)
        for call in data["calls"]:
            quarters[call["ticker"]].append(call["label"])
        names = {t: v for t, v in data.get("companies", {}).items() if not t.startswith("_")}
        terms_file = Path(path).with_name("terms.json")
        terms = set(json.loads(terms_file.read_text())["terms"]) if terms_file.exists() else set()
        return cls({t: sorted(qs) for t, qs in quarters.items()}, names, terms)

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


# Capitalised words that are not tickers. A capitalised 2-5 letter word not on this list, and not a corpus
# ticker, is taken as a ticker outside the corpus; so is any cashtag ($NVDA). Lowercase company names
# ("nvidia") are not recognised: the question then goes to the agent like any other.
NOT_TICKERS = {
    "AI", "AGI", "API", "APIS", "ASIC", "AWS", "CEO", "CFO", "COO", "CTO", "CPU", "CPUS", "DC", "EBIT", "EBITDA",
    "ESG", "ETF", "EU", "EV", "EVS", "FX", "FY", "GAAP", "GPU", "GPUS", "HBM", "IPO", "IR", "IT", "KPI", "LLM",
    "LLMS", "ML", "OEM", "OEMS", "PC", "PCS", "QOQ", "ROI", "ROE", "SEC", "SKU", "SKUS", "UK", "US", "USA", "USD",
    "YOY", "AND", "OR", "THE", "NOT", "OK", "ARPU", "NPS", "NOI", "EMAIL", "PHONE",
}


def unknown_tickers(text: str, corpus: "Corpus") -> list[str]:
    """Ticker-like words in the text that are not corpus tickers, in order of mention: cashtags, and
    capitalised 2-5 letter words that are not common abbreviations or terms the transcripts use
    (corpus/terms.json: AEC, EDV, EPYC...). A cashtag of a corpus term still counts ($ARM)."""
    from retrieval import ABBREVIATIONS
    skip = NOT_TICKERS | {a.upper() for a in ABBREVIATIONS} | set(corpus.quarters) | corpus.terms
    found: list[str] = []
    for m in re.finditer(r"(?<![A-Za-z0-9&$])(\$)?([A-Za-z]{1,5})(?![A-Za-z0-9&])", text):
        cash, word = m.group(1), m.group(2)
        symbol = word.upper()
        flagged = (symbol not in corpus.quarters) if cash else (word.isupper() and len(word) >= 2 and symbol not in skip)
        if flagged and symbol not in found:
            found.append(symbol)
    return found


def quarter_label(label: str) -> str:
    """'2026Q2' -> 'Q2 2026' (the company's own fiscal quarter, as in corpus/calls.json)."""
    return f"Q{label[-1]} {label[:4]}"


def _ordinal(label: str) -> int:
    return int(label[:4]) * 4 + int(label[-1])


def corpus_reply(context: "Context", corpus: "Corpus") -> dict[str, Any] | None:
    """The fixed reply when the resolved call is not in the corpus, or None when it is (flow 5).
    offer = the quarter offered instead; it is remembered, so a "yes" next resolves to it."""
    if context.ticker and context.ticker not in corpus.quarters:
        return {"kind": "unknown_ticker", "offer": None,
                "message": f"I don't have earnings calls for {context.ticker}. "
                           f"I can answer about {', '.join(corpus.quarters)}."}
    labels = corpus.quarters.get(context.ticker or "", [])
    if context.quarter and labels and context.quarter not in labels:
        target = _ordinal(context.quarter)
        nearest = min(labels, key=lambda q: (abs(_ordinal(q) - target), -_ordinal(q)))
        return {"kind": "quarter_not_in_corpus", "offer": nearest,
                "message": (f"The corpus has {context.ticker}'s calls for {quarter_label(labels[0])} to "
                            f"{quarter_label(labels[-1])}, not {quarter_label(context.quarter)}. "
                            f"Do you want the {quarter_label(nearest)} call?")}
    return None


ONE_COMPANY = re.compile(r"\b(they|their|them|theirs|the company|the company's|this company|that company|"
                         r"management|its|the CEO|the CFO)\b", re.IGNORECASE)


def clarify_reply(context: "Context", question: str, corpus: "Corpus") -> dict[str, Any] | None:
    """The clarifying question when the message needs one company and none is known (flow 2), else None."""
    if context.how != "none" or not ONE_COMPANY.search(question):
        return None
    names = list(corpus.quarters)
    return {"kind": "clarify", "offer": None,
            "message": f"Which company do you mean? I have calls for {', '.join(names[:-1])} and {names[-1]}."}


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
    how: str                                  # payload | named | unknown_ticker | carried | cross_company | cross_quarter | none
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
        unknown = unknown_tickers(question, corpus)
        if unknown and not named:   # "What about NVDA?" must not carry the previous company over
            return Context(unknown[0], quarter_in(question), "unknown_ticker", len(history))
        if unknown:
            notes.append(f"not in the corpus: {', '.join(unknown)}")
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
        if ticker and quarter is None and how == "named" and ticker in corpus.quarters:
            # "What did AMD say about margins?" with no quarter anywhere: the latest call, not all of them
            # mixed (flow 2). A comparison never gets here (cross_quarter above).
            quarter = corpus.quarters[ticker][-1]
            notes.append("no quarter given: latest quarter in the corpus")

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
    if result.get("status") not in ("ok", "out_of_corpus", "clarify", "approval_pending") or not answer:
        return None
    reply = result.get("fixed_reply") or {}
    return {"question": result.get("question") or question, "answer": answer.get("answer", ""),
            "covered": bool(answer.get("covered")),
            # An out-of-corpus quarter remembers the quarter offered instead, so "yes" resolves to it. An
            # unknown ticker is remembered too, so "Q3 2026" next can name the quarter for a fetch request.
            "ticker": context.ticker, "quarter": reply.get("offer") or context.quarter,
            "citations": list(answer.get("citations") or [])}


def _approval_reply(context: Context, payload: dict[str, Any], session_id: str | None, requests: Any,
                    corpus: Corpus, history: list[dict[str, Any]]) -> Any:
    """For answer.run: called with the question as the guardrail left it (masked), and only if the guardrail
    let it through, so an advice request is refused, not proposed, and no raw personal data is stored.

    When the ticker came from the previous turn ("Q3 2026" after "What did NVDA say about demand?"), the
    request stores the earlier question with this message in brackets: the fetch must answer the real
    question, not "Q3 2026". (Found in the live check, 18 Nov.)"""
    import approvals

    companies = list(corpus.quarters)
    if not context.quarter:
        return {"kind": "needs_quarter", "offer": None, "message": approvals.ask_for_quarter(context.ticker, companies)}

    earlier = history[-1]["question"] if context.how == "carried" and history else None   # stored masked

    def reply(masked_question: str) -> dict[str, Any]:
        question = f"{earlier} ({masked_question})" if earlier else masked_question
        return approvals.propose(requests, ticker=context.ticker, quarter=context.quarter, question=question,
                                 session_id=session_id, user_id=payload.get("user_id"), role=payload.get("role"),
                                 companies=companies)
    return reply


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
             requests: Any = None, **deps: Any) -> dict[str, Any]:
    """One conversational turn: load the session, resolve the call, answer, remember the turn.

    Without a session id or a store, the turn stands alone (no history, nothing saved), but the
    message is still resolved, so a company named in it pins the search. ``deps`` go to answer.run.
    With a request store (approvals.py), an out-of-corpus ticker becomes a fetch request for a
    reviewer once the user has named a quarter; without one, it gets the fixed out-of-corpus reply.
    """
    import answer  # here, not at the top: answer imports nothing from this module, tests patch it

    question = str(payload.get("question", "")).strip()
    actor_id = str(payload.get("user_id") or "anonymous")
    history = store.load(actor_id, session_id, HISTORY_TURNS) if (store and session_id) else []
    context = resolve(question, payload, history, corpus)
    reply: Any = corpus_reply(context, corpus) or clarify_reply(context, question, corpus)
    if reply and reply["kind"] == "unknown_ticker" and requests is not None:
        reply = _approval_reply(context, payload, session_id, requests, corpus, history)
    result = answer.run({**payload, "ticker": context.ticker, "quarter": context.quarter,
                         "history": history, "context": context.to_dict(), "fixed_reply": reply}, **deps)
    record = turn_record(question, result, context)
    result["memory"] = {"loaded": len(history), "saved": False, "error": None}
    if store and session_id and record:
        try:
            store.save(actor_id, session_id, record)
            result["memory"]["saved"] = True
        except Exception as exc:  # noqa: BLE001 - a memory failure costs the next turn its context, not this answer
            result["memory"]["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return result
