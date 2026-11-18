"""agent/session.py: follow-up resolution and session memory, offline."""

import json

import pytest

import fake_model
import session
from session import AgentCoreMemoryStore, Corpus, InMemoryStore, quarter_in, resolve
from test_answer import FakeRetriever, model_answering

CORPUS = Corpus.load()


@pytest.fixture(autouse=True)
def restore_brief(monkeypatch):
    monkeypatch.setattr(fake_model, "VALID_BRIEF", fake_model.VALID_BRIEF)


def prev(ticker="AMD", quarter="2026Q2"):
    return [{"question": "Why did gross margin expand?", "answer": "Data center mix.", "covered": True,
             "ticker": ticker, "quarter": quarter, "citations": ["AMD_2026Q2_t003_p1"]}]


# --- resolution -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("question, history, expected", [
    ("And operating margin?", prev(), ("AMD", "2026Q2", "carried")),
    ("What about Meta?", prev(), ("META", "2026Q2", "named")),            # company switches, quarter kept
    ("What did META say in Q3 2026?", prev(), ("META", "2026Q3", "named")),
    ("And in Q1 2026?", prev(), ("AMD", "2026Q1", "carried")),
    ("What about Rivian?", prev("AMD", "2025Q3"), ("RIVN", None, "named")),  # RIVN has no 2025Q3 call
    ("How did Nebius describe demand?", [], ("NBIS", None, "named")),
    ("Why did gross margin expand?", [], (None, None, "none")),
    ("Which companies talked about tariffs?", prev(), (None, None, "cross_company")),
    ("Compare AMD and Credo on gross margin", prev(), (None, None, "cross_company")),
    ("And in the latest quarter?", prev("META", "2025Q4"), ("META", "2026Q3", "carried")),
    ("And the previous quarter?", prev("AMD", "2026Q2"), ("AMD", "2026Q1", "carried")),
    ("And the quarter before?", prev("AMD", "2025Q3"), ("AMD", None, "carried")),  # nothing earlier in the corpus
    ("What did Credo say about Q1 FY2026 revenue?", [], ("CRDO", "2026Q1", "named")),
    ("And how did that compare with the quarter before?", prev("RIVN", "2026Q3"), ("RIVN", None, "cross_quarter")),
    ("Compare that with Meta", prev(), (None, None, "cross_company")),
    ("How did AMD's gross margin compare across quarters?", [], ("AMD", None, "cross_quarter")),
    ("Compare Q2 2026 with the outlook", prev(), ("AMD", "2026Q2", "carried")),   # a quarter named: pinned
    ("Compare gross margins", [], (None, None, "none")),
])
def test_follow_ups_resolve_by_the_context_rules(question, history, expected):
    c = resolve(question, {}, history, CORPUS)
    assert (c.ticker, c.quarter, c.how) == expected


def test_the_callers_ticker_and_quarter_win():
    c = resolve("What about Meta?", {"ticker": "amd", "quarter": "2025q4"}, prev(), CORPUS)
    assert (c.ticker, c.quarter, c.how) == ("AMD", "2025Q4", "payload")


def test_common_words_are_not_companies():
    assert CORPUS.named_tickers("What is the meta point about root causes and amd-like chips?") == []
    assert CORPUS.named_tickers("What did Meta's CFO say?") == ["META"]


@pytest.mark.parametrize("text, q", [("2026Q2", "2026Q2"), ("Q2 2026", "2026Q2"), ("q1 fy2026", "2026Q1"),
                                     ("FY26 Q3", "2026Q3"), ("Q4 '25", "2025Q4"), ("the second quarter", None)])
def test_quarter_forms(text, q):
    assert quarter_in(text) == q


def test_every_corpus_ticker_has_names():
    assert set(CORPUS.names) == set(CORPUS.quarters)


# --- history shown to the model -------------------------------------------------------------------------------

def test_history_block_is_short_and_most_recent_last():
    turns = [{**prev()[0], "question": f"q{i}", "answer": "x " * 400} for i in range(5)]
    block = session.history_block(turns)
    assert block.count("- Q (") == session.HISTORY_TURNS and block.index("q3") < block.index("q4")
    assert "q1" not in block and len(block) < 2000


def test_no_history_means_the_same_prompt_as_before():
    assert session.history_block([]) == ""


# --- a conversation -------------------------------------------------------------------------------------------

def run(store, question, session_id="s-1", **payload):
    r = FakeRetriever()
    out = session.run_turn({"question": question, **payload}, session_id, store, CORPUS,
                           model=model_answering(["AMD_2026Q2_t003_p1"], tool_args={"query": "margin"}), retriever=r)
    return out, r


def test_a_follow_up_searches_the_previous_call_and_sees_the_earlier_turn():
    store = InMemoryStore()
    first, _ = run(store, "Why did AMD's gross margin expand in Q2 2026?")
    assert first["context"]["how"] == "named" and first["memory"] == {"loaded": 0, "saved": True, "error": None}
    second, r = run(store, "And operating margin?")
    assert second["context"] == {"ticker": "AMD", "quarter": "2026Q2", "how": "carried", "history_turns": 1, "notes": []}
    assert (r.calls[0]["ticker"], r.calls[0]["quarter"]) == ("AMD", "2026Q2")   # pinned in code
    saved = store.load("anonymous", "s-1", 5)
    assert [t["question"] for t in saved] == ["Why did AMD's gross margin expand in Q2 2026?", "And operating margin?"]
    assert set(saved[0]) == {"question", "answer", "covered", "ticker", "quarter", "citations"}  # no passage text


def test_sessions_and_users_do_not_share_memory():
    store = InMemoryStore()
    run(store, "Why did AMD's gross margin expand in Q2 2026?", session_id="s-1")
    other, _ = run(store, "And operating margin?", session_id="s-2")
    assert other["context"]["how"] == "none"
    other_user, _ = run(store, "And operating margin?", session_id="s-1", user_id="u-2")
    assert other_user["context"]["how"] == "none"


def test_blocked_or_rejected_turns_are_not_remembered():
    store = InMemoryStore()
    out = session.run_turn({"question": ""}, "s-1", store, CORPUS)
    assert out["status"] == "bad_request" and out["memory"]["saved"] is False
    assert store.load("anonymous", "s-1", 5) == []


def test_a_memory_failure_keeps_the_answer():
    class Broken(InMemoryStore):
        def save(self, *a):
            raise RuntimeError("throttled")
    out, _ = run(Broken(), "Why did AMD's gross margin expand in Q2 2026?")
    assert out["status"] == "ok" and out["memory"]["error"] == "RuntimeError: throttled"


def test_without_a_session_the_turn_stands_alone():
    out, r = run(None, "Why did Credo's revenue grow?", session_id=None)
    assert out["memory"]["loaded"] == 0 and r.calls[0]["ticker"] == "CRDO"


# --- AgentCore Memory store -----------------------------------------------------------------------------------

class FakeMemoryClient:
    def __init__(self):
        self.events = []

    def create_event(self, **kw):
        self.events.append({"eventTimestamp": f"2026-11-18T10:00:{len(self.events):02d}",
                            "payload": [{"conversational": {"role": r, "content": {"text": t}}} for t, r in kw["messages"]],
                            "key": (kw["actor_id"], kw["session_id"])})

    def list_events(self, memory_id, actor_id, session_id, max_results):
        mine = [e for e in self.events if e["key"] == (actor_id, session_id)]
        return list(reversed(mine))  # newest first: the store must sort


def test_agentcore_store_round_trips_turns_in_order():
    s = AgentCoreMemoryStore("mem-1", FakeMemoryClient())
    for i in range(4):
        s.save("u", "s", {**prev()[0], "question": f"q{i}"})
    assert [t["question"] for t in s.load("u", "s", 3)] == ["q1", "q2", "q3"]
    assert s.load("u", "other", 3) == []


# --- guardrail on follow-ups, and what memory keeps -------------------------------------------------------------

from guard import Check  # noqa: E402

ADVICE = Check(blocked=True, blocked_by=["topic:InvestmentAdvice"], text="I only report what management said…")


class ContextGuard:
    """Blocks a bare "What about Meta?" as advice, passes it after a previous question; always blocks "buy"."""

    def __init__(self, mask=None):
        self.inputs, self.mask = [], mask

    def check_input(self, text):
        self.inputs.append(text)
        if "buy" in text or (text == "What about Meta?"):
            return ADVICE
        if self.mask and self.mask in text:
            return Check(masked=["EMAIL"], text=text.replace(self.mask, "{EMAIL}"))
        return Check()

    def check_output(self, question, answer_text, sources):
        return Check()


def guarded(store, question, guard):
    return session.run_turn({"question": question}, "s-1", store, CORPUS, guard=guard,
                            model=model_answering(["AMD_2026Q2_t003_p1"], tool_args={"query": "margin"}),
                            retriever=FakeRetriever())


def test_a_short_follow_up_is_checked_again_with_the_previous_question():
    store, g = InMemoryStore(), ContextGuard()
    guarded(store, "Why did AMD's gross margin expand in Q2 2026?", g)
    out = guarded(store, "What about Meta?", g)
    assert out["status"] == "ok" and out["context"]["ticker"] == "META"
    assert g.inputs[-1].startswith("Previous question: Why did AMD's") and "input_in_context" in out["guardrail"]


def test_a_real_advice_follow_up_stays_blocked():
    store, g = InMemoryStore(), ContextGuard()
    guarded(store, "Why did AMD's gross margin expand in Q2 2026?", g)
    out = guarded(store, "Should I buy it?", g)
    assert out["status"] == "blocked" and out["guardrail"]["input_in_context"]["blocked"]


def test_without_history_there_is_no_second_check():
    out = guarded(InMemoryStore(), "What about Meta?", g := ContextGuard())
    assert out["status"] == "blocked" and len(g.inputs) == 1


def test_memory_keeps_the_masked_question_never_the_raw_one():
    store = InMemoryStore()
    guarded(store, "Mail me at a@b.com: why did AMD's margin expand in Q2 2026?", ContextGuard(mask="a@b.com"))
    (saved,) = store.load("anonymous", "s-1", 5)
    assert "a@b.com" not in json.dumps(saved) and "{EMAIL}" in saved["question"]
