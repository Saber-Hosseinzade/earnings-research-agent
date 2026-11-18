"""Approval gate (flow 6), offline: proposing, deciding, the single fetch, and the reviewer actions."""

import json
from datetime import datetime, timedelta, timezone

import pytest

import approvals
import fake_model
import serve
import session
from approvals import InMemoryRequests
from fake_model import FakeModel
from guard import Check
from test_answer import FakeRetriever
from test_audit import ListWriter
from test_session import ContextGuard, NoModel

CORPUS = session.Corpus.load()
T0 = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
TRANSCRIPT = "Jensen (CEO): Data center demand was extraordinary this quarter.\nColette (CFO): Revenue grew strongly."


@pytest.fixture(autouse=True)
def restore_brief(monkeypatch):
    monkeypatch.setattr(fake_model, "VALID_BRIEF", fake_model.VALID_BRIEF)


class Fetcher:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"ok": True, "ticker": "NVDA", "quarter": "2026Q3", "call_date": "2026-11-19",
                                 "truncated": False, "transcript": TRANSCRIPT}

    def __call__(self, ticker, quarter):
        self.calls.append((ticker, quarter))
        return self.result


def answering(quote="Data center demand was extraordinary this quarter.", covered=True):
    fake_model.VALID_BRIEF = {"covered_in_call": covered, "answer": "Management called data center demand extraordinary.",
                              "quote": quote, "speaker": "Jensen"}
    return FakeModel("good", fetches=0)


def turn(requests, question, session_id="s-1", store=None, user_id="u-viewer", guard=None):
    return session.run_turn({"question": question, "user_id": user_id, "role": "viewer"}, session_id,
                            store if store is not None else session.InMemoryStore(), CORPUS, requests=requests,
                            guard=guard, model=NoModel(), retriever=FakeRetriever())


def proposed(requests, **kw):
    out = turn(requests, "What did NVDA say about data center demand in Q3 2026?", **kw)
    return out, out["approval"]["request_id"]


# --- proposing ------------------------------------------------------------------------------------------------

def test_an_out_of_corpus_ticker_with_a_quarter_becomes_a_pending_request():
    r = InMemoryRequests()
    out, rid = proposed(r)
    assert out["status"] == "approval_pending" and out["approval"] == {"request_id": rid, "status": "pending", "new": True}
    item = r.get(rid)
    assert (item["ticker"], item["quarter"], item["status"], item["requested_by"]) == ("NVDA", "2026Q3", "pending", "u-viewer")
    assert "once a reviewer approves" in out["answer"]["answer"]


def test_without_a_quarter_the_user_is_asked_for_one_and_nothing_is_stored():
    r = InMemoryRequests()
    out = turn(r, "What did NVDA say about data center demand?")
    assert out["status"] == "out_of_corpus" and "If you name a quarter" in out["answer"]["answer"] and r.items == {}


def test_naming_the_quarter_next_proposes_the_fetch():
    r, store = InMemoryRequests(), session.InMemoryStore()
    turn(r, "What did NVDA say about data center demand?", store=store)
    out = turn(r, "Q3 2026", store=store)
    assert out["status"] == "approval_pending" and r.pending()[0]["quarter"] == "2026Q3"
    # the request carries the real question, not just "Q3 2026" (found in the live check)
    assert r.pending()[0]["question"] == "What did NVDA say about data center demand? (Q3 2026)"


def test_asking_again_does_not_duplicate_the_request():
    r = InMemoryRequests()
    _, rid = proposed(r)
    again, rid2 = proposed(r)
    assert rid2 == rid and again["approval"]["new"] is False and len(r.items) == 1


def test_advice_about_an_unknown_ticker_is_refused_not_proposed():
    r = InMemoryRequests()
    out = turn(r, "Should I buy NVDA after Q3 2026?", guard=ContextGuard())
    assert out["status"] == "blocked" and r.items == {}


def test_the_stored_question_is_masked():
    r = InMemoryRequests()
    turn(r, "Mail a@b.com: what did NVDA say about demand in Q3 2026?", guard=ContextGuard(mask="a@b.com"))
    (item,) = r.items.values()
    assert "a@b.com" not in json.dumps(item) and "{EMAIL}" in item["question"]


def test_without_a_request_store_the_old_reply_stands():
    out = session.run_turn({"question": "What did NVDA say in Q3 2026?"}, "s-1", None, CORPUS, model=NoModel())
    assert out["status"] == "out_of_corpus" and out["answer"]["answer"].startswith("I don't have earnings calls for NVDA")


# --- deciding -------------------------------------------------------------------------------------------------

def test_nothing_is_fetched_before_approval_and_a_decline_fetches_nothing():
    r, f = InMemoryRequests(), Fetcher()
    _, rid = proposed(r)
    assert f.calls == []
    d = approvals.decide(r, rid, "decline", user_id="u-rev", role="reviewer", fetch=f)
    assert d.status == "declined" and f.calls == [] and r.get(rid)["decided_by"] == "u-rev"


def test_a_declined_request_is_not_proposed_again_in_the_session():
    r = InMemoryRequests()
    _, rid = proposed(r)
    approvals.decide(r, rid, "decline", user_id="u-rev", role="reviewer", fetch=Fetcher())
    out, _ = proposed(r)
    assert out["approval"]["status"] == "declined" and "declined" in out["answer"]["answer"] and len(r.items) == 1


@pytest.mark.parametrize("user_id, role, why", [("u-viewer2", "viewer", "only a reviewer"),
                                                ("u-viewer", "reviewer", "their own request"),
                                                ("u-rev", None, "only a reviewer")])
def test_only_another_reviewer_may_decide(user_id, role, why):
    r, f = InMemoryRequests(), Fetcher()
    _, rid = proposed(r)
    d = approvals.decide(r, rid, "approve", user_id=user_id, role=role, fetch=f)
    assert d.status == "refused" and why in d.message and f.calls == [] and r.get(rid)["status"] == "pending"


def test_approval_fetches_once_and_answers_with_a_checked_quote():
    r, f = InMemoryRequests(), Fetcher()
    _, rid = proposed(r)
    d = approvals.decide(r, rid, "approve", user_id="u-rev", role="reviewer", fetch=f, model=answering())
    assert d.status == "approved" and f.calls == [("NVDA", "2026Q3")]
    result = r.get(rid)["result"]
    assert r.get(rid)["status"] == "answered" and result["covered"] and result["quote_in_transcript"] is True
    assert result["answer"].endswith(approvals.SOURCE_NOTE)


def test_an_invented_quote_is_flagged():
    r = InMemoryRequests()
    _, rid = proposed(r)
    approvals.decide(r, rid, "approve", user_id="u-rev", role="reviewer", fetch=Fetcher(),
                     model=answering(quote="Demand will double next year."))
    assert r.get(rid)["result"]["quote_in_transcript"] is False


def test_a_request_is_decided_once():
    r, f = InMemoryRequests(), Fetcher()
    _, rid = proposed(r)
    approvals.decide(r, rid, "approve", user_id="u-rev", role="reviewer", fetch=f, model=answering())
    again = approvals.decide(r, rid, "approve", user_id="u-rev2", role="reviewer", fetch=f, model=answering())
    assert again.status == "conflict" and "already answered" in again.message and len(f.calls) == 1


def test_an_expired_request_cannot_be_approved():
    r, f = InMemoryRequests(), Fetcher()
    _, rid = proposed(r)
    d = approvals.decide(r, rid, "approve", user_id="u-rev", role="reviewer", fetch=f,
                         now=datetime.now(timezone.utc) + timedelta(hours=25))
    assert d.status == "conflict" and "expired" in d.message and f.calls == []


def test_a_failed_fetch_is_recorded_not_answered():
    r = InMemoryRequests()
    _, rid = proposed(r)
    f = Fetcher({"ok": False, "error": {"type": "not_found", "message": "no transcript", "retryable": False}})
    d = approvals.decide(r, rid, "approve", user_id="u-rev", role="reviewer", fetch=f, model=NoModel())
    assert "fetch failed: not_found" in d.message and r.get(rid)["status"] == "failed"


def test_the_answer_goes_through_the_output_guardrail():
    class BlockingGuard(ContextGuard):
        def check_output(self, question, answer_text, sources):
            self.sources = sources
            return Check(blocked=True, blocked_by=["topic:InvestmentAdvice"], text="I can't give that answer.")
    r, g = InMemoryRequests(), BlockingGuard()
    _, rid = proposed(r)
    approvals.decide(r, rid, "approve", user_id="u-rev", role="reviewer", fetch=Fetcher(), model=answering(), guard=g)
    result = r.get(rid)["result"]
    assert result["covered"] is False and result["answer"].startswith("I can't give that answer.")
    assert g.sources == ["Data center demand was extraordinary this quarter."]   # the verified quote


# --- the reviewer actions on the runtime -------------------------------------------------------------------------

def deps(**kw):
    return {"guard": None, "audit_writer": ListWriter(), "store": session.InMemoryStore(), "corpus": CORPUS,
            "requests": InMemoryRequests(), "fetch": Fetcher(), "model": NoModel(), "retriever": FakeRetriever(), **kw}


def test_serve_end_to_end_request_list_approve_status():
    d = deps()
    asked = serve.handle({"question": "What did NVDA say about data center demand in Q3 2026?", "user_id": "u-viewer",
                          "role": "viewer"}, "rt-1", **d)
    rid = asked["approval"]["request_id"]
    assert serve.handle({"action": "list_requests", "role": "viewer"}, "rt-2", **d)["status"] == "refused"
    listed = serve.handle({"action": "list_requests", "role": "reviewer"}, "rt-2", **d)
    assert [x["request_id"] for x in listed["requests"]] == [rid] and "session_id" not in listed["requests"][0]
    decided = serve.handle({"action": "decide", "request_id": rid, "decision": "approve", "user_id": "u-rev",
                            "role": "reviewer"}, "rt-2", **{**d, "model": answering()})
    assert decided["status"] == "approved" and decided["result"]["covered"]
    # the requester sees it; someone else doesn't; the requester's session remembers it
    mine = serve.handle({"action": "request_status", "request_id": rid, "user_id": "u-viewer"}, "rt-1", **d)
    assert mine["request"]["status"] == "answered" and mine["request"]["result"]["answer"]
    assert serve.handle({"action": "request_status", "request_id": rid, "user_id": "u-other"}, "rt-3", **d) == {"status": "not_found"}
    assert d["store"].load("u-viewer", "rt-1", 5)[-1]["ticker"] == "NVDA"
    # audit: the turn and the decision, with who decided
    recs = d["audit_writer"].records
    assert recs[0]["approval"]["status"] == "pending"
    assert recs[1]["status"] == "decision:approved" and recs[1]["approval"]["decided_by"] == "u-rev"


def test_unknown_action_is_a_bad_request():
    assert serve.handle({"action": "delete_everything"}, "rt", **deps())["status"] == "bad_request"
