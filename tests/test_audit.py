"""agent/audit.py and its use in answer.run, offline."""

import json
from datetime import datetime, timezone

import pytest

import answer
import audit
import fake_model
from guard import Check
from test_answer import FakeRetriever, model_answering
from test_guard import FakeGuard

Q = {"question": "Why did gross margin expand?", "ticker": "amd", "quarter": "2026q2",
     "session_id": "s-1", "user_id": "u-1", "role": "viewer"}


@pytest.fixture(autouse=True)
def restore_brief(monkeypatch):
    monkeypatch.setattr(fake_model, "VALID_BRIEF", fake_model.VALID_BRIEF)


class ListWriter:
    def __init__(self, fail=None):
        self.records, self.fail = [], fail

    def write(self, record):
        if self.fail:
            raise self.fail
        self.records.append(json.loads(json.dumps(record)))  # must be JSON-serialisable


def test_record_has_exactly_the_documented_fields():
    w = ListWriter()
    answer.run(Q, model=model_answering(["AMD_2026Q2_t003_p1"]), retriever=FakeRetriever(), audit_writer=w)
    (r,) = w.records
    assert set(r) == {
        "schema_version", "turn_id", "ts", "session_id", "user_id", "role", "question", "ticker", "quarter",
        "status", "tool_calls", "searches_refused", "cited", "citation_check", "guardrail", "context", "approval",
        "covered", "answer", "model_id", "prompts", "tokens_used", "model_calls", "latency_ms", "final_answer",
        "stopped_reason", "error"}
    for name in r:  # the module docstring lists every field
        assert name in audit.__doc__


def test_record_says_who_asked_what_and_what_ran():
    w = ListWriter()
    out = answer.run(Q, model=model_answering(["AMD_2026Q2_t003_p1"]), retriever=FakeRetriever(), audit_writer=w)
    (r,) = w.records
    assert (r["session_id"], r["user_id"], r["role"]) == ("s-1", "u-1", "viewer")
    assert (r["question"], r["ticker"], r["quarter"]) == ("Why did gross margin expand?", "AMD", "2026Q2")
    assert r["status"] == "ok" and r["cited"] == ["AMD_2026Q2_t003_p1"] and r["citation_check"]["ok"]
    assert r["tool_calls"][0]["returned"] == out["retrieval"]["returned"]
    assert r["prompts"]["answer_system"]["source"] == "file" and r["approval"] is None
    assert out["audit"] == {"turn_id": r["turn_id"], "written": True, "error": None}


def test_record_holds_no_passage_text():
    w = ListWriter()
    answer.run(Q, model=model_answering(["AMD_2026Q2_t003_p1"]), retriever=FakeRetriever(), audit_writer=w)
    assert "Can you talk about margins?" not in json.dumps(w.records[0])  # text of a returned, uncited passage


def test_masked_question_is_logged_never_the_raw_one():
    w = ListWriter()
    g = FakeGuard(inp=Check(masked=["EMAIL"], text="Mail {EMAIL}. Why did margin expand?"))
    answer.run({**Q, "question": "Mail a@b.co. Why did margin expand?"}, model=model_answering(["AMD_2026Q2_t003_p1"]),
               retriever=FakeRetriever(), guard=g, audit_writer=w)
    blob = json.dumps(w.records[0])
    assert "a@b.co" not in blob and w.records[0]["question"] == "Mail {EMAIL}. Why did margin expand?"
    assert w.records[0]["guardrail"]["input"]["masked"] == ["EMAIL"]


def test_blocked_and_rejected_turns_are_recorded_too():
    w = ListWriter()
    g = FakeGuard(inp=Check(blocked=True, blocked_by=["topic:InvestmentAdvice"], text="No advice."))
    answer.run({"question": "Should I buy AMD?"}, model=model_answering([]), retriever=FakeRetriever(), guard=g,
               audit_writer=w)
    answer.run({"question": " "}, audit_writer=w)
    assert [r["status"] for r in w.records] == ["blocked", "bad_request"]
    assert w.records[0]["guardrail"]["input"]["blocked_by"] == ["topic:InvestmentAdvice"]
    assert w.records[0]["tool_calls"] == []


def test_a_failed_write_is_reported_not_raised():
    out = answer.run(Q, model=model_answering(["AMD_2026Q2_t003_p1"]), retriever=FakeRetriever(),
                     audit_writer=ListWriter(fail=RuntimeError("throttled")))
    assert out["status"] == "ok" and out["audit"]["written"] is False and "throttled" in out["audit"]["error"]


def test_without_a_writer_there_is_no_audit_key():
    out = answer.run(Q, model=model_answering(["AMD_2026Q2_t003_p1"]), retriever=FakeRetriever())
    assert "audit" not in out


def test_guardrail_part_keeps_scores_units_and_latency():
    part = audit._guard_part({"blocked_by": [], "masked": [], "grounding_flagged": True, "error": None,
                              "grounding": {"GROUNDING": {"score": 0.5}, "RELEVANCE": {"score": 0.9}},
                              "usage": {"contextualGroundingPolicyUnits": 3}, "latency_ms": 210, "text": "x"})
    assert part == {"blocked_by": [], "masked": [], "grounding": 0.5, "relevance": 0.9, "grounding_flagged": True,
                    "error": None, "text_units": {"contextualGroundingPolicyUnits": 3}, "latency_ms": 210}


def test_build_is_deterministic_given_id_and_time():
    t = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
    r = audit.build({"status": "bad_request"}, {"question": "x"}, turn_id="abc", now=t)
    assert r["turn_id"] == "abc" and r["ts"] == "2026-11-17T12:00:00.000+00:00"


class FakeLogs:
    def __init__(self):
        self.streams, self.events = [], []

    def create_log_stream(self, **kw):
        self.streams.append(kw)

    def put_log_events(self, **kw):
        self.events.append(kw)


def test_cloudwatch_writer_one_stream_per_process():
    client = FakeLogs()
    w = audit.CloudWatchWriter("/g", client)
    w.write({"a": 1})
    w.write({"a": 2})
    assert len(client.streams) == 1 and client.streams[0]["logGroupName"] == "/g"
    assert [json.loads(e["logEvents"][0]["message"])["a"] for e in client.events] == [1, 2]
    assert {e["logStreamName"] for e in client.events} == {client.streams[0]["logStreamName"]}


def test_jsonl_writer_appends(tmp_path):
    w = audit.JsonlWriter(tmp_path / "a" / "audit.jsonl")
    w.write({"n": 1})
    w.write({"n": 2})
    assert [json.loads(line)["n"] for line in (tmp_path / "a" / "audit.jsonl").read_text().splitlines()] == [1, 2]
