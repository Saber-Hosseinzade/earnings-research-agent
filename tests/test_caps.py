"""agent/caps.py: spend caps per session and per day, and their wiring in serve.py. Offline."""

from datetime import datetime, timezone

import pytest

import approvals
import caps
import serve
import session
from test_answer import FakeRetriever, model_answering
from test_audit import ListWriter
from test_guard import FakeGuard

NOW = datetime(2026, 11, 20, 23, 59, tzinfo=timezone.utc)


def test_counts_until_the_session_limit_then_refuses():
    c, limits = caps.InMemoryCounter(), caps.Limits(per_session=2, per_day=10)
    assert [caps.take(c, "s1", limits, NOW).allowed for _ in range(3)] == [True, True, False]
    v = caps.take(c, "s1", limits, NOW)
    assert (v.cap, v.limit) == ("session", 2) and "new conversation" in v.message
    assert caps.take(c, "s2", limits, NOW).allowed  # another session still runs


def test_day_limit_holds_across_sessions_and_resets_at_midnight_utc():
    c, limits = caps.InMemoryCounter(), caps.Limits(per_session=10, per_day=2)
    assert caps.take(c, "a", limits, NOW).allowed and caps.take(c, "b", limits, NOW).allowed
    v = caps.take(c, "c", limits, NOW)
    assert (v.allowed, v.cap) == (False, "day") and "midnight UTC" in v.message
    assert caps.take(c, "c", limits, datetime(2026, 11, 21, 0, 1, tzinfo=timezone.utc)).allowed


def test_a_refused_turn_moves_no_counter():
    c, limits = caps.InMemoryCounter(), caps.Limits(per_session=5, per_day=1)
    caps.take(c, "a", limits, NOW)
    caps.take(c, "b", limits, NOW)  # refused by the day cap
    assert c.counts.get("session#b") is None


def test_counter_failure_refuses():
    class Broken:
        def take(self, keys):
            raise RuntimeError("table gone")
    v = caps.take(Broken(), "s", caps.Limits(), NOW)
    assert not v.allowed and "table gone" in v.error and "could not check" in v.message


def test_limits_from_env():
    assert caps.Limits.from_env({"TURNS_PER_SESSION": "3", "TURNS_PER_DAY": "7"}) == caps.Limits(3, 7)
    assert caps.Limits.from_env({}) == caps.Limits(caps.TURNS_PER_SESSION, caps.TURNS_PER_DAY)


class FakeDynamo:
    """transact_write_items with the condition evaluated, all or none."""

    class exceptions:
        class TransactionCanceledException(Exception):
            def __init__(self, reasons):
                super().__init__("cancelled")
                self.response = {"CancellationReasons": reasons}

    def __init__(self):
        self.items, self.calls = {}, []

    def transact_write_items(self, TransactItems):
        self.calls.append(TransactItems)
        reasons, ok = [], True
        for t in TransactItems:
            u = t["Update"]
            key, limit = u["Key"]["counter"]["S"], int(u["ExpressionAttributeValues"][":limit"]["N"])
            passes = self.items.get(key, {}).get("n", 0) < limit
            reasons.append({"Code": "None" if passes else "ConditionalCheckFailed"})
            ok = ok and passes
        if not ok:
            raise self.exceptions.TransactionCanceledException(reasons)
        for t in TransactItems:
            u = t["Update"]
            item = self.items.setdefault(u["Key"]["counter"]["S"], {"n": 0})
            item["n"] += 1
            item.setdefault("ttl", int(u["ExpressionAttributeValues"][":ttl"]["N"]))


def test_dynamo_counter_names_the_key_at_its_limit():
    db = FakeDynamo()
    c = caps.DynamoCounter(table="t", client=db)
    limits = caps.Limits(per_session=1, per_day=5)
    assert caps.take(c, "s", limits, NOW).allowed
    assert caps.take(c, "s", limits, NOW).cap == "session"
    assert db.items["day#2026-11-20"]["n"] == 1  # the refused turn did not count
    update = db.calls[0][0]["Update"]
    assert update["TableName"] == "t" and "n < :limit" in update["ConditionExpression"]
    # day counters expire a day after the day ends
    assert db.items["day#2026-11-20"]["ttl"] == int(datetime(2026, 11, 22, tzinfo=timezone.utc).timestamp())


def _deps(**extra):
    return {**dict(model=model_answering(["AMD_2026Q2_t003_p1"]), retriever=FakeRetriever(), guard=FakeGuard(),
                   audit_writer=ListWriter(), store=session.InMemoryStore(), corpus=session.Corpus.load(),
                   requests=approvals.InMemoryRequests(), fetch=lambda t, q: pytest.fail("fetched")), **extra}


def test_capped_turn_runs_nothing_and_is_audited():
    counter = caps.InMemoryCounter()
    q = {"question": "Why?", "ticker": "AMD", "quarter": "2026Q2"}
    limits = caps.Limits(per_session=1, per_day=10)
    assert serve.handle(q, "rt-1", **_deps(counter=counter, limits=limits))["status"] == "ok"
    deps = _deps(counter=counter, limits=limits, model=None)
    deps["guard"] = g = FakeGuard()
    out = serve.handle(q, "rt-1", **deps)
    assert out["status"] == "capped" and out["cap"]["cap"] == "session"
    assert g.calls == []  # not even the guardrail ran
    assert deps["audit_writer"].records[0]["status"] == "capped"


def test_reads_are_not_counted():
    counter = caps.InMemoryCounter()
    out = serve.handle({"action": "list_requests", "role": "reviewer"}, "rt-1", **_deps(counter=counter))
    assert out["status"] == "ok" and counter.counts == {}


def test_decisions_are_counted():
    counter = caps.InMemoryCounter()
    serve.handle({"action": "decide", "role": "reviewer", "request_id": "nope", "decision": "approve"}, "rt-1",
                 **_deps(counter=counter))
    assert counter.counts["session#rt-1"] == 1
