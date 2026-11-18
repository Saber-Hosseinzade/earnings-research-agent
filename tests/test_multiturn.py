"""eval/run_multiturn.py: the multi-turn set and its checks, offline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run_multiturn as mt  # noqa: E402
import session  # noqa: E402

KEYS = {"status", "how", "ticker", "quarter", "covered", "cites", "cites_only", "mentions", "clarifies"}
CORPUS = session.Corpus.load()


def test_set_is_well_formed():
    convs = mt.load()
    assert len({c["id"] for c in convs}) == len(convs)
    for c in convs:
        assert c["flow"] in {1, 2, 3, 4, 5, 6, 7} and c["turns"]
        for t in c["turns"]:
            if "action" in t:
                assert t["action"] in {"decide", "request_status", "list_requests"} and t.get("as", "user") in mt.ACTORS
                assert set(t["expect"]) <= {"status", "request_status", "covered", "quote_ok", "fetches"}
                continue
            assert t["user"].strip() and set(t["expect"]) <= KEYS
            for p in t["expect"].get("cites", []) + t["expect"].get("cites_only", []):
                ticker = p.split("_")[0]
                assert ticker in CORPUS.quarters, p


def ok(**kw):
    out = {"status": "ok", "answer": {"covered": True, "answer": "Gross margin rose.", "citations": ["AMD_2026Q2_t003"]},
           "context": {"how": "named", "ticker": "AMD", "quarter": "2026Q2"}, "retrieval": {"queries": [{"query": "q"}]}}
    out.update(kw)
    return out


def test_a_matching_turn_passes():
    assert mt.check({"status": "ok", "ticker": "AMD", "quarter": "2026Q2", "covered": True, "cites": ["AMD_2026Q2"],
                     "cites_only": ["AMD_"], "mentions": "margin"}, ok()) == []


@pytest.mark.parametrize("expect, failed", [
    ({"quarter": None}, ["quarter"]),
    ({"cites": ["AMD_2026Q1"]}, ["cites:AMD_2026Q1"]),
    ({"cites_only": ["META_"]}, ["cites_only"]),
    ({"mentions": "expense"}, ["mentions"]),
    ({"clarifies": True}, ["clarifies"]),
    ({"status": "blocked", "covered": False}, ["status", "covered"]),
])
def test_each_check_can_fail(expect, failed):
    assert mt.check(expect, ok()) == failed


def test_a_clarifying_question_passes():
    out = ok(answer={"covered": False, "answer": "Which company do you mean?", "citations": []}, retrieval={"queries": []})
    assert mt.check({"clarifies": True}, out) == []


def test_score_counts_goal_success_and_refusals():
    convs = [{"id": "A", "flow": 4, "turns": [{"user": "q", "expect": {"status": "ok"}},
                                              {"user": "buy?", "expect": {"status": "blocked"}}]},
             {"id": "B", "flow": 4, "turns": [{"user": "q", "expect": {"status": "ok"}},
                                              {"action": "decide", "expect": {"status": "refused"}}]}]
    runs = {"A": [{"failed": [], "status": "ok"}, {"failed": [], "status": "blocked"}],
            "B": [{"failed": ["status"], "status": "blocked"}, {"failed": [], "status": "refused"}]}
    s = mt.score(convs, runs)
    assert (s["goal_success"], s["turns_passed"], s["turns"]) == (1, 3, 4)
    assert s["refusal"] == {"expected_blocked": 1, "blocked": 1, "expected_not_blocked": 2, "wrongly_blocked": 1}
    assert s["by_flow"]["4"] == {"conversations": 2, "passed": 1}


def test_flow6_conversation_runs_offline_with_the_fixture_fetch(monkeypatch):
    """The reviewer steps, the single fetch and their checks, with a scripted model and no AWS."""
    import fake_model
    from fake_model import FakeModel
    monkeypatch.setattr(fake_model, "VALID_BRIEF", {"covered_in_call": True, "answer": "Gross margin was 41%.",
                                                    "quote": "Gross margin was 41%, down one point", "speaker": "Ben Kato"})
    import approvals
    real = approvals.answer_from_transcript
    monkeypatch.setattr(approvals, "answer_from_transcript", lambda q, f, model=None: real(q, f, FakeModel("good", fetches=0)))
    conv = next(c for c in mt.load() if c["id"] == "MT-16")
    verdicts, _ = mt.run_conversation(conv, CORPUS, guard=None)
    assert [v["failed"] for v in verdicts] == [[], [], [], [], []]
    assert verdicts[-1]["fetches"] == 1 and verdicts[1]["fetches"] == 0
