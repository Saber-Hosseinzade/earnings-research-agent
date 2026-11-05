"""agent/ask.py (v1.1), offline."""

import json

import pytest

import ask
import fake_model
import tools
from fake_model import FakeModel

ANSWER = {"covered_in_call": True, "answer": "Revenue grew on software demand.",
          "quote": "Revenue grew on software demand.", "speaker": "CEO"}


class _Payload:
    def read(self):
        return json.dumps({"ok": True, "ticker": "IBM", "quarter": "2024Q4", "truncated": False,
                           "transcript": "CEO (Chief Executive): Revenue grew on software demand."}).encode()


@pytest.fixture(autouse=True)
def fake_lambda(monkeypatch):
    monkeypatch.setattr(tools._lambda, "invoke", lambda **kw: {"Payload": _Payload()})
    monkeypatch.setattr(fake_model, "VALID_BRIEF", ANSWER)  # what the scripted model hands back


def test_answers_a_question_with_a_quote():
    out = ask.run({"ticker": "ibm", "quarter": "2024q4", "question": "Why did revenue grow?"}, model=FakeModel("good"))
    assert out["status"] == "ok"
    assert out["answer"]["covered_in_call"] is True and out["answer"]["speaker"] == "CEO"
    assert out["run"]["tool_calls"] == 1


@pytest.mark.parametrize("payload", [
    {"ticker": "IBM", "quarter": "2024Q4"},               # no question
    {"ticker": "IBM", "quarter": "2024Q4", "question": "  "},
    {"ticker": "IBM", "quarter": "Q4", "question": "Why?"},
    {"question": "Why?"},
])
def test_bad_requests_never_reach_the_model(payload):
    out = ask.run(payload, model=FakeModel("loop"))
    assert out["status"] == "bad_request" and out["answer"] is None


def test_a_runaway_model_is_stopped_by_the_same_budget():
    out = ask.run({"ticker": "IBM", "quarter": "2024Q4", "question": "Why?"}, model=FakeModel("loop"))
    assert out["status"] == "stopped" and out["answer"] is None
