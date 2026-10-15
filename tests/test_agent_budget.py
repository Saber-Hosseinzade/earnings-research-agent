"""The agent loop, offline: the happy path, and proof that the budget stops a runaway agent."""

import json

import pytest

import main
import tools
from budget import RunBudget
from fake_model import FakeModel


class _Payload:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body


@pytest.fixture(autouse=True)
def fake_lambda(monkeypatch):
    body = json.dumps({"ok": True, "ticker": "IBM", "quarter": "2024Q4", "truncated": False,
                       "transcript": "CEO (Chief Executive): Revenue grew on software demand."}).encode()
    calls = []

    def invoke(**kwargs):
        calls.append(kwargs)
        return {"Payload": _Payload(body)}

    monkeypatch.setattr(tools._lambda, "invoke", invoke)
    return calls


def test_happy_path_returns_a_valid_brief(fake_lambda):
    out = main.run({"ticker": "ibm", "quarter": "2024q4"}, model=FakeModel("good"))
    assert out["status"] == "ok"
    assert out["brief"]["ticker"] == "IBM"
    assert out["run"]["stopped_reason"] is None
    assert len(fake_lambda) == 1


def test_runaway_agent_is_stopped_by_model_call_limit(fake_lambda):
    model = FakeModel("loop")
    budget = RunBudget(max_model_calls=4, max_tool_calls=2, max_total_tokens=10**9)
    out = main.run({"ticker": "IBM", "quarter": "2024Q4"}, model=model, budget=budget)

    assert out["status"] == "stopped"
    assert out["run"]["stopped_reason"] == "max_model_calls"
    assert model.calls <= 4                      # the hard ceiling held
    assert len(fake_lambda) <= 2                 # past the tool limit, calls were cancelled
    assert out["run"]["tools_cancelled"] >= 1


def test_runaway_agent_is_stopped_by_token_limit(fake_lambda):
    model = FakeModel("loop", tokens_per_call=5_000)
    budget = RunBudget(max_model_calls=50, max_tool_calls=50, max_total_tokens=12_000)
    out = main.run({"ticker": "IBM", "quarter": "2024Q4"}, model=model, budget=budget)

    assert out["status"] == "stopped"
    assert out["run"]["stopped_reason"] == "max_total_tokens"
    assert model.calls <= 3


def test_bad_request_never_reaches_the_model():
    model = FakeModel("good")
    out = main.run({"ticker": "IBM", "quarter": "last quarter"}, model=model)
    assert out["status"] == "bad_request"
    assert model.calls == 0
