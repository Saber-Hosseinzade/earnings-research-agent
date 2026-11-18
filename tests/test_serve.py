"""agent/serve.py, the v2 runtime entrypoint, offline."""

import pytest

import fake_model
import serve
from test_answer import FakeRetriever, model_answering
from test_audit import ListWriter
from test_guard import FakeGuard

ENV = {name: "x" for name in serve.REQUIRED_ENV}


@pytest.fixture(autouse=True)
def restore_brief(monkeypatch):
    monkeypatch.setattr(fake_model, "VALID_BRIEF", fake_model.VALID_BRIEF)


@pytest.mark.parametrize("name", serve.REQUIRED_ENV)
def test_refuses_to_start_without_guardrail_audit_log_or_kb(name):
    with pytest.raises(RuntimeError, match=name):
        serve.check_env({**ENV, name: ""})


def test_starts_with_everything_set():
    serve.check_env(ENV)


def test_session_id_is_the_runtimes_not_the_callers():
    w, g = ListWriter(), FakeGuard()
    out = serve.handle({"question": "Why?", "ticker": "AMD", "quarter": "2026Q2", "session_id": "forged"},
                       "rt-session-1", model=model_answering(["AMD_2026Q2_t003_p1"]),
                       retriever=FakeRetriever(), guard=g, audit_writer=w)
    assert out["status"] == "ok" and out["audit"]["written"]
    assert w.records[0]["session_id"] == "rt-session-1"
    assert [c[0] for c in g.calls] == ["input", "output"]  # the guardrail ran on question and answer


def test_entrypoint_takes_the_request_context():
    # BedrockAgentCoreApp passes the RequestContext only to a handler whose 2nd parameter is "context".
    assert serve.app._takes_context(serve.invoke)
