"""ui/view.py and ui/runtime_client.py: what the UI sends and shows. Offline, without Streamlit."""

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

import view  # noqa: E402
from runtime_client import RuntimeClient  # noqa: E402

DEV = {"UI_AUTH": "dev", "UI_DEV_USER": "me", "UI_DEV_ROLE": "viewer"}


# --- identity ------------------------------------------------------------------------------------------------------

def test_dev_identity_for_local_runs():
    assert view.dev_identity(DEV) == view.Identity("me", "viewer")


@pytest.mark.parametrize("env", [{}, {**DEV, "UI_AUTH": ""}, {**DEV, "UI_DEV_ROLE": "admin"}, {**DEV, "UI_DEV_USER": " "},
                                 {**DEV, "ECS_CONTAINER_METADATA_URI_V4": "http://169.254.170.2/v4/x"}])
def test_no_identity_without_dev_mode_a_valid_role_or_on_ecs(env):
    with pytest.raises(view.LoginRequired):
        view.dev_identity(env)


def test_role_and_user_come_from_the_identity_not_the_question():
    who = view.Identity("me", "viewer")
    assert view.chat_payload("  Why?  ", who) == {"question": "Why?", "user_id": "me", "role": "viewer"}
    assert view.action_payload("decide", who, request_id="r", role="reviewer")["role"] == "reviewer"  # explicit only


def test_session_ids_are_long_enough_for_agentcore_and_unique():
    a, b = view.new_session_id(), view.new_session_id()
    assert len(a) >= 33 and a != b


# --- rendering -----------------------------------------------------------------------------------------------------

PASSAGE = {"passage_id": "AMD_2026Q2_t003_p1", "ticker": "AMD", "quarter": "2026Q2", "speaker": "Jean Hu",
           "role": "CFO", "text": "Gross margin was 54%."}
OK = {"status": "ok", "answer": {"covered": True, "answer": "Margin rose on mix.", "citations": [PASSAGE["passage_id"]]},
      "citation_check": {"ok": True}, "passages": [PASSAGE], "context": {"ticker": "AMD", "quarter": "2026Q2"},
      "guardrail": {"input": {}, "output": {"grounding_flagged": False}}}


def test_covered_answer_shows_its_sources_and_the_call():
    t = view.render(OK)
    assert t.text == "Margin rose on mix." and t.notes == ["From the AMD Q2 2026 call."]
    assert [(c.label, c.text) for c in t.citations] == [("AMD Q2 2026 · Jean Hu (CFO)", "Gross margin was 54%.")]


def test_flags_become_notes():
    t = view.render({**OK, "citation_check": {"ok": False},
                     "guardrail": {"output": {"grounding_flagged": True}}})
    assert any("Citation check failed" in n for n in t.notes) and any("grounding" in n for n in t.notes)


def test_not_covered_answer_has_no_sources():
    t = view.render({**OK, "answer": {"covered": False, "answer": "Not discussed.", "citations": []}, "passages": []})
    assert t.text == "Not discussed." and t.citations == [] and t.notes == []


@pytest.mark.parametrize("status", ["blocked", "clarify", "out_of_corpus", "capped"])
def test_fixed_replies_show_their_text_only(status):
    t = view.render({"status": status, "answer": {"covered": False, "answer": "Fixed reply.", "citations": []}})
    assert (t.text, t.notes, t.citations) == ("Fixed reply.", [], [])


def test_approval_reply_remembers_the_request():
    t = view.render({"status": "approval_pending", "approval": {"request_id": "abc", "status": "pending"},
                     "answer": {"covered": False, "answer": "Sent to a reviewer.", "citations": []}})
    assert t.request_id == "abc"


@pytest.mark.parametrize("status", list(view.FAILED))
def test_failures_get_one_plain_sentence_and_the_detail_collapsed(status):
    t = view.render({"status": status, "error": "ThrottlingException: slow down", "answer": None})
    assert t.text == view.FAILED[status] and t.detail == "ThrottlingException: slow down"


def test_dollar_signs_are_not_latex():
    assert view.markdown_safe("$51 billion and $2") == "\\$51 billion and \\$2"


def test_fetched_answer_shows_the_quote_and_the_call_date():
    res = {"status": "ok", "covered": True, "answer": "Demand was strong. (fetched)", "quote": "Record revenue.",
           "speaker": "Colette Kress", "quote_in_transcript": True, "call_date": "2025-11-19", "truncated": False}
    t = view.render_fetched(res, "NVDA", "2026Q3")
    assert [(c.label, c.text) for c in t.citations] == [("NVDA Q3 2026 · Colette Kress", "Record revenue.")]
    assert t.notes == ["Fetched call held on 2025-11-19 (NVDA fiscal Q3 2026)."]


def test_fetched_quote_not_in_the_transcript_is_flagged_not_shown():
    t = view.render_fetched({"status": "ok", "answer": "x", "quote": "y", "quote_in_transcript": False}, "N", "2026Q3")
    assert t.citations == [] and "not found" in t.notes[0]


def test_request_status_lines():
    pending = {"status": "ok", "request": {"ticker": "NVDA", "quarter": "2026Q3", "status": "pending", "result": None}}
    assert view.request_status_text(pending) == ("Fetch of NVDA Q3 2026 is waiting for a reviewer.", None)
    answered = {"status": "ok", "request": {"ticker": "NVDA", "quarter": "2026Q3", "status": "answered",
                                            "result": {"status": "ok", "answer": "A.", "quote": ""}}}
    line, turn = view.request_status_text(answered)
    assert "approved" in line and turn.text == "A."
    assert view.request_status_text({"status": "not_found"})[0] == "That request could not be found."


# --- runtime client ------------------------------------------------------------------------------------------------

class FakeAgentCore:
    def __init__(self, body=b'{"status": "ok"}', fail=None):
        self.body, self.fail, self.calls = body, fail, []

    def invoke_agent_runtime(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        return {"response": io.BytesIO(self.body)}


def test_client_sends_json_to_the_one_runtime_with_the_session():
    fake = FakeAgentCore()
    out = RuntimeClient("arn:rt", client=fake).call({"question": "Why?"}, "ui-" + "a" * 32)
    assert out == {"status": "ok"}
    kw = fake.calls[0]
    assert kw["agentRuntimeArn"] == "arn:rt" and kw["runtimeSessionId"] == "ui-" + "a" * 32
    assert json.loads(kw["payload"]) == {"question": "Why?"}


@pytest.mark.parametrize("fake", [FakeAgentCore(fail=RuntimeError("AccessDenied")), FakeAgentCore(body=b"not json"),
                                  FakeAgentCore(body=b"[1]")])
def test_client_never_raises(fake):
    out = RuntimeClient("arn:rt", client=fake).call({}, "s")
    assert out["status"] == "ui_error" and out["error"]
