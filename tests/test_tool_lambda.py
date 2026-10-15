"""The Lambda tool, offline: timeout budget, validation, upstream error mapping."""

import io
import json
import socket

import pytest

import handler


class Ctx:
    def __init__(self, remaining_ms):
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self):
        return self.remaining_ms


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setattr(handler, "get_api_key", lambda: "test-key")


def test_http_timeout_stays_below_remaining_time():
    assert handler.http_timeout_s(15_000) == handler.MAX_HTTP_TIMEOUT_S        # capped
    assert handler.http_timeout_s(5_000) == pytest.approx(5.0 - handler.SAFETY_MARGIN_S)
    assert handler.http_timeout_s(2_200) is None                                 # no time to try


def test_no_time_left_returns_typed_error_without_calling_upstream(monkeypatch):
    monkeypatch.setattr(handler.urllib.request, "urlopen", lambda *a, **k: pytest.fail("called upstream"))
    out = handler.lambda_handler({"ticker": "IBM", "quarter": "2024Q4"}, Ctx(1_000))
    assert out["error"]["type"] == "timeout"


@pytest.mark.parametrize("event", [{"ticker": "IBM; drop", "quarter": "2024Q4"}, {"ticker": "IBM", "quarter": "Q4"}])
def test_bad_parameters_are_rejected(event):
    out = handler.lambda_handler(event, Ctx(15_000))
    assert out["error"]["type"] == "bad_request"


def _respond(monkeypatch, body):
    monkeypatch.setattr(handler.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(json.dumps(body).encode()))


def test_rate_limit_note_is_an_error_not_data(monkeypatch):
    _respond(monkeypatch, {"Information": "Thank you for using Alpha Vantage! Our standard API rate limit is..."})
    out = handler.lambda_handler({"ticker": "IBM", "quarter": "2024Q4"}, Ctx(15_000))
    assert out["error"] == {"type": "rate_limited", "message": out["error"]["message"], "retryable": True}


def test_empty_transcript_is_not_found(monkeypatch):
    _respond(monkeypatch, {"symbol": "IBM", "quarter": "2024Q4", "transcript": []})
    out = handler.lambda_handler({"ticker": "IBM", "quarter": "2024Q4"}, Ctx(15_000))
    assert out["error"]["type"] == "not_found"


def test_transcript_is_truncated_to_budget(monkeypatch):
    monkeypatch.setattr(handler, "MAX_TRANSCRIPT_CHARS", 100)
    entries = [{"speaker": "CEO", "title": "Chief Executive", "content": "x" * 60}] * 5
    _respond(monkeypatch, {"symbol": "IBM", "quarter": "2024Q4", "transcript": entries})
    out = handler.lambda_handler({"ticker": "ibm", "quarter": "2024q4"}, Ctx(15_000))
    assert out["ok"] and out["truncated"]
    assert len(out["transcript"]) <= 100


def test_slow_upstream_becomes_timeout_error(monkeypatch):
    def slow(*a, **k):
        raise socket.timeout("timed out")

    monkeypatch.setattr(handler.urllib.request, "urlopen", slow)
    out = handler.lambda_handler({"ticker": "IBM", "quarter": "2024Q4"}, Ctx(15_000))
    assert out["error"]["type"] == "timeout" and out["error"]["retryable"]
