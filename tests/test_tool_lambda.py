"""The Lambda tool, offline: timeout budget, validation, paging, upstream error mapping."""

import io
import json
import socket
import sys
import types
import urllib.error

import pytest

import handler

REAL_GET_CONFIG = handler.get_config  # before the autouse fixture replaces it

KEY = "test-key-123"
BASE = "https://transcripts.example"


class Ctx:
    def __init__(self, remaining_ms, per_call_ms=0):
        self.remaining_ms, self.per_call_ms = remaining_ms, per_call_ms

    def get_remaining_time_in_millis(self):
        left = self.remaining_ms
        self.remaining_ms -= self.per_call_ms   # time passes between pages
        return left


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(handler, "get_config", lambda: {"api_key": KEY, "base_url": BASE})


def turns(n, start=0, text="x"):
    return [{"speakerName": f"S{i}", "speakerRole": "CFO", "text": text} for i in range(start, start + n)]


def pages(monkeypatch, *bodies):
    """Serve the given page bodies in order; record the requests."""
    sent = []

    def urlopen(req, timeout):
        sent.append({"url": req.full_url, "auth": req.headers.get("Authorization"), "timeout": timeout})
        body = bodies[len(sent) - 1]
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(handler.urllib.request, "urlopen", urlopen)
    return sent


def http_error(code):
    return urllib.error.HTTPError(f"{BASE}/x?k={KEY}", code, "err", {}, None)


def test_http_timeout_stays_below_remaining_time():
    assert handler.http_timeout_s(30_000) == handler.MAX_HTTP_TIMEOUT_S        # capped
    assert handler.http_timeout_s(5_000) == pytest.approx(5.0 - handler.SAFETY_MARGIN_S)
    assert handler.http_timeout_s(2_200) is None                                 # no time to try


def test_no_time_left_returns_typed_error_without_calling_upstream(monkeypatch):
    monkeypatch.setattr(handler.urllib.request, "urlopen", lambda *a, **k: pytest.fail("called upstream"))
    out = handler.lambda_handler({"ticker": "NVDA", "quarter": "2026Q3"}, Ctx(1_000))
    assert out["error"]["type"] == "timeout"


@pytest.mark.parametrize("event", [{"ticker": "IBM; drop", "quarter": "2026Q3"}, {"ticker": "NVDA", "quarter": "Q3"},
                                   {"ticker": "NVDA", "quarter": "2026Q5"}])
def test_bad_parameters_are_rejected(event):
    out = handler.lambda_handler(event, Ctx(30_000))
    assert out["error"]["type"] == "bad_request"


def test_one_page_call_is_formatted_with_speaker_and_role(monkeypatch):
    sent = pages(monkeypatch, {"callDate": "2026-11-19", "data": turns(2, text="Demand was strong."), "hasMore": False})
    out = handler.lambda_handler({"ticker": "nvda", "quarter": "2026q3"}, Ctx(30_000))
    assert out["ok"] and out["call_date"] == "2026-11-19" and out["turns_total"] == 2
    assert out["transcript"].splitlines()[0] == "S0 (CFO): Demand was strong."
    assert sent[0]["url"] == f"{BASE}/v1/stocks/NVDA/earnings-calls/2026/3/speakers?limit=200&offset=0"
    assert sent[0]["auth"] == f"Bearer {KEY}"


def test_a_long_call_is_paged(monkeypatch):
    sent = pages(monkeypatch, {"data": turns(200), "hasMore": True}, {"data": turns(30, 200), "hasMore": False})
    out = handler.lambda_handler({"ticker": "NVDA", "quarter": "2026Q3"}, Ctx(30_000))
    assert out["turns_total"] == 230 and sent[1]["url"].endswith("offset=200")


def test_each_page_gets_a_timeout_from_the_time_left(monkeypatch):
    sent = pages(monkeypatch, {"data": turns(200), "hasMore": True}, {"data": turns(5, 200), "hasMore": False})
    handler.lambda_handler({"ticker": "NVDA", "quarter": "2026Q3"}, Ctx(8_000, per_call_ms=2_000))
    assert sent[0]["timeout"] > sent[1]["timeout"]


def test_running_out_of_time_mid_call_returns_what_arrived_marked_truncated(monkeypatch):
    pages(monkeypatch, {"data": turns(200), "hasMore": True})
    out = handler.lambda_handler({"ticker": "NVDA", "quarter": "2026Q3"}, Ctx(4_600, per_call_ms=2_000))
    assert out["ok"] and out["truncated"] and out["turns_total"] == 200


@pytest.mark.parametrize("code, kind, retryable", [(404, "not_found", False), (429, "rate_limited", False),
                                                   (401, "unauthorized", False), (503, "upstream_error", True)])
def test_http_errors_are_typed_and_never_leak_the_key(monkeypatch, code, kind, retryable):
    pages(monkeypatch, http_error(code))
    out = handler.lambda_handler({"ticker": "NVDA", "quarter": "2026Q3"}, Ctx(30_000))
    assert out["error"]["type"] == kind and out["error"]["retryable"] is retryable
    assert KEY not in json.dumps(out) and BASE not in json.dumps(out)


def test_empty_call_is_not_found(monkeypatch):
    pages(monkeypatch, {"data": [], "hasMore": False})
    out = handler.lambda_handler({"ticker": "NVDA", "quarter": "2026Q3"}, Ctx(30_000))
    assert out["error"]["type"] == "not_found"


def test_transcript_is_truncated_to_budget(monkeypatch):
    monkeypatch.setattr(handler, "MAX_TRANSCRIPT_CHARS", 100)
    pages(monkeypatch, {"data": turns(5, text="x" * 60), "hasMore": False})
    out = handler.lambda_handler({"ticker": "NVDA", "quarter": "2026Q3"}, Ctx(30_000))
    assert out["ok"] and out["truncated"] and len(out["transcript"]) <= 100


def test_slow_upstream_becomes_timeout_error(monkeypatch):
    pages(monkeypatch, socket.timeout("timed out"))
    out = handler.lambda_handler({"ticker": "NVDA", "quarter": "2026Q3"}, Ctx(30_000))
    assert out["error"]["type"] == "timeout" and out["error"]["retryable"]


@pytest.mark.parametrize("secret, ok", [("just-a-key", False), ('{"api_key": "k"}', False),
                                        ('{"api_key": " k ", "base_url": "https://h/"}', True)])
def test_secret_must_be_json_with_key_and_base_url(monkeypatch, secret, ok):
    class SecretsManager:
        def get_secret_value(self, SecretId):
            return {"SecretString": secret}

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda name: SecretsManager()))
    monkeypatch.setattr(handler, "_config_cache", None)
    monkeypatch.setenv("SECRET_ARN", "arn:secret")
    if ok:
        assert REAL_GET_CONFIG() == {"api_key": "k", "base_url": "https://h"}
    else:
        with pytest.raises(RuntimeError, match="must be JSON"):
            REAL_GET_CONFIG()
