"""Lambda tool: fetch one earnings-call transcript from the third-party transcript API.

The same API the evaluation corpus was pulled from (scripts/transcript_source.py), so a live
fetch reads the same kind of text the agent is evaluated on. The provider is not named in code:
its base URL and your own key live in the Lambda's secret.

    event:   {"ticker": "NVDA", "quarter": "2026Q3"}     quarter = the company's own fiscal quarter
    returns: {"ok": true, ticker, quarter, call_date, turns_returned, turns_total, truncated, transcript}
             or {"ok": false, "error": {"type", "message", "retryable"}}

Design points (each one is a safeguard the agent relies on):

* The HTTP timeout of every page is derived from the Lambda's *remaining* time, minus a
  safety margin. A slow upstream returns a clean, typed error instead of the Lambda timing out.
* Parameters are validated before any network call.
* HTTP errors are mapped to typed errors (not_found, rate_limited, unauthorized, upstream_error)
  rather than passed through. A 429 is not retryable: the free tier's cap is per day.
* A call is paged (up to PAGE turns per request); every page is one request against the quota.
* The transcript is truncated to a character budget, which bounds the tokens one tool call can
  push into the agent's context.
* Key and base URL come from Secrets Manager (a JSON secret), cached per container, and never
  appear in a message.

Standard library + boto3 only (boto3 ships with the Lambda Python runtime).
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any

ENDPOINT = "/v1/stocks/{ticker}/earnings-calls/{fy}/{fq}/speakers"
PAGE = 200
MAX_PAGES = 5
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")

SAFETY_MARGIN_S = float(os.environ.get("SAFETY_MARGIN_S", "2.0"))
MAX_HTTP_TIMEOUT_S = float(os.environ.get("MAX_HTTP_TIMEOUT_S", "10.0"))
MIN_HTTP_TIMEOUT_S = 0.5
MAX_TRANSCRIPT_CHARS = int(os.environ.get("MAX_TRANSCRIPT_CHARS", "100000"))

_config_cache: dict[str, str] | None = None


# ---------------------------------------------------------------- helpers


def error(kind: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {"ok": False, "error": {"type": kind, "message": message, "retryable": retryable}}


def http_timeout_s(remaining_ms: int) -> float | None:
    """Timeout for the next upstream request, or None if there is no time left to try.

    Kept strictly below the Lambda's remaining time so the function can still
    return a typed error after the HTTP call gives up.
    """
    budget = remaining_ms / 1000.0 - SAFETY_MARGIN_S
    if budget < MIN_HTTP_TIMEOUT_S:
        return None
    return min(MAX_HTTP_TIMEOUT_S, budget)


def validate(event: dict[str, Any]) -> tuple[str, str] | dict[str, Any]:
    ticker = str(event.get("ticker", "")).strip().upper()
    quarter = str(event.get("quarter", "")).strip().upper()
    if not TICKER_RE.match(ticker):
        return error("bad_request", f"ticker {ticker!r} is not a valid symbol, e.g. 'NVDA'")
    if not QUARTER_RE.match(quarter):
        return error("bad_request", f"quarter {quarter!r} must be the fiscal quarter, like '2026Q3'")
    return ticker, quarter


def get_config() -> dict[str, str]:
    """{"api_key", "base_url"} from the JSON secret, set out of band."""
    global _config_cache
    if _config_cache is None:
        import boto3  # imported lazily so unit tests need no AWS

        raw = boto3.client("secretsmanager").get_secret_value(SecretId=os.environ["SECRET_ARN"])["SecretString"]
        try:
            value = json.loads(raw)
            _config_cache = {"api_key": str(value["api_key"]).strip(), "base_url": str(value["base_url"]).strip().rstrip("/")}
        except (ValueError, KeyError, TypeError):
            raise RuntimeError('the secret must be JSON: {"api_key": "...", "base_url": "https://..."}') from None
    return _config_cache


def get_page(url: str, key: str, timeout_s: float) -> dict[str, Any]:
    """One page as parsed JSON, or a typed error (which has "ok": False)."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (socket.timeout, TimeoutError):
        return error("timeout", f"upstream did not answer within {timeout_s:.1f}s", retryable=True)
    except urllib.error.HTTPError as exc:  # never include the key or URL in a message
        if exc.code == 404:
            return error("not_found", "no transcript for that ticker and fiscal quarter")
        if exc.code == 429:
            return error("rate_limited", "the transcript API's request quota is used up", retryable=False)
        if exc.code in (401, 403):
            return error("unauthorized", f"HTTP {exc.code}: check the API key in the secret")
        return error("upstream_error", f"HTTP {exc.code}", retryable=exc.code >= 500)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            return error("timeout", f"upstream did not answer within {timeout_s:.1f}s", retryable=True)
        return error("upstream_error", "network error", retryable=True)
    except json.JSONDecodeError:
        return error("upstream_error", "upstream returned non-JSON")


def to_result(turns: list[dict[str, Any]], meta: dict[str, Any], ticker: str, quarter: str) -> dict[str, Any]:
    if not turns:
        return error("not_found", f"no transcript for {ticker} {quarter}")
    lines: list[str] = []
    used = 0
    truncated = False
    for turn in turns:
        line = f"{turn.get('speakerName') or '?'} ({turn.get('speakerRole') or ''}): {turn.get('text') or ''}"
        if used + len(line) > MAX_TRANSCRIPT_CHARS:
            truncated = True
            break
        lines.append(line)
        used += len(line)
    return {
        "ok": True,
        "ticker": ticker,
        "quarter": quarter,
        "call_date": meta.get("callDate"),
        "turns_returned": len(lines),
        "turns_total": len(turns),
        "truncated": truncated,
        "transcript": "\n".join(lines),
    }


def fetch(ticker: str, quarter: str, context: Any, config: dict[str, str]) -> dict[str, Any]:
    fy, fq = QUARTER_RE.match(quarter).groups()
    path = ENDPOINT.format(ticker=ticker, fy=fy, fq=fq)
    turns: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    offset = 0
    for _ in range(MAX_PAGES):
        timeout_s = http_timeout_s(context.get_remaining_time_in_millis())
        if timeout_s is None:
            if turns:  # what arrived so far is still a usable (partial) transcript
                return {**to_result(turns, meta, ticker, quarter), "truncated": True}
            return error("timeout", "not enough Lambda time left to call upstream", retryable=True)
        page = get_page(f"{config['base_url']}{path}?limit={PAGE}&offset={offset}", config["api_key"], timeout_s)
        if page.get("ok") is False:
            return page
        meta = meta or {k: v for k, v in page.items() if k not in ("data", "turnCount", "offset", "hasMore")}
        data = page.get("data") or []
        turns.extend(data)
        if not page.get("hasMore") or not data:
            break
        offset += len(data)
    return to_result(turns, meta, ticker, quarter)


# ---------------------------------------------------------------- handler


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    checked = validate(event or {})
    if isinstance(checked, dict):
        return checked
    ticker, quarter = checked
    if http_timeout_s(context.get_remaining_time_in_millis()) is None:
        return error("timeout", "not enough Lambda time left to call upstream", retryable=True)
    return fetch(ticker, quarter, context, get_config())
