"""Lambda tool: fetch an earnings-call transcript excerpt from Alpha Vantage.

Design points (each one is a safeguard the agent relies on):

* The HTTP timeout is derived from the Lambda's *remaining* time, minus a
  safety margin. A slow upstream therefore returns a clean, typed error to the
  agent instead of the whole Lambda timing out and the agent seeing nothing.
* Parameters are validated before any network call.
* Alpha Vantage reports rate limits and bad symbols with HTTP 200 and a
  message field. Those are mapped to typed errors rather than passed through
  as if they were data.
* The transcript is truncated to a character budget, which bounds the tokens
  one tool call can push into the agent's context.
* The API key comes from Secrets Manager and is cached per container.

Standard library + boto3 only (boto3 ships with the Lambda Python runtime).
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_URL = "https://www.alphavantage.co/query"
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
QUARTER_RE = re.compile(r"^\d{4}Q[1-4]$")

SAFETY_MARGIN_S = float(os.environ.get("SAFETY_MARGIN_S", "2.0"))
MAX_HTTP_TIMEOUT_S = float(os.environ.get("MAX_HTTP_TIMEOUT_S", "10.0"))
MIN_HTTP_TIMEOUT_S = 0.5
MAX_TRANSCRIPT_CHARS = int(os.environ.get("MAX_TRANSCRIPT_CHARS", "12000"))

_api_key_cache: str | None = None


# ---------------------------------------------------------------- helpers


def error(kind: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {"ok": False, "error": {"type": kind, "message": message, "retryable": retryable}}


def http_timeout_s(remaining_ms: int) -> float | None:
    """Timeout for the upstream call, or None if there is no time left to try.

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
        return error("bad_request", f"ticker {ticker!r} is not a valid symbol, e.g. 'IBM'")
    if not QUARTER_RE.match(quarter):
        return error("bad_request", f"quarter {quarter!r} must look like '2024Q4'")
    return ticker, quarter


def get_api_key() -> str:
    global _api_key_cache
    if _api_key_cache is None:
        import boto3  # imported lazily so unit tests need no AWS

        secret = boto3.client("secretsmanager").get_secret_value(SecretId=os.environ["SECRET_ARN"])
        _api_key_cache = secret["SecretString"].strip()
    return _api_key_cache


def map_upstream(body: dict[str, Any], ticker: str, quarter: str) -> dict[str, Any]:
    """Turn an Alpha Vantage response into a tool result or a typed error."""
    # Alpha Vantage signals problems with HTTP 200 plus one of these keys.
    for key in ("Note", "Information"):
        if key in body:
            return error("rate_limited", str(body[key])[:300], retryable=True)
    if "Error Message" in body:
        return error("upstream_error", str(body["Error Message"])[:300])

    entries = body.get("transcript") or []
    if not entries:
        return error("not_found", f"no transcript for {ticker} {quarter}")

    lines: list[str] = []
    used = 0
    truncated = False
    for entry in entries:
        line = f"{entry.get('speaker', '?')} ({entry.get('title', '')}): {entry.get('content', '')}"
        if used + len(line) > MAX_TRANSCRIPT_CHARS:
            truncated = True
            break
        lines.append(line)
        used += len(line)

    return {
        "ok": True,
        "ticker": ticker,
        "quarter": quarter,
        "segments_returned": len(lines),
        "segments_total": len(entries),
        "truncated": truncated,
        "transcript": "\n".join(lines),
    }


def fetch(ticker: str, quarter: str, timeout_s: float, api_key: str) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": ticker, "quarter": quarter, "apikey": api_key}
    )
    try:
        with urllib.request.urlopen(f"{API_URL}?{query}", timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (socket.timeout, TimeoutError):
        return error("timeout", f"upstream did not answer within {timeout_s:.1f}s", retryable=True)
    except urllib.error.HTTPError as exc:
        return error("upstream_error", f"HTTP {exc.code}", retryable=exc.code >= 500)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            return error("timeout", f"upstream did not answer within {timeout_s:.1f}s", retryable=True)
        return error("upstream_error", f"network error: {exc.reason}", retryable=True)
    except json.JSONDecodeError:
        return error("upstream_error", "upstream returned non-JSON")
    return map_upstream(body, ticker, quarter)


# ---------------------------------------------------------------- handler


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    checked = validate(event or {})
    if isinstance(checked, dict):
        return checked
    ticker, quarter = checked

    timeout_s = http_timeout_s(context.get_remaining_time_in_millis())
    if timeout_s is None:
        return error("timeout", "not enough Lambda time left to call upstream", retryable=True)

    return fetch(ticker, quarter, timeout_s, get_api_key())
