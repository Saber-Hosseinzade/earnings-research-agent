"""The transcript API used for the v2 corpus. The only provider-specific code.

Configuration (shell or .env.local; shell wins):

    TRANSCRIPT_API_KEY    your own key
    TRANSCRIPT_API_BASE   the provider's base URL (see the README, "Evaluation corpus")

To use another provider, change this file: `fetch` must return one dict per call
with `data` (a list of turns with `text`, `speakerName`, `speakerRole`),
`fiscalYear`, `fiscalQuarter` and `callDate`. eval/run_eval.py reads that shape.

Standard library only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

KEY_VAR = "TRANSCRIPT_API_KEY"
BASE_VAR = "TRANSCRIPT_API_BASE"
ENDPOINT = "/v1/stocks/{ticker}/earnings-calls/{fy}/{fq}/speakers"
SOURCE = f"transcript API {ENDPOINT}"
PAGE = 200
PAUSE_S = 0.5
HTTP_TIMEOUT_S = 30.0


def api_key() -> str:
    return os.environ.get(KEY_VAR, "").strip()


def base_url() -> str:
    return os.environ.get(BASE_VAR, "").strip().rstrip("/")


def get_page(ticker: str, fiscal_year: int, fiscal_quarter: int, offset: int,
             key: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = ENDPOINT.format(ticker=ticker, fy=fiscal_year, fq=fiscal_quarter)
    url = f"{base_url()}{path}?limit={PAGE}&offset={offset}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return json.load(resp), None
    except urllib.error.HTTPError as exc:  # never include the key in a message
        kind = {404: "not_found", 429: "rate_limited", 401: "unauthorized"}.get(exc.code, "upstream_error")
        return None, {"type": kind, "message": f"HTTP {exc.code}"}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, {"type": "upstream_error", "message": type(exc).__name__}


def fetch(ticker: str, fiscal_year: int, fiscal_quarter: int,
          key: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """All turns of one call (paged), the error if any, and requests used."""
    merged: dict[str, Any] | None = None
    offset, used = 0, 0
    while True:
        time.sleep(PAUSE_S)
        page, err = get_page(ticker, fiscal_year, fiscal_quarter, offset, key)
        used += 1
        if err is not None:
            return None, err, used
        if merged is None:
            merged = {k: v for k, v in page.items() if k not in ("data", "turnCount", "offset", "hasMore")}
            merged["data"] = []
        merged["data"].extend(page["data"])
        if not page.get("hasMore") or not page["data"]:
            break
        offset += page["turnCount"]
    merged["turnCount"] = len(merged["data"])
    return merged, None, used
