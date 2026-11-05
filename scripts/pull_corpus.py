#!/usr/bin/env python3
"""Pull the fixed v2 corpus: 6 tickers x 4 quarters of earnings-call transcripts.

One raw JSON per call goes to corpus/raw/ (git-ignored: Alpha Vantage terms and
company copyright). corpus/manifest.json (committed) records, per call, the
label requested, fetch time, segment and character counts and the sha256 of the
raw file, so the corpus is reproducible without publishing it.

Usage (from the repo root):

    export ALPHAVANTAGE_API_KEYS=key1,key2    # one or more; never commit them
                                              # (or put the line in .env.local)
    python scripts/pull_corpus.py             # fetches only calls not yet on disk
    python scripts/pull_corpus.py --force     # refetch everything
    python scripts/pull_corpus.py --dry-run   # list what would be fetched

Keys are used in order; a key that hits the daily cap is dropped and the next
one takes over. Calls already in corpus/raw/ are skipped, so a run that stops on
the daily cap can be resumed the next day without spending requests twice.

Gaps are printed and recorded in the manifest, not filled automatically: the
quarter table in PROJECT_CONTEXT.md is the source of truth, so a fallback to an
older quarter is a decision made there, then reflected in CORPUS below.

Label check: the period the call states about itself ("third quarter 2026",
"first quarter of fiscal 2027") is read from the opening segments and compared
with the expected period. That is how the CRDO fiscal-vs-calendar question is
answered: a "mismatch" or "unknown" needs a look at the raw file.

Standard library only, plus tool_lambda/handler.py for the response mapping, so
the corpus and the v1 tool classify Alpha Vantage replies the same way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool_lambda"))
from handler import API_URL, map_upstream  # noqa: E402

ENV_FILE = ROOT / ".env.local"   # git-ignored via .env.*; see .env.local.example
RAW_DIR = ROOT / "corpus" / "raw"
MANIFEST = ROOT / "corpus" / "manifest.json"
SOURCE = "Alpha Vantage EARNINGS_CALL_TRANSCRIPT"
FROZEN = "2026-11-05"  # calls reported after this date are not added

HTTP_TIMEOUT_S = 30.0
PAUSE_S = 1.5          # between requests; the free tier also has a burst limit
BURST_RETRIES = 3


@dataclass(frozen=True)
class Call:
    ticker: str
    label: str           # the `quarter` value sent to Alpha Vantage
    fiscal_quarter: int  # the period the call should state about itself
    fiscal_year: int

    @property
    def path(self) -> Path:
        return RAW_DIR / f"{self.ticker}_{self.label}.json"


def _calendar(ticker: str, quarters: list[tuple[int, int]]) -> list[Call]:
    return [Call(ticker, f"{y}Q{q}", q, y) for y, q in quarters]


_LAST_FOUR = [(2025, 4), (2026, 1), (2026, 2), (2026, 3)]
# Latest call not on Alpha Vantage as of 5 Nov: next-older quarter used (§8.3).
_FALLBACK = [(2025, 3), (2025, 4), (2026, 1), (2026, 2)]

# PROJECT_CONTEXT.md §8.3. Keep in sync with that table.
CORPUS: list[Call] = [
    *_calendar("AMD", _FALLBACK),   # 2026Q3 not returned
    *_calendar("RIVN", _LAST_FOUR),
    *_calendar("ROOT", _FALLBACK),  # 2026Q3 not returned
    *_calendar("META", _LAST_FOUR),
    # Fiscal year ends early May. Alpha Vantage uses fiscal labels (checked
    # 5 Nov: each label returned the call for that fiscal quarter).
    # 2027Q1 (Q1 FY27) not returned, so Q1 FY26 is used instead.
    Call("CRDO", "2026Q1", 1, 2026),
    Call("CRDO", "2026Q2", 2, 2026),
    Call("CRDO", "2026Q3", 3, 2026),
    Call("CRDO", "2026Q4", 4, 2026),
    *_calendar("NBIS", [(2025, 3), (2025, 4), (2026, 1), (2026, 2)]),
]


# ---------------------------------------------------------------- label check

_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_ORD_RE = "(first|second|third|fourth|1st|2nd|3rd|4th)"
_YEAR_RE = r"(?:fiscal\s+(?:year\s+)?)?(?:20|')?(\d{2})\b"
_PATTERNS = [
    # "third quarter 2026", "first quarter of fiscal 2027", "fourth quarter fiscal year '26"
    (re.compile(rf"\b{_ORD_RE}\s+quarter\s+(?:of\s+)?(?:the\s+)?{_YEAR_RE}", re.I), "oy"),
    # "Q3 2026", "Q1 of fiscal 2027", "Q1 FY27"
    (re.compile(r"\bQ([1-4])\s+(?:of\s+)?(?:fiscal\s+(?:year\s+)?|FY\s?)?(?:20|')?(\d{2})\b", re.I), "qy"),
    # "fiscal 2027 first quarter", "2026 third quarter"
    (re.compile(rf"\b(?:fiscal\s+(?:year\s+)?)?20(\d{{2}})\s+{_ORD_RE}\s+quarter", re.I), "yo"),
]
_OPENING_SEGMENTS = 6


def stated_period(segments: list[dict[str, Any]]) -> tuple[int, int, str] | None:
    """(quarter, year, matched text) the call states about itself, or None."""
    for seg in segments[:_OPENING_SEGMENTS]:
        text = str(seg.get("content", ""))
        found = []
        for pattern, order in _PATTERNS:
            for m in pattern.finditer(text):
                if order == "oy":
                    q, y = _ORD[m.group(1).lower()], m.group(2)
                elif order == "qy":
                    q, y = int(m.group(1)), m.group(2)
                else:
                    y, q = m.group(1), _ORD[m.group(2).lower()]
                found.append((m.start(), q, 2000 + int(y), m.group(0)))
        if found:
            _, q, y, text_match = min(found)  # earliest mention in the segment
            return q, y, text_match
    return None


def label_check(call: Call, segments: list[dict[str, Any]]) -> tuple[str, str | None]:
    stated = stated_period(segments)
    if stated is None:
        return "unknown", None
    q, y, text = stated
    return ("match" if (q, y) == (call.fiscal_quarter, call.fiscal_year) else "mismatch"), text


# ---------------------------------------------------------------- fetching


class KeyPool:
    """API keys in order; a key that hits the daily cap is dropped."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.index = 0
        self.requests = [0] * len(keys)

    @property
    def current(self) -> str | None:
        return self.keys[self.index] if self.index < len(self.keys) else None

    def exhaust(self) -> None:
        self.index += 1

    def summary(self) -> str:
        return ", ".join(f"key {i + 1}: {n}" for i, n in enumerate(self.requests))


def http_get(call: Call, api_key: str) -> tuple[bytes | None, dict[str, Any] | None]:
    """Raw response bytes, or a typed error. Never puts the key in a message."""
    query = urllib.parse.urlencode(
        {"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": call.ticker,
         "quarter": call.label, "apikey": api_key}
    )
    try:
        with urllib.request.urlopen(f"{API_URL}?{query}", timeout=HTTP_TIMEOUT_S) as resp:
            return resp.read(), None
    except urllib.error.HTTPError as exc:
        return None, {"type": "upstream_error", "message": f"HTTP {exc.code}", "retryable": exc.code >= 500}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return None, {"type": "upstream_error", "message": f"network error: {type(reason).__name__}", "retryable": True}


def fetch(call: Call, keys: KeyPool, pause_s: float = PAUSE_S) -> tuple[bytes | None, dict[str, Any] | None]:
    """Fetch one call, rotating keys on the daily cap and backing off on bursts."""
    burst_tries = 0
    while (key := keys.current) is not None:
        time.sleep(pause_s)
        keys.requests[keys.index] += 1
        raw, err = http_get(call, key)
        if err is None:
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return None, {"type": "upstream_error", "message": "upstream returned non-JSON", "retryable": False}
            result = map_upstream(body if isinstance(body, dict) else {}, call.ticker, call.label)
            if result["ok"]:
                return raw, None
            err = result["error"]
        if err["type"] == "rate_limited" and not err["retryable"]:
            print(f"    key {keys.index + 1} hit the daily cap, rotating", file=sys.stderr)
            keys.exhaust()
            burst_tries = 0
            continue
        if err["retryable"] and burst_tries < BURST_RETRIES:
            burst_tries += 1
            time.sleep(pause_s * 4 * burst_tries)
            continue
        return None, err
    return None, {"type": "rate_limited", "message": "all keys hit the daily cap; resume tomorrow", "retryable": False}


# ---------------------------------------------------------------- manifest


def manifest_entry(call: Call, raw: bytes, fetched_at: str) -> dict[str, Any]:
    segments = json.loads(raw).get("transcript") or []
    check, stated = label_check(call, segments)
    return {
        "ticker": call.ticker,
        "quarter": call.label,
        "expected_period": f"Q{call.fiscal_quarter} {call.fiscal_year}",
        "stated_period": stated,
        "label_check": check,
        "source": SOURCE,
        "fetched_at": fetched_at,
        "segments": len(segments),
        "chars": sum(len(str(s.get("content", ""))) for s in segments),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file": str(call.path.relative_to(ROOT)),
    }


def load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST.exists():
        return {}
    return {f"{c['ticker']}_{c['quarter']}": c for c in json.loads(MANIFEST.read_text()).get("calls", [])}


def write_manifest(entries: dict[str, dict[str, Any]], gaps: list[dict[str, Any]]) -> None:
    order = [f"{c.ticker}_{c.label}" for c in CORPUS]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "source": SOURCE,
        "frozen": FROZEN,
        "expected_calls": len(CORPUS),
        "calls": [entries[k] for k in order if k in entries],
        "gaps": gaps,
    }, indent=2) + "\n")


# ---------------------------------------------------------------- main


def load_env_file(path: Path) -> None:
    """Set KEY=value lines from path, without overriding the shell environment.

    Deliberately minimal: blank lines and # comments are skipped, an optional
    `export ` prefix and matching surrounding quotes are stripped. No expansion.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(name, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="refetch calls already on disk")
    parser.add_argument("--dry-run", action="store_true", help="list what would be fetched; no requests")
    parser.add_argument("--only", metavar="TICKER", action="append", help="limit to these tickers")
    args = parser.parse_args(argv)

    calls = [c for c in CORPUS if not args.only or c.ticker in {t.upper() for t in args.only}]
    todo = [c for c in calls if args.force or not c.path.exists()]

    if args.dry_run:
        for c in calls:
            print(f"{'fetch' if c in todo else 'have '}  {c.ticker:5} {c.label}")
        print(f"{len(todo)} request(s) needed (free tier: 25 per key per day)")
        return 0

    load_env_file(ENV_FILE)
    keys = [k.strip() for k in os.environ.get("ALPHAVANTAGE_API_KEYS", "").split(",") if k.strip()]
    if todo and not keys:
        print("ALPHAVANTAGE_API_KEYS is not set (comma-separated keys; shell or .env.local)", file=sys.stderr)
        return 2
    pool = KeyPool(keys)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_manifest()
    gaps: list[dict[str, Any]] = []

    for call in calls:
        key = f"{call.ticker}_{call.label}"
        if call in todo:
            raw, err = fetch(call, pool)
            if err is not None:
                print(f"GAP   {call.ticker:5} {call.label}: {err['type']}: {err['message']}")
                gaps.append({"ticker": call.ticker, "quarter": call.label, **err})
                continue
            call.path.write_bytes(raw)
            entries[key] = manifest_entry(call, raw, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        elif key not in entries:
            # On disk but not in the manifest (e.g. manifest deleted): rebuild from the file.
            mtime = datetime.fromtimestamp(call.path.stat().st_mtime, timezone.utc)
            entries[key] = manifest_entry(call, call.path.read_bytes(), mtime.isoformat(timespec="seconds"))
        e = entries[key]
        print(f"ok    {call.ticker:5} {call.label}: {e['segments']:3} segments, {e['chars']:7,} chars, "
              f"label {e['label_check']}" + (f" ({e['stated_period']!r})" if e["stated_period"] else ""))

    # With --only, the manifest's gap list covers only those tickers until a full run.
    write_manifest(entries, gaps)

    have = sum(1 for c in CORPUS if f"{c.ticker}_{c.label}" in entries)
    print(f"\n{have}/{len(CORPUS)} calls in corpus; requests this run: {pool.summary() or 'none'}")
    unchecked = [e for e in entries.values() if e["label_check"] != "match"]
    for e in unchecked:
        print(f"CHECK {e['ticker']:5} {e['quarter']}: label {e['label_check']}, expected {e['expected_period']}"
              + (f", call says {e['stated_period']!r}" if e["stated_period"] else ""))
    if gaps:
        print(f"{len(gaps)} gap(s): per §8.3, use the next-older quarter for that ticker and update the table")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
