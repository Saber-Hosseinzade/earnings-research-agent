#!/usr/bin/env python3
"""Pull the v2 corpus from Equibles: the same 24 (ticker, fiscal quarter) pairs as
scripts/pull_corpus.py, so the golden set's calls are unchanged.

One raw JSON per call goes to corpus/equibles/raw/ (git-ignored: company
copyright, provider terms). corpus/manifest_equibles.json (committed) records,
per call, the turn, speaker and character counts and the sha256 of the raw file.

Usage (from the repo root):

    export EQUIBLES_API_KEY=eq_...           # or put the line in .env.local
    python scripts/pull_corpus_equibles.py             # fetches only missing calls
    python scripts/pull_corpus_equibles.py --dry-run   # list what would be fetched
    python scripts/pull_corpus_equibles.py --force     # refetch everything

Equibles labels calls by the company's own fiscal year and quarter, and the
response states them, so the label check compares the response with the request
instead of reading the opening text. Free tier: 100 requests a day, shared with
MCP. A call is one request per 200 turns.

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_corpus import CORPUS, ROOT, FROZEN, Call, load_env_file  # noqa: E402

ENV_FILE = ROOT / ".env.local"
RAW_DIR = ROOT / "corpus" / "equibles" / "raw"
MANIFEST = ROOT / "corpus" / "manifest_equibles.json"
SOURCE = "Equibles /v1/stocks/{ticker}/earnings-calls/{fy}/{fq}/speakers"
BASE = "https://api.equibles.com"
PAGE = 200
PAUSE_S = 0.5
HTTP_TIMEOUT_S = 30.0


def raw_path(call: Call) -> Path:
    return RAW_DIR / f"{call.ticker}_{call.label}.json"


def get_page(call: Call, offset: int, key: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    url = (f"{BASE}/v1/stocks/{call.ticker}/earnings-calls/{call.fiscal_year}/{call.fiscal_quarter}"
           f"/speakers?limit={PAGE}&offset={offset}")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return json.load(resp), None
    except urllib.error.HTTPError as exc:  # never include the key in a message
        kind = {404: "not_found", 429: "rate_limited", 401: "unauthorized"}.get(exc.code, "upstream_error")
        return None, {"type": kind, "message": f"HTTP {exc.code}"}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, {"type": "upstream_error", "message": type(exc).__name__}


def fetch(call: Call, key: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """All turns of one call (paged), the error if any, and requests used."""
    merged: dict[str, Any] | None = None
    offset, used = 0, 0
    while True:
        time.sleep(PAUSE_S)
        page, err = get_page(call, offset, key)
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


def manifest_entry(call: Call, raw: bytes, fetched_at: str) -> dict[str, Any]:
    body = json.loads(raw)
    turns = body["data"]
    got = (body.get("fiscalYear"), body.get("fiscalQuarter"))
    want = (call.fiscal_year, call.fiscal_quarter)
    return {
        "ticker": call.ticker,
        "quarter": call.label,
        "expected_period": f"Q{call.fiscal_quarter} FY{call.fiscal_year}",
        "returned_period": f"Q{got[1]} FY{got[0]}",
        "label_check": "match" if got == want else "mismatch",
        "call_date": (body.get("callDate") or "")[:10],
        "is_filed_record": body.get("isFiledRecord"),
        "source": SOURCE,
        "fetched_at": fetched_at,
        "turns": len(turns),
        "speakers": len({t.get("speakerName") or t.get("speakerRole") for t in turns}),
        "unnamed_turns": sum(1 for t in turns if not t.get("speakerName")),
        "chars": sum(len(str(t.get("text", ""))) for t in turns),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file": str(raw_path(call).relative_to(ROOT)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", metavar="TICKER", action="append")
    args = ap.parse_args(argv)

    calls = [c for c in CORPUS if not args.only or c.ticker in {t.upper() for t in args.only}]
    todo = [c for c in calls if args.force or not raw_path(c).exists()]
    if args.dry_run:
        for c in calls:
            print(f"{'fetch' if c in todo else 'have '}  {c.ticker:5} {c.label}")
        print(f"{len(todo)} call(s) to fetch, at least {len(todo)} request(s) (free tier: 100 a day)")
        return 0

    load_env_file(ENV_FILE)
    key = os.environ.get("EQUIBLES_API_KEY", "").strip()
    if todo and not key:
        print("EQUIBLES_API_KEY is not set (shell or .env.local)", file=sys.stderr)
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prior = json.loads(MANIFEST.read_text()).get("calls", []) if MANIFEST.exists() else []
    entries = {f"{e['ticker']}_{e['quarter']}": e for e in prior}
    gaps: list[dict[str, Any]] = []
    requests = 0

    for call in calls:
        name = f"{call.ticker}_{call.label}"
        if call in todo:
            body, err, used = fetch(call, key)
            requests += used
            if err is not None:
                print(f"GAP   {call.ticker:5} {call.label}: {err['type']}: {err['message']}")
                gaps.append({"ticker": call.ticker, "quarter": call.label, **err})
                if err["type"] in ("rate_limited", "unauthorized"):
                    break
                continue
            raw = (json.dumps(body, indent=1) + "\n").encode()
            raw_path(call).write_bytes(raw)
            entries[name] = manifest_entry(call, raw, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        elif name not in entries:
            mtime = datetime.fromtimestamp(raw_path(call).stat().st_mtime, timezone.utc)
            entries[name] = manifest_entry(call, raw_path(call).read_bytes(), mtime.isoformat(timespec="seconds"))
        e = entries[name]
        print(f"ok    {call.ticker:5} {call.label}: {e['turns']:3} turns, {e['chars']:7,} chars, "
              f"{e['call_date']}, label {e['label_check']}")

    order = [f"{c.ticker}_{c.label}" for c in CORPUS]
    MANIFEST.write_text(json.dumps({
        "source": SOURCE, "frozen": FROZEN, "expected_calls": len(CORPUS),
        "calls": [entries[k] for k in order if k in entries], "gaps": gaps,
    }, indent=2) + "\n")

    print(f"\n{sum(1 for k in order if k in entries)}/{len(CORPUS)} calls; requests this run: {requests}")
    for e in entries.values():
        if e["label_check"] != "match" or e["is_filed_record"]:
            print(f"CHECK {e['ticker']:5} {e['quarter']}: label {e['label_check']}, filed-record summary: {e['is_filed_record']}")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
