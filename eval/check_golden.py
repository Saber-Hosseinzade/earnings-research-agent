#!/usr/bin/env python3
"""Check a golden set against the corpus it was written from. No model calls.

    python eval/check_golden.py                      # eval/golden_set.jsonl vs corpus/v2/raw/
    python eval/check_golden.py --golden my.jsonl --corpus path/to/raw

Every row: required fields, unique id, known kind, and a transcript file for its
ticker and quarter. Answerable rows: the passage points at a real turn, speaker,
role and start second match that turn, and the quote is in that turn word for
word (ignoring case, spacing and curly quotes) and in no other turn. Unanswerable
rows: no passage, no key facts, and the absence pattern has no hits anywhere in
the call. Exit code 1 if any row fails. The format is described in
docs/golden_set.md.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
CORPUS_RAW = ROOT / "corpus" / "v2" / "raw"

ANSWERABLE_KINDS = {"fact", "reason", "guidance", "assumption"}
UNANSWERABLE_KIND = "not_discussed"
MAX_QUOTE_CHARS = 200  # keep quotes short: the golden set is public, the transcripts are not
ROW_FIELDS = ("id", "ticker", "quarter", "answerable", "kind", "question", "expected_answer", "key_facts", "passage")
PASSAGE_FIELDS = ("turn", "speaker", "role", "start_seconds", "quote")


def norm(text: str) -> str:
    """Same normalisation as the citation check in eval/run_eval.py."""
    return re.sub(r"\s+", " ", text.lower().replace("’", "'").replace("—", "-")).strip(" .\"")


def load_turns(corpus: Path, ticker: str, quarter: str) -> list[dict[str, Any]]:
    return json.loads((corpus / f"{ticker}_{quarter}.json").read_text())["data"]


def start_of(turn: dict[str, Any]) -> int | None:
    """The turn's start second, rounded half up (140.5 -> 141); None if the call gives none."""
    s = turn.get("startSeconds")
    return None if s is None else math.floor(s + 0.5)


def absence_hits(turns: list[dict[str, Any]], pattern: str) -> int:
    rx = re.compile(pattern, re.I)
    return sum(len(rx.findall(str(t.get("text", "")))) for t in turns)


def check_row(row: dict[str, Any], corpus: Path) -> list[str]:
    """Problems with one row; empty if it passes."""
    missing = [f for f in ROW_FIELDS if f not in row]
    if missing:
        return [f"missing fields: {', '.join(missing)}"]
    errs: list[str] = []
    for f in ("question", "expected_answer"):
        if not isinstance(row[f], str) or not row[f].strip():
            errs.append(f"{f} is empty")
    if not isinstance(row["key_facts"], list):
        errs.append("key_facts is not a list")
    try:
        turns = load_turns(corpus, row["ticker"], row["quarter"])
    except FileNotFoundError:
        return errs + [f"no transcript file {row['ticker']}_{row['quarter']}.json in {corpus}"]

    if row["answerable"] is True:
        if row["kind"] not in ANSWERABLE_KINDS:
            errs.append(f"kind {row['kind']!r} is not one of {sorted(ANSWERABLE_KINDS)}")
        if not row["key_facts"]:
            errs.append("answerable row has no key_facts")
        p = row["passage"]
        if not isinstance(p, dict) or any(f not in p for f in PASSAGE_FIELDS):
            return errs + [f"passage needs {', '.join(PASSAGE_FIELDS)}"]
        if not isinstance(p["turn"], int) or not 0 <= p["turn"] < len(turns):
            return errs + [f"passage turn {p['turn']!r} is outside 0..{len(turns) - 1}"]
        t = turns[p["turn"]]
        if p["speaker"] != t.get("speakerName"):
            errs.append(f"speaker {p['speaker']!r} != turn's {t.get('speakerName')!r}")
        if p["role"] != t.get("speakerRole"):
            errs.append(f"role {p['role']!r} != turn's {t.get('speakerRole')!r}")
        if p["start_seconds"] != start_of(t):
            errs.append(f"start_seconds {p['start_seconds']!r} != turn's {start_of(t)!r} (rounded)")
        q = norm(str(p["quote"]))
        if not q:
            errs.append("quote is empty")
        elif len(p["quote"]) > MAX_QUOTE_CHARS:
            errs.append(f"quote is {len(p['quote'])} chars (max {MAX_QUOTE_CHARS})")
        else:
            where = [i for i, x in enumerate(turns) if q in norm(str(x.get("text", "")))]
            if p["turn"] not in where:
                errs.append(f"quote not found in turn {p['turn']}" + (f"; found in turn(s) {where}" if where else ""))
            elif len(where) > 1:
                errs.append(f"quote also in turn(s) {[i for i in where if i != p['turn']]}; make it longer")
    elif row["answerable"] is False:
        if row["kind"] != UNANSWERABLE_KIND:
            errs.append(f"unanswerable row must have kind {UNANSWERABLE_KIND!r}")
        if row["passage"] is not None:
            errs.append("unanswerable row must have passage null")
        if row["key_facts"]:
            errs.append("unanswerable row must have no key_facts")
        a = row.get("absence_check")
        if not isinstance(a, dict) or not a.get("pattern"):
            errs.append("unanswerable row needs absence_check.pattern")
        else:
            try:
                hits = absence_hits(turns, a["pattern"])
            except re.error as exc:
                errs.append(f"absence_check.pattern is not a valid regex: {exc}")
            else:
                if hits:
                    errs.append(f"absence pattern has {hits} hit(s) in the call: the topic was discussed")
                if a.get("hits") != hits:
                    errs.append(f"absence_check.hits is {a.get('hits')!r}, recomputed {hits}")
    else:
        errs.append("answerable must be true or false")
    return errs


def check(golden: Path, corpus: Path) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    problems: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for n, line in enumerate(golden.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems[f"line {n}"] = [f"not valid JSON: {exc}"]
            continue
        rid = str(row.get("id", f"line {n}"))
        seen[rid] += 1
        rows.append(row)
        errs = check_row(row, corpus)
        if seen[rid] > 1:
            errs.append("duplicate id")
        if errs:
            problems[rid] = problems.get(rid, []) + errs
    return problems, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--corpus", type=Path, default=CORPUS_RAW, help="folder of TICKER_QUARTER.json files")
    args = ap.parse_args(argv)
    if not args.corpus.is_dir():
        print(f"corpus folder not found: {args.corpus} (pull it first, see the README)", file=sys.stderr)
        return 2

    problems, rows = check(args.golden, args.corpus)
    for rid, errs in problems.items():
        for e in errs:
            print(f"FAIL  {rid}: {e}")
    answerable = [r for r in rows if r.get("answerable") is True]
    print(f"\n{len(rows)} rows: {len(answerable)} answerable, {len(rows) - len(answerable)} unanswerable; "
          f"{len(problems)} with problems")
    print("per ticker: " + ", ".join(f"{t} {n}" for t, n in sorted(Counter(r.get('ticker') for r in answerable).items())))
    print("per kind:   " + ", ".join(f"{k} {n}" for k, n in sorted(Counter(r.get('kind') for r in rows).items())))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
