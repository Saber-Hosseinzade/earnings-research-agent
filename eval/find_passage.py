#!/usr/bin/env python3
"""Find the passage for a golden-set row in one call. No model calls.

    python eval/find_passage.py AMD 2026Q2 --list                 # every turn: index, speaker, role, start
    python eval/find_passage.py AMD 2026Q2 "gross margin"         # turns that match, with a passage to paste
    python eval/find_passage.py AMD 2026Q1 --absent "dividend|buyback|repurchas"

PATTERN is a case-insensitive regular expression. For each matching turn this
prints the `passage` object for the golden set, with the matching sentence as
the quote: shorten it to the words that carry the answer, then run
eval/check_golden.py. --absent prints the `absence_check` object for an
unanswerable row and exits 1 if the pattern has any hits (the topic was
discussed, so the question is not unanswerable). See docs/golden_set.md.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_golden import CORPUS_RAW, absence_hits, load_turns, start_of  # noqa: E402

SENTENCE_END = re.compile(r"[.!?]\s+")  # "$11.5 billion" is not a sentence end


def sentence_around(text: str, pos: int) -> str:
    """The sentence of text that contains position pos."""
    ends = [m.end() for m in SENTENCE_END.finditer(text)]
    start = max([e for e in ends if e <= pos], default=0)
    stop = min([e for e in ends if e > pos], default=len(text))
    return text[start:stop].strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ticker")
    ap.add_argument("quarter", help="label as in the file name, e.g. 2026Q2")
    ap.add_argument("pattern", nargs="?", help="case-insensitive regex to look for")
    ap.add_argument("--list", action="store_true", help="list every turn instead of searching")
    ap.add_argument("--absent", metavar="PATTERN", help="count hits for an unanswerable row's topic")
    ap.add_argument("--corpus", type=Path, default=CORPUS_RAW, help="folder of TICKER_QUARTER.json files")
    args = ap.parse_args(argv)

    try:
        turns = load_turns(args.corpus, args.ticker.upper(), args.quarter.upper())
    except FileNotFoundError:
        print(f"no {args.ticker.upper()}_{args.quarter.upper()}.json in {args.corpus}", file=sys.stderr)
        return 2

    if args.list:
        for i, t in enumerate(turns):
            print(f"{i:3}  {start_of(t) if start_of(t) is not None else '-':>5}s  "
                  f"{t.get('speakerName') or '-'} ({t.get('speakerRole') or '-'}), {len(str(t.get('text', '')))} chars")
        return 0

    if args.absent:
        hits = absence_hits(turns, args.absent)
        print(json.dumps({"absence_check": {"pattern": args.absent, "hits": hits}}))
        if hits:
            print(f"{hits} hit(s): the call discusses this, so it is not an unanswerable question", file=sys.stderr)
        return 1 if hits else 0

    if not args.pattern:
        ap.error("give a PATTERN, --list or --absent")
    rx = re.compile(args.pattern, re.I)
    found = 0
    for i, t in enumerate(turns):
        text = str(t.get("text", ""))
        for m in rx.finditer(text):
            found += 1
            passage = {"turn": i, "speaker": t.get("speakerName"), "role": t.get("speakerRole"),
                       "start_seconds": start_of(t), "quote": sentence_around(text, m.start())}
            print(json.dumps({"passage": passage}, ensure_ascii=False))
    print(f"{found} match(es) in {len(turns)} turns", file=sys.stderr)
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
