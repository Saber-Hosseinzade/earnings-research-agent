#!/usr/bin/env python3
"""Turn the transcript corpus into Knowledge Base documents: one passage per speaker turn.

    python scripts/build_kb_docs.py                      # corpus/v2/raw -> build/kb_docs
    python scripts/build_kb_docs.py --corpus DIR --out DIR

The Knowledge Base is set to no chunking, so each file written here is exactly
one retrievable passage, and its ID is known before ingestion. A passage keeps
the call's turn index, the same index the golden set uses, so retrieval can be
scored as "the expected turn came back" and a cited passage ID can be checked
against what the retriever returned.

Turns longer than MAX_PASSAGE_CHARS are split at sentence ends into parts
(``_p1``, ``_p2`` ...) that share the turn index: one embedding for a 17,000-
character prepared statement would blur every topic in it.

Each passage file starts with a one-line header (company, quarter, speaker) so
the embedding carries which call it is from. Next to it, ``<file>.metadata.json``
holds the attributes Bedrock returns with each result and can filter on.

The output holds transcript text: it goes to the Knowledge Base bucket, never
to the repo (``build/`` is git-ignored). Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPUS_RAW = ROOT / "corpus" / "v2" / "raw"
OUT = ROOT / "build" / "kb_docs"
MAX_PASSAGE_CHARS = 2000
SENTENCE_END = re.compile(r"[.!?]\s+")  # "$11.5 billion" is not a sentence end
FILE_NAME = re.compile(r"^([A-Z]+)_(\d{4}Q[1-4])\.json$")


def split_text(text: str, limit: int = MAX_PASSAGE_CHARS) -> list[str]:
    """Split text at sentence ends into parts of at most `limit` chars.

    A single sentence longer than the limit is kept whole rather than cut mid-word.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    sentences, start = [], 0
    for m in SENTENCE_END.finditer(text):
        sentences.append(text[start : m.end()].strip())
        start = m.end()
    if start < len(text):
        sentences.append(text[start:].strip())

    parts, current = [], ""
    for s in sentences:
        if current and len(current) + 1 + len(s) > limit:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}" if current else s
    if current:
        parts.append(current)
    return parts


def passage_id(ticker: str, quarter: str, turn: int, part: int | None) -> str:
    base = f"{ticker}_{quarter}_t{turn:03d}"
    return base if part is None else f"{base}_p{part}"


def passages_for_call(call: dict[str, Any], ticker: str, quarter: str) -> list[dict[str, Any]]:
    """Passages for one call: text plus the metadata Bedrock stores with it."""
    company = call.get("eventTitle") or ticker
    out = []
    for turn, t in enumerate(call["data"]):
        parts = split_text(str(t.get("text") or ""))
        speaker = t.get("speakerName") or "Unknown"
        role = t.get("speakerRole") or "Unknown"
        start = t.get("startSeconds")
        for i, body in enumerate(parts, start=1):
            pid = passage_id(ticker, quarter, turn, i if len(parts) > 1 else None)
            meta = {
                "passage_id": pid,
                "ticker": ticker,
                "quarter": quarter,
                "fiscal_year": call.get("fiscalYear"),
                "fiscal_quarter": call.get("fiscalQuarter"),
                "turn": turn,
                "part": i,
                "parts": len(parts),
                "speaker": speaker,
                "role": role,
                "start_seconds": None if start is None else int(start + 0.5),
            }
            header = f"{company} | {ticker} {quarter} | {speaker} ({role})"
            out.append({"id": pid, "text": f"{header}\n\n{body}\n", "metadata": meta})
    return out


def write_passage(out_dir: Path, p: dict[str, Any]) -> None:
    path = out_dir / f"{p['id']}.txt"
    path.write_text(p["text"])
    # Bedrock reads <file>.metadata.json next to each source file; null values are dropped.
    attrs = {k: v for k, v in p["metadata"].items() if v is not None}
    (out_dir / f"{path.name}.metadata.json").write_text(json.dumps({"metadataAttributes": attrs}, indent=1))


def build(corpus: Path, out_dir: Path) -> dict[str, int]:
    """Write every call's passages to out_dir (emptied first). Returns passage counts per call."""
    files = sorted(f for f in corpus.glob("*.json") if FILE_NAME.match(f.name))
    if not files:
        raise FileNotFoundError(f"no TICKER_QUARTER.json files in {corpus}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    counts = {}
    for f in files:
        ticker, quarter = FILE_NAME.match(f.name).groups()
        passages = passages_for_call(json.loads(f.read_text()), ticker, quarter)
        for p in passages:
            write_passage(out_dir, p)
        counts[f"{ticker}_{quarter}"] = len(passages)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS_RAW, help="folder of TICKER_QUARTER.json files")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (emptied first)")
    args = ap.parse_args(argv)
    try:
        counts = build(args.corpus, args.out)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    for call, n in counts.items():
        print(f"{call:14} {n:4} passages")
    print(f"{len(counts)} calls, {sum(counts.values())} passages -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
