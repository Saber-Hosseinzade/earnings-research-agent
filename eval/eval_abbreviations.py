#!/usr/bin/env python3
"""Abbreviations in search queries: does spelling them out (agent/retrieval.py) find the passages?

    export AWS_PROFILE=sandbox AWS_REGION=eu-central-1 MODEL_ID=eu.amazon.nova-2-lite-v1:0
    python eval/eval_abbreviations.py retrieval     # Retrieve only, no model: ~240 calls, fractions of a cent
    python eval/eval_abbreviations.py agent [--runs 2]   # answer.run on eval/abbreviation_set.jsonl, off vs on

retrieval  For each term in retrieval.ABBREVIATIONS and each call with a passage that mentions it
           (the abbreviation or the full phrase, or a variant listed in ALSO), the bare abbreviation
           and the expanded query are sent with the call's ticker/quarter filter. Hit = such a passage
           in the top 5. Also reports the full phrase alone, as a ceiling.
agent      Each question uses an abbreviation, on a call that discusses the topic. Scored in code, no
           judge: "found" = covered=true and at least one cited passage mentions the topic. This says the
           agent found the topic, not that the answer is right. Off = expansion disabled for the run.

Writes eval/results/abbreviations-<mode>-<date>.json: IDs, ranks and verdicts, no passage text.
Needs build/kb_docs (make kb-docs) for the passage texts; the corpus is the one the KB was synced from.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

import retrieval  # noqa: E402

KB_DOCS = ROOT / "build" / "kb_docs"
QUESTIONS = ROOT / "eval" / "abbreviation_set.jsonl"
RESULTS = ROOT / "eval" / "results"
TOP_K = 5
STACK = "EarningsResearchAgentKB"

# Other ways management says the same thing, so a passage that uses them counts as on topic.
ALSO = {
    "capex": [r"capital spending"],
    "ARR": [r"annuali[sz]ed recurring revenue"],
    "OCF": [r"cash flow from operations"],
    "TAM": [r"addressable market"],
    "CAC": [r"acquisition cost"],
    "DAP": [r"daily active", r"\bDAU\b"],
}


def topic_pattern(term: str) -> re.Pattern[str]:
    full = retrieval.ABBREVIATIONS[term]
    parts = [r"(?<![A-Za-z&])" + re.escape(term) + r"s?(?![A-Za-z&])", re.escape(full.rstrip("s"))] + ALSO.get(term, [])
    return re.compile("|".join(parts), re.IGNORECASE)


def load_docs(folder: Path = KB_DOCS) -> dict[str, dict[str, str]]:
    """call ("AMD_2026Q2") -> passage_id -> text."""
    docs: dict[str, dict[str, str]] = defaultdict(dict)
    for f in sorted(folder.glob("*.txt")):
        docs["_".join(f.stem.split("_")[:2])][f.stem] = f.read_text()
    return docs


def on_topic(term: str, texts: dict[str, str]) -> set[str]:
    pat = topic_pattern(term)
    return {pid for pid, t in texts.items() if pat.search(t)}


def first_rank(ids: list[str], relevant: set[str]) -> int | None:
    return next((i + 1 for i, pid in enumerate(ids) if pid in relevant), None)


def run_retrieval(retrieve, docs) -> dict[str, Any]:
    rows = []
    for term, full in retrieval.ABBREVIATIONS.items():
        for call, texts in sorted(docs.items()):
            relevant = on_topic(term, texts)
            if not relevant:
                continue
            ticker, quarter = call.split("_")
            ranks = {name: first_rank([p["passage_id"] for p in retrieve(q, ticker, quarter, TOP_K)], relevant)
                     for name, q in (("bare", term), ("expanded", retrieval.expand_abbreviations(term)),
                                     ("phrase", full))}
            rows.append({"term": term, "call": call, "on_topic": len(relevant), **ranks})
    summary = {name: {"hit@1": sum(1 for r in rows if r[name] == 1),
                      f"hit@{TOP_K}": sum(1 for r in rows if r[name] is not None)}
               for name in ("bare", "expanded", "phrase")}
    return {"pairs": len(rows), "summary": summary, "rows": rows}


def score_answer(row: dict[str, Any], out: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    ans = out.get("answer") or {}
    cited = ans.get("citations") or []
    relevant = on_topic(row["term"], texts)
    return {"id": row["id"], "status": out.get("status"), "covered": ans.get("covered"),
            "cited": cited, "cited_on_topic": [c for c in cited if c in relevant],
            "found": bool(ans.get("covered")) and any(c in relevant for c in cited),
            "queries": [q.get("sent", q["query"]) for q in (out.get("retrieval") or {}).get("queries", [])],
            "tokens": (out.get("run") or {}).get("tokens_used")}


def run_agent(questions, docs, runs: int) -> dict[str, Any]:
    import answer

    original = retrieval.expand_abbreviations
    results: dict[str, list] = {}
    for setting in ("off", "on"):
        retrieval.expand_abbreviations = (lambda q: q) if setting == "off" else original
        for n in range(1, runs + 1):
            scored = []
            for row in questions:
                out = answer.run({"question": row["question"], "ticker": row["ticker"], "quarter": row["quarter"]})
                scored.append(score_answer(row, out, docs[f"{row['ticker']}_{row['quarter']}"]))
                print(setting, n, row["id"], scored[-1]["found"], scored[-1]["queries"], flush=True)
            results[f"{setting}-run{n}"] = scored
    retrieval.expand_abbreviations = original
    summary = {k: {"found": sum(s["found"] for s in v), "of": len(v),
                   "tokens": sum(s["tokens"] or 0 for s in v)} for k, v in results.items()}
    return {"questions": len(questions), "summary": summary, "runs": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["retrieval", "agent"])
    ap.add_argument("--runs", type=int, default=1, help="agent mode: runs per setting")
    args = ap.parse_args(argv)

    region = os.environ.get("AWS_REGION", "eu-central-1")
    if not os.environ.get("KNOWLEDGE_BASE_ID"):
        import boto3
        outs = boto3.client("cloudformation", region_name=region).describe_stacks(StackName=STACK)["Stacks"][0]["Outputs"]
        os.environ["KNOWLEDGE_BASE_ID"] = next(o["OutputValue"] for o in outs if "KnowledgeBaseId" in o["OutputKey"])
    docs = load_docs()
    if not docs:
        sys.exit("build/kb_docs is empty: run make kb-docs")

    if args.mode == "retrieval":
        report = run_retrieval(retrieval.bedrock_retriever(region=region), docs)
    else:
        questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
        report = run_agent(questions, docs, args.runs)
        report["model_id"] = os.environ.get("MODEL_ID")
    print(json.dumps(report["summary"], indent=1))
    path = RESULTS / f"abbreviations-{args.mode}-{date.today().isoformat()}.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
