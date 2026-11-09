#!/usr/bin/env python3
"""Score Knowledge Base retrieval on the golden set. No agent, no judge.

    export AWS_PROFILE=sandbox AWS_REGION=eu-central-1
    python eval/eval_retrieval.py --dry-run
    python eval/eval_retrieval.py                  # writes eval/results/retrieval-<store>-<date>.json

For each answerable golden question, the question text is sent to the
Knowledge Base's Retrieve API twice:

  unfiltered  the whole corpus, as a user who names no call would get it
  filtered    restricted to the question's ticker and quarter by metadata,
              as the agent's tool will call it once the call is known

A hit is a returned passage from the golden turn of the right call (any part
of a split turn). Reported per mode: recall@k for k in 1, 3, 5, 10 (share of
questions with a hit in the top k) and MRR (mean of 1/rank of the first hit,
0 when none). Unanswerable questions have no golden turn and are skipped.

The results file holds passage IDs, ranks and scores, never passage text, so
it is committed whole. `kb_docs_sha256` ties the numbers to the passages that
were ingested (build/kb_docs at run time; run `make kb-sync` after a rebuild).

Each Retrieve call embeds the question once with the KB's embedding model:
60 calls per run, fractions of a cent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
RESULTS = ROOT / "eval" / "results"
KB_DOCS = ROOT / "build" / "kb_docs"
STACK = "EarningsResearchAgentKB"
REGION = "eu-central-1"
TOP_K = 10
KS = (1, 3, 5, 10)
MODES = ("unfiltered", "filtered")

# retrieve(question, ticker_or_None, quarter_or_None, k) -> [{"passage_id", "ticker", "quarter", "turn", "score"}]
Retriever = Callable[[str, str | None, str | None, int], list[dict[str, Any]]]


def load_golden(path: Path = GOLDEN) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_hit_rank(row: dict[str, Any], results: list[dict[str, Any]]) -> int | None:
    """1-based rank of the first passage from the golden turn of the golden call, or None."""
    want = (row["ticker"], row["quarter"], row["passage"]["turn"])
    for rank, r in enumerate(results, start=1):
        if (r["ticker"], r["quarter"], r["turn"]) == want:
            return rank
    return None


def summarise(ranks: list[int | None]) -> dict[str, Any]:
    n = len(ranks)
    out: dict[str, Any] = {"questions": n}
    for k in KS:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        out[f"recall@{k}"] = {"hits": hits, "rate": round(hits / n, 3) if n else None}
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3) if n else None
    return out


def score(golden: list[dict[str, Any]], retrieve: Retriever, k: int = TOP_K) -> tuple[list[dict], dict]:
    rows = [r for r in golden if r["answerable"]]
    per_question, ranks = [], {m: [] for m in MODES}
    for row in rows:
        entry: dict[str, Any] = {"id": row["id"], "ticker": row["ticker"], "quarter": row["quarter"],
                                 "kind": row["kind"], "gold_turn": row["passage"]["turn"]}
        for mode in MODES:
            t, q = (row["ticker"], row["quarter"]) if mode == "filtered" else (None, None)
            results = retrieve(row["question"], t, q, k)
            rank = first_hit_rank(row, results)
            ranks[mode].append(rank)
            entry[mode] = {
                "rank": rank,
                "top1_right_call": bool(results) and (results[0]["ticker"], results[0]["quarter"])
                == (row["ticker"], row["quarter"]),
                "returned": [[r["passage_id"], round(r["score"], 4)] for r in results],
            }
        per_question.append(entry)
        print(f"{row['id']:8} unfiltered rank={entry['unfiltered']['rank']}  filtered rank={entry['filtered']['rank']}")
    summary = {m: summarise(ranks[m]) for m in MODES}
    summary["unfiltered"]["top1_right_call"] = sum(e["unfiltered"]["top1_right_call"] for e in per_question)
    summary["skipped_unanswerable"] = len(golden) - len(rows)
    return per_question, summary


def kb_docs_sha256(folder: Path = KB_DOCS) -> str | None:
    """One hash over every passage file's name and bytes, in name order."""
    if not folder.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(folder.iterdir()):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def bedrock_retriever(kb_id: str, region: str) -> Retriever:
    import boto3

    client = boto3.client("bedrock-agent-runtime", region_name=region)

    def retrieve(question: str, ticker: str | None, quarter: str | None, k: int) -> list[dict[str, Any]]:
        search: dict[str, Any] = {"numberOfResults": k}
        if ticker:
            search["filter"] = {"andAll": [{"equals": {"key": "ticker", "value": ticker}},
                                           {"equals": {"key": "quarter", "value": quarter}}]}
        resp = client.retrieve(knowledgeBaseId=kb_id, retrievalQuery={"text": question},
                               retrievalConfiguration={"vectorSearchConfiguration": search})
        out = []
        for r in resp["retrievalResults"]:
            m = r.get("metadata", {})
            out.append({"passage_id": m.get("passage_id"), "ticker": m.get("ticker"), "quarter": m.get("quarter"),
                        "turn": None if m.get("turn") is None else int(m["turn"]), "score": r.get("score", 0.0)})
        return out

    return retrieve


def stack_outputs(stack: str, region: str) -> dict[str, str]:
    import boto3

    outs = boto3.client("cloudformation", region_name=region).describe_stacks(StackName=stack)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in outs.get("Outputs", [])}


def embedding_model_of(kb_id: str, region: str) -> str:
    """The embedding model the deployed KB actually uses (ARN, account-free for foundation models)."""
    import boto3

    kb = boto3.client("bedrock-agent", region_name=region).get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]
    return kb["knowledgeBaseConfiguration"]["vectorKnowledgeBaseConfiguration"]["embeddingModelArn"]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="show the plan; no AWS calls")
    ap.add_argument("--label", default="retrieval", help="result file prefix")
    ap.add_argument("--stack", default=STACK)
    ap.add_argument("--region", default=REGION)
    args = ap.parse_args(argv)

    golden = load_golden()
    answerable = sum(r["answerable"] for r in golden)
    if args.dry_run:
        print(f"{answerable} answerable questions x {len(MODES)} modes = {answerable * len(MODES)} Retrieve calls, top {TOP_K}")
        return 0

    out = stack_outputs(args.stack, args.region)
    store = out.get("VectorStore", "unknown")
    per_question, summary = score(golden, bedrock_retriever(out["KnowledgeBaseId"], args.region))

    embedding_model = embedding_model_of(out["KnowledgeBaseId"], args.region)

    result = {
        "label": args.label,
        "date": date.today().isoformat(),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vector_store": store,
        "embedding_model_arn": embedding_model,
        "top_k": TOP_K,
        "git_head": git_head(),
        "kb_docs_sha256": kb_docs_sha256(),
        "corpus_manifest": "corpus/manifest_v2.json",
        "hit_rule": "a returned passage from the golden turn of the golden call (any part of a split turn)",
        "summary": summary,
        "questions": per_question,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.label}-{store}-{result['date']}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
