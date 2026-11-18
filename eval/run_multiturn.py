#!/usr/bin/env python3
"""Run the multi-turn set (eval/multiturn_set.json) and score each turn in code. No judge model.

    export AWS_PROFILE=sandbox AWS_REGION=eu-central-1 MODEL_ID=eu.amazon.nova-2-lite-v1:0
    python eval/run_multiturn.py --dry-run
    python eval/run_multiturn.py --label multiturn-baseline     # writes eval/results/<label>-<date>.json

Each conversation runs through session.run_turn, the same path as the deployed runtime (serve.py),
with a fresh in-memory session, the Knowledge Base and, unless --no-guardrail, the Bedrock Guardrail.

Scores (docs/conversation_design.md, Evaluation):
  turn          passes when every expectation listed for it holds (keys in multiturn_set.json)
  goal success  a conversation passes when all its turns pass
  refusal       turns expecting "blocked" that were blocked, and turns expecting no refusal that were not
  by flow       conversations passed per flow
The scores file holds verdicts, IDs and counts; answer text goes to <label>-<date>.outputs.json (git-ignored).
Expectation checks are literal (a regex for a topic, call prefixes for citations): they show behaviour on a
small set, not rates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

SET = ROOT / "eval" / "multiturn_set.json"
RESULTS = ROOT / "eval" / "results"
CONTEXT_KEYS = ("how", "ticker", "quarter")


def load(path: Path = SET) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["conversations"]


def check(expect: dict[str, Any], out: dict[str, Any]) -> list[str]:
    """Names of the expectations that failed for one turn (empty = pass)."""
    failed: list[str] = []
    answer = out.get("answer") or {}
    text = answer.get("answer") or ""
    cited = answer.get("citations") or []
    context = out.get("context") or {}
    searched = bool((out.get("retrieval") or {}).get("queries"))

    if "status" in expect and out.get("status") != expect["status"]:
        failed.append("status")
    for key in CONTEXT_KEYS:
        if key in expect and context.get(key) != expect[key]:
            failed.append(key)
    if "covered" in expect and bool(answer.get("covered")) != expect["covered"]:
        failed.append("covered")
    for prefix in expect.get("cites", []):
        if not any(c.startswith(prefix) for c in cited):
            failed.append(f"cites:{prefix}")
    if "cites_only" in expect and any(not c.startswith(tuple(expect["cites_only"])) for c in cited):
        failed.append("cites_only")
    if "mentions" in expect and not re.search(expect["mentions"], text, re.IGNORECASE):
        failed.append("mentions")
    if expect.get("clarifies") and (answer.get("covered") or cited or searched or "?" not in text):
        failed.append("clarifies")
    return failed


def score(conversations: list[dict[str, Any]], runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    turns = passed = 0
    by_flow: dict[str, dict[str, int]] = defaultdict(lambda: {"conversations": 0, "passed": 0})
    refusal = {"expected_blocked": 0, "blocked": 0, "expected_not_blocked": 0, "wrongly_blocked": 0}
    failures: dict[str, int] = defaultdict(int)
    goal = 0
    for conv in conversations:
        verdicts = runs[conv["id"]]
        ok = all(not v["failed"] for v in verdicts)
        goal += ok
        flow = by_flow[str(conv["flow"])]
        flow["conversations"] += 1
        flow["passed"] += ok
        for turn, v in zip(conv["turns"], verdicts):
            turns += 1
            passed += not v["failed"]
            for name in v["failed"]:
                failures[name.split(":")[0]] += 1
            want = turn["expect"].get("status")
            if want == "blocked":
                refusal["expected_blocked"] += 1
                refusal["blocked"] += v["status"] == "blocked"
            else:
                refusal["expected_not_blocked"] += 1
                refusal["wrongly_blocked"] += v["status"] == "blocked"
    return {"conversations": len(conversations), "goal_success": goal, "turns": turns, "turns_passed": passed,
            "by_flow": dict(sorted(by_flow.items())), "refusal": refusal,
            "failed_checks": dict(sorted(failures.items()))}


def run_conversation(conv: dict[str, Any], corpus: Any, guard: Any) -> tuple[list[dict], list[dict]]:
    import session

    store, sid = session.InMemoryStore(), f"eval-{conv['id']}"
    verdicts, outputs = [], []
    for turn in conv["turns"]:
        out = session.run_turn({"question": turn["user"]}, sid, store, corpus, guard=guard)
        answer = out.get("answer") or {}
        verdicts.append({
            "user": turn["user"], "failed": check(turn["expect"], out), "status": out.get("status"),
            "context": {k: (out.get("context") or {}).get(k) for k in CONTEXT_KEYS},
            "covered": answer.get("covered"), "cited": answer.get("citations") or [],
            "queries": [q.get("sent", q["query"]) for q in (out.get("retrieval") or {}).get("queries", [])],
            "tokens": (out.get("run") or {}).get("tokens_used"),
        })
        outputs.append({"user": turn["user"], "answer": answer.get("answer"), "status": out.get("status")})
        print(conv["id"], f"{turn['user'][:50]:50}", out.get("status"), verdicts[-1]["failed"] or "pass", flush=True)
    return verdicts, outputs


def stack_env(region: str, guardrail: bool) -> None:
    import boto3
    cf = boto3.client("cloudformation", region_name=region)

    def outputs(name: str) -> dict[str, str]:
        return {o["OutputKey"]: o["OutputValue"] for o in cf.describe_stacks(StackName=name)["Stacks"][0]["Outputs"]}

    if not os.environ.get("KNOWLEDGE_BASE_ID"):
        kb = outputs("EarningsResearchAgentKB")
        os.environ["KNOWLEDGE_BASE_ID"] = next(v for k, v in kb.items() if "KnowledgeBaseId" in k)
    if guardrail and not os.environ.get("GUARDRAIL_ID"):
        g = outputs("EarningsResearchAgentGuardrail")
        os.environ["GUARDRAIL_ID"], os.environ["GUARDRAIL_VERSION"] = g["GuardrailId"], g["GuardrailVersionNumber"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="list the conversations; no AWS calls")
    ap.add_argument("--label", default="multiturn")
    ap.add_argument("--only", help="comma-separated conversation ids")
    ap.add_argument("--no-guardrail", action="store_true")
    args = ap.parse_args(argv)

    conversations = load()
    if args.only:
        keep = set(args.only.split(","))
        conversations = [c for c in conversations if c["id"] in keep]
    if args.dry_run:
        for c in conversations:
            print(c["id"], f"flow {c['flow']}", " | ".join(t["user"] for t in c["turns"]))
        return 0

    region = os.environ.get("AWS_REGION", "eu-central-1")
    stack_env(region, not args.no_guardrail)
    import prompts
    import session
    from guard import BedrockGuard

    guard = None if args.no_guardrail else BedrockGuard()
    corpus = session.Corpus.load()
    runs, outputs = {}, {}
    for conv in conversations:
        runs[conv["id"]], outputs[conv["id"]] = run_conversation(conv, corpus, guard)

    summary = score(conversations, runs)
    stamp = date.today().isoformat()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {"label": args.label, "date": stamp, "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model_id": os.environ.get("MODEL_ID"), "git_head": head, "guardrail": not args.no_guardrail,
              "prompt": prompts.get("answer_system").info(), "summary": summary, "conversations": runs}
    base = RESULTS / f"{args.label}-{stamp}"
    base.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    Path(f"{base}.outputs.json").write_text(json.dumps({"label": args.label, "date": stamp, "outputs": outputs}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {base.relative_to(ROOT)}.json (scores, committed) and .outputs.json (answers, git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
