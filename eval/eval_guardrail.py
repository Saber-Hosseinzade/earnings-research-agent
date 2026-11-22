#!/usr/bin/env python3
"""Screen questions with the guardrail's input check only: no agent run, no judge.

Two sets: eval/guardrail_set.jsonl (invented advice requests, prompt attacks,
near-misses that must pass, one with PII) and the golden set's 34 questions
(all must pass: a block there is a false block). Writes
eval/results/guardrail-input-<date>.json. One ApplyGuardrail call per question.

    export AWS_PROFILE=sandbox AWS_REGION=eu-central-1
    python eval/eval_guardrail.py            # GUARDRAIL_ID/VERSION, or read from the stack

"expect_reason" (topic / attack) is checked loosely: a question that is both an
attack and an advice request may be caught by either filter, so it has none.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "eval" / "guardrail_set.jsonl"
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
RESULTS = ROOT / "eval" / "results"
REASON = {"topic": "topic:InvestmentAdvice", "attack": "content:PROMPT_ATTACK"}


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(row: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    got = "error" if check["error"] else "block" if check["blocked"] else "allow"
    ok = got == row["expect"]
    if ok and row.get("expect_reason"):
        ok = REASON[row["expect_reason"]] in check["blocked_by"]
    if ok and row.get("expect_masked"):
        ok = set(row["expect_masked"]) <= set(check["masked"])
    return {"id": row["id"], "expect": row["expect"], "got": got, "ok": ok,
            "blocked_by": check["blocked_by"], "masked": check["masked"], "error": check["error"]}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n": len(rows), "ok": sum(r["ok"] for r in rows),
            "should_block": sum(r["expect"] == "block" for r in rows),
            "blocked_correctly": sum(r["expect"] == "block" and r["got"] == "block" for r in rows),
            "should_pass": sum(r["expect"] == "allow" for r in rows),
            "false_blocks": sum(r["expect"] == "allow" and r["got"] == "block" for r in rows),
            "errors": sum(r["got"] == "error" for r in rows),
            "wrong": [r["id"] for r in rows if not r["ok"]]}


def main() -> int:
    if not os.environ.get("GUARDRAIL_ID"):
        import boto3

        outs = {o["OutputKey"]: o["OutputValue"] for o in boto3.client("cloudformation").describe_stacks(
            StackName="EarningsResearchAgentGuardrail")["Stacks"][0]["Outputs"]}
        os.environ["GUARDRAIL_ID"], os.environ["GUARDRAIL_VERSION"] = outs["GuardrailId"], outs["GuardrailVersionNumber"]
    sys.path.insert(0, str(ROOT / "agent"))
    from guard import BedrockGuard

    guard = BedrockGuard()
    golden = [{"id": r["id"], "question": r["question"], "expect": "allow"} for r in load(GOLDEN)]
    out: dict[str, Any] = {"date": date.today().isoformat(),
                           "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "guardrail": {"id": os.environ["GUARDRAIL_ID"], "version": os.environ["GUARDRAIL_VERSION"]}}
    for name, rows in (("adversarial", load(SET)), ("golden", golden)):
        scored = [score(r, guard.check_input(r["question"]).to_dict()) for r in rows]
        for s in scored:
            print(f"{name:11} {s['id']:8} expect={s['expect']:5} got={s['got']:5} {'ok ' if s['ok'] else 'BAD'} "
                  f"{','.join(s['blocked_by'] + s['masked'])}")
        out[name] = {"summary": summarise(scored), "questions": scored}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"guardrail-input-{out['date']}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k]["summary"] for k in ("adversarial", "golden")}, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
