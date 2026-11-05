#!/usr/bin/env python3
"""Score the v1 agent on the golden set. Retrieval, grounding, task and abstention
are scored separately (PROJECT_CONTEXT.md 8.7).

v1 takes {ticker, quarter} and returns an EarningsBrief, not an answer to a
question. So v1.0 is scored as shipped: it runs once per call, and every golden
question for that call is scored against that one brief. Low scores on specific
questions are the honest baseline, not a bug in the harness.

Transcripts come from corpus/equibles/raw/ through a local tool that replaces
the Lambda tool (same "Speaker (Role): text" format, same 100,000-character cap),
so a run uses no Alpha Vantage or Equibles requests. agent/ is not modified.

    export MODEL_ID=eu.amazon.nova-2-lite-v1:0     # agent model
    export AWS_PROFILE=sandbox AWS_REGION=eu-central-1
    python eval/run_eval.py --dry-run
    python eval/run_eval.py --limit 3              # smoke test: first 3 questions
    python eval/run_eval.py                        # full run, writes eval/results/

Scores by dimension:
  retrieval   the right call was fetched (v1 definition; the real retrieval
              metric starts with the Knowledge Base)
  task        judge: does the brief contain the expected answer
              (correct / partial / missing)
  grounding   judge: are the brief's claims about the question supported by the
              gold transcript turn (supported / unsupported / no_claim)
  abstention  unanswerable questions only: did the brief avoid answering
  false abstentions  answerable questions whose brief has no relevant claim

The judge is a different model from the agent (JUDGE_MODEL_ID). An LLM judge can
be wrong; the per-question verdicts are saved so any score can be checked by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
CORPUS_RAW = ROOT / "corpus" / "equibles" / "raw"
RESULTS = ROOT / "eval" / "results"
MAX_TRANSCRIPT_CHARS = 100_000  # same as the Lambda tool's default
DEFAULT_JUDGE = "eu.amazon.nova-pro-v1:0"
MAX_JUDGE_CALLS = 40  # cost cap: 34 questions + a few retries


# ---------------------------------------------------------------- data


def load_golden() -> list[dict[str, Any]]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def load_call(ticker: str, quarter: str) -> list[dict[str, Any]]:
    return json.loads((CORPUS_RAW / f"{ticker}_{quarter}.json").read_text())["data"]


def render_transcript(turns: list[dict[str, Any]]) -> tuple[str, bool]:
    """Same text shape as tool_lambda/handler.py: 'Speaker (Role): text' lines, capped."""
    lines, used, truncated = [], 0, False
    for t in turns:
        line = f"{t.get('speakerName') or t.get('speakerRole') or '?'} ({t.get('speakerRole') or ''}): {t['text']}"
        if used + len(line) > MAX_TRANSCRIPT_CHARS:
            truncated = True
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines), truncated


# ---------------------------------------------------------------- agent side


def run_agent(target: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run an unmodified agent module (v1.0 main, v1.1 ask) with a local transcript tool in place of the Lambda."""
    sys.path.insert(0, str(ROOT / "agent"))
    import importlib  # noqa: PLC0415

    v1 = importlib.import_module({"v1.0": "main", "v1.1": "ask"}[target])  # needs MODEL_ID etc. at call time
    ticker, quarter = payload["ticker"], payload["quarter"]
    from strands import tool  # noqa: PLC0415

    fetched: list[tuple[str, str]] = []

    @tool
    def get_earnings_call_transcript(ticker: str, quarter: str) -> str:
        """Fetch an excerpt of a company's earnings-call transcript.

        Returns JSON. On success: ok=true, the transcript text as "Speaker (Title): content"
        lines, and truncated=true if it was cut to fit a length budget. On failure: ok=false
        with error.type and error.retryable. Do not retry an error that is not retryable.

        Args:
            ticker: Stock ticker symbol, e.g. "IBM".
            quarter: Fiscal quarter in the form YYYYQn, e.g. "2024Q4".
        """
        fetched.append((ticker.upper(), quarter.upper()))
        try:
            turns = load_call(ticker.upper(), quarter.upper())
        except FileNotFoundError:
            return json.dumps({"ok": False, "error": {"type": "not_found",
                               "message": f"no transcript for {ticker} {quarter}", "retryable": False}})
        text, truncated = render_transcript(turns)
        return json.dumps({"ok": True, "ticker": ticker.upper(), "quarter": quarter.upper(),
                           "segments_returned": len(turns), "segments_total": len(turns),
                           "truncated": truncated, "transcript": text})

    v1.get_earnings_call_transcript = get_earnings_call_transcript  # build_agent reads this name
    result = v1.run(payload)
    result["fetched"] = fetched
    result["output"] = result.get("brief", result.get("answer"))
    return result


# ---------------------------------------------------------------- scoring


def score_retrieval(row: dict[str, Any], fetched: list[tuple[str, str]]) -> bool:
    return (row["ticker"], row["quarter"]) in fetched


JUDGE_PROMPT = """You are grading a research assistant's output. Be strict and literal.

QUESTION: {question}
EXPECTED ANSWER: {expected}
KEY FACTS THAT SHOULD APPEAR: {key_facts}
GOLD TRANSCRIPT EVIDENCE (the turn that answers the question; may be empty if the call never covers it):
{evidence}

ASSISTANT OUTPUT (a brief of the whole call, or a direct answer to the question):
{brief}

Return ONLY a JSON object with these keys:
"has_claim": true if the output states anything that directly answers the question, else false.
"task": "correct" if the output contains the expected answer and its key facts, "partial" if some but not all, "missing" if none.
"grounding": "supported" if every claim in the output that addresses the question is backed by the gold evidence, "unsupported" if any such claim is not backed or is contradicted, "no_claim" if there is no such claim.
"note": one short sentence on why."""


def parse_judge(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("judge returned no JSON")
    v = json.loads(m.group(0))
    if v.get("task") not in {"correct", "partial", "missing"} or v.get("grounding") not in {
        "supported", "unsupported", "no_claim"} or not isinstance(v.get("has_claim"), bool):
        raise ValueError(f"judge verdict malformed: {v}")
    return v


def evidence_text(row: dict[str, Any]) -> str:
    p = row.get("passage")
    if not p:
        return ""
    turns = load_call(row["ticker"], row["quarter"])
    return str(turns[p["turn"]]["text"])


def judge_row(row: dict[str, Any], brief: dict[str, Any] | None, ask: Callable[[str], str]) -> dict[str, Any]:
    if brief is None:  # the agent produced nothing
        return {"has_claim": False, "task": "missing", "grounding": "no_claim", "note": "no brief produced"}
    prompt = JUDGE_PROMPT.format(
        question=row["question"], expected=row["expected_answer"],
        key_facts="; ".join(row["key_facts"]) or "(none: the call does not cover this)",
        evidence=evidence_text(row) or "(none)", brief=json.dumps(brief, ensure_ascii=False, indent=1))
    last: Exception | None = None
    for _ in range(2):
        try:
            return parse_judge(ask(prompt))
        except (ValueError, json.JSONDecodeError) as exc:
            last = exc
    return {"has_claim": False, "task": "missing", "grounding": "no_claim", "note": f"judge failed: {last}", "judge_error": True}


def quote_in_call(quote: str, row: dict[str, Any]) -> bool:
    """Citation check in code: is the quoted sentence really in the call (ignoring case, spacing, dashes)?"""
    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", t.lower().replace("\u2019", "'").replace("\u2014", "-")).strip(" .\"")

    text = norm(" ".join(t["text"] for t in load_call(row["ticker"], row["quarter"])))
    return norm(quote) in text


def declined(r: dict[str, Any]) -> bool:
    """Did the system decline to answer? v1.1 says so itself (covered_in_call=false). A v1 brief has no way to
    say it, so there it means the brief makes no claim about the question. The judge's has_claim is not used
    for v1.1: it counts 'the call did not discuss this' as a claim."""
    return r["abstained"] if "abstained" in r else not r["verdict"]["has_claim"]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    n = len(ans)
    task = Counter(r["verdict"]["task"] for r in ans)
    ground = Counter(r["verdict"]["grounding"] for r in ans)
    return {
        "answerable": n,
        "unanswerable": len(una),
        "retrieval_right_call": sum(r["retrieval"] for r in ans),
        "task": {k: task.get(k, 0) for k in ("correct", "partial", "missing")},
        "grounding": {k: ground.get(k, 0) for k in ("supported", "unsupported", "no_claim")},
        "false_abstentions": sum(declined(r) for r in ans),
        "abstention_correct": sum(declined(r) for r in una),
        "answered_anyway": sum(not declined(r) for r in una),
        "unanswerable_answers_unsupported": sum(r["verdict"]["grounding"] == "unsupported" for r in una),
        "judge_errors": sum(bool(r["verdict"].get("judge_error")) for r in rows),
        **_v11_extras(rows),
    }


def _v11_extras(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quoted = [r for r in rows if r.get("quote_in_call") is not None]
    if not quoted and not any("abstained" in r for r in rows):
        return {}
    return {
        "quotes_given": len(quoted),
        "quotes_found_in_call": sum(r["quote_in_call"] for r in quoted),
    }


# ---------------------------------------------------------------- judge client


def bedrock_judge(model_id: str) -> Callable[[str], str]:
    import boto3  # noqa: PLC0415

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "eu-central-1"))

    def ask(prompt: str) -> str:
        resp = client.converse(modelId=model_id, messages=[{"role": "user", "content": [{"text": prompt}]}],
                               inferenceConfig={"temperature": 0.0, "maxTokens": 400})
        return resp["output"]["message"]["content"][0]["text"]

    return ask


# ---------------------------------------------------------------- main


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="show the plan; no model calls")
    ap.add_argument("--limit", type=int, help="score only the first N questions")
    ap.add_argument("--target", choices=["v1.0", "v1.1"], default="v1.0",
                    help="v1.0: brief, one run per call. v1.1: question in, answer out, one run per question")
    ap.add_argument("--label", help="result file prefix (default baseline-<target>)")
    args = ap.parse_args(argv)

    golden = load_golden()[: args.limit]
    args.label = args.label or f"baseline-{args.target}"
    per_question = args.target == "v1.1"
    calls = sorted({(r["ticker"], r["quarter"]) for r in golden})
    judge_id = os.environ.get("JUDGE_MODEL_ID", DEFAULT_JUDGE)
    if args.dry_run:
        print(f"{len(golden)} questions, {len(golden) if per_question else len(calls)} agent runs, {len(golden)} judge calls (cap {MAX_JUDGE_CALLS})")
        print(f"agent MODEL_ID={os.environ.get('MODEL_ID', '(unset)')}  judge={judge_id}")
        return 0
    if "MODEL_ID" not in os.environ:
        print("MODEL_ID is not set", file=sys.stderr)
        return 2
    if len(golden) > MAX_JUDGE_CALLS:
        print(f"{len(golden)} questions exceeds the judge cap of {MAX_JUDGE_CALLS}", file=sys.stderr)
        return 2

    results: dict[str, dict[str, Any]] = {}  # key: call (v1.0) or question id (v1.1)

    def get_result(row: dict[str, Any]) -> dict[str, Any]:
        key = row["id"] if per_question else f"{row['ticker']}_{row['quarter']}"
        if key not in results:
            print(f"run   {key} ...", flush=True)
            payload = {"ticker": row["ticker"], "quarter": row["quarter"]}
            if per_question:
                payload["question"] = row["question"]
            results[key] = run_agent(args.target, payload)
            print(f"      status={results[key]['status']} tokens={results[key]['run'].get('tokens_used', '?')}")
        return results[key]

    ask = bedrock_judge(judge_id)
    scored = []
    for row in golden:
        res = get_result(row)
        verdict = judge_row(row, res["output"], ask)
        entry = {"id": row["id"], "ticker": row["ticker"], "quarter": row["quarter"],
                 "answerable": row["answerable"], "kind": row["kind"], "status": res["status"],
                 "retrieval": score_retrieval(row, res["fetched"]), "verdict": verdict}
        if per_question and res["output"]:
            out_ = res["output"]
            entry["abstained"] = not out_["covered_in_call"]
            entry["quote_in_call"] = quote_in_call(out_["quote"], row) if out_["quote"] else None
        scored.append(entry)
        print(f"judge {row['id']:8} task={verdict['task']:8} grounding={verdict['grounding']:10} claim={verdict['has_claim']}")

    summary = aggregate(scored)
    prompt_src = (ROOT / "agent" / ("ask.py" if per_question else "main.py")).read_text()
    out = {
        "label": args.label,
        "date": date.today().isoformat(),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_model_id": os.environ["MODEL_ID"],
        "judge_model_id": judge_id,
        "git_head": git("rev-parse", "--short", "HEAD"),
        "git_tag_v1.0": git("rev-parse", "--short", "v1.0"),
        "agent_source_sha256": hashlib.sha256(prompt_src.encode()).hexdigest(),
        "corpus_manifest": "corpus/manifest_equibles.json",
        "target": args.target,
        "scored_as": ("v1 EarningsBrief, one brief per call, every question for that call judged against it"
                      if not per_question else "v1.1 CallAnswer, one run per question"),
        "summary": summary,
        "agent_runs": {k: {"status": r["status"], "run": r["run"]} for k, r in results.items()},
        "questions": scored,
        "outputs": {k: r["output"] for k, r in results.items()},
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.label}-{out['date']}{'-partial' if args.limit else ''}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
