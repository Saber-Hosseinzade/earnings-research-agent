#!/usr/bin/env python3
"""Score the v1 agent on the golden set. Retrieval, grounding, task and abstention
are scored separately (PROJECT_CONTEXT.md 8.7).

v1 takes {ticker, quarter} and returns an EarningsBrief, not an answer to a
question. So v1.0 is scored as shipped: it runs once per call, and every golden
question for that call is scored against that one brief. Low scores on specific
questions are the honest baseline, not a bug in the harness.

Transcripts come from corpus/v2/raw/ through a local tool that replaces
the Lambda tool (same "Speaker (Role): text" format, same 100,000-character cap),
so a run makes no transcript API requests. agent/ is not modified.

    export MODEL_ID=eu.amazon.nova-2-lite-v1:0     # agent model
    export AWS_PROFILE=sandbox AWS_REGION=eu-central-1
    python eval/run_eval.py --dry-run
    python eval/run_eval.py --limit 3              # smoke test: first 3 questions
    python eval/run_eval.py                        # full run, writes eval/results/
                                                   # (<label>-<date>.json = scores, committed;
                                                   #  .outputs.json = agent outputs, git-ignored)

Scores by dimension:
  retrieval   the right call was fetched (v1 definition; the real retrieval
              metric starts with the Knowledge Base)
  task        judge: does the brief contain the expected answer
              (correct / partial / missing)
  grounding   judge: are the brief's claims about the question supported by the
              gold transcript turn (supported / unsupported / no_claim)
  abstention  unanswerable questions only: did the brief avoid answering
  false abstentions  answerable questions whose brief has no relevant claim

v2.0 (agent/answer.py) answers from Knowledge Base passages and cites them by
ID. It gets the same ticker and quarter as v1.1 (pinned as the search filter),
and the same judge prompt, so task, grounding and abstention compare directly.
It adds citation scores (PROJECT_CONTEXT.md 8.11):
  citation check   code: every cited ID was returned in that run (agent/citations.py)
  gold returned    code: a passage from the golden turn came back from the search
  gold cited       code: the answer cited a passage from the golden turn
  citation support judge, second prompt: do the cited passages back the answer
                   (supported / partial / unsupported); covered answers with a
                   valid citation only. Passage text from build/kb_docs (local).
Retrieval (right call) for v2.0 = a returned passage is from the golden call.
v2.0 needs KNOWLEDGE_BASE_ID, or reads it from the EarningsResearchAgentKB stack.

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
CORPUS_RAW = ROOT / "corpus" / "v2" / "raw"
RESULTS = ROOT / "eval" / "results"
MAX_TRANSCRIPT_CHARS = 100_000  # same as the Lambda tool's default
KB_DOCS = ROOT / "build" / "kb_docs"
DEFAULT_JUDGE = "eu.amazon.nova-pro-v1:0"
MAX_JUDGE_CALLS = 40  # cost cap on questions per run: 34 questions + a few retries
MAX_JUDGE_REQUESTS = 90  # hard cap on judge API calls per run (v2.0: up to 2 prompts per question, plus retries)
PASSAGE_ID = re.compile(r"^([A-Z]+)_(\d{4}Q[1-4])_t(\d{3})(?:_p\d+)?$")
TARGET_MODULE = {"v1.0": "main", "v1.1": "ask", "v2.0": "answer"}


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
    """Run an unmodified agent module (v1.0 main, v1.1 ask) with a local transcript tool in place of the Lambda.
    v2.0 (answer) runs as shipped against the deployed Knowledge Base."""
    sys.path.insert(0, str(ROOT / "agent"))
    import importlib  # noqa: PLC0415

    v1 = importlib.import_module(TARGET_MODULE[target])  # needs MODEL_ID etc. at call time
    if target == "v2.0":
        result = v1.run(payload)
        returned = (result.get("retrieval") or {}).get("returned", [])
        result["fetched"] = sorted({parse_passage_id(pid)[:2] for pid in returned if parse_passage_id(pid)})
        result["output"] = result.get("answer")
        return result
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


def parse_passage_id(pid: str) -> tuple[str, str, int] | None:
    """AMD_2026Q2_t003_p1 -> ("AMD", "2026Q2", 3); None if it is not a passage ID."""
    m = PASSAGE_ID.match(pid or "")
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def is_gold(pid: str, row: dict[str, Any]) -> bool:
    p = row.get("passage")
    return bool(p) and parse_passage_id(pid) == (row["ticker"], row["quarter"], p["turn"])


def passage_text(pid: str, folder: Path | None = None) -> str | None:
    f = (folder or KB_DOCS) / f"{pid}.txt"
    return f.read_text() if PASSAGE_ID.match(pid or "") and f.is_file() else None


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


CITATION_PROMPT = """You are checking whether a research assistant's answer is backed by the passages it cites.
Be strict and literal. Judge only against the passages below, not against what you know.

ANSWER: {answer}

CITED PASSAGES:
{passages}

Return ONLY a JSON object with these keys:
"support": "supported" if every factual claim in the answer is stated in the cited passages, "partial" if some claims are and some are not, "unsupported" if none are.
"note": one short sentence on why."""


def judge_citations(answer: str, pids: list[str], ask: Callable[[str], str],
                    folder: Path | None = None) -> dict[str, Any]:
    texts = [(pid, passage_text(pid, folder)) for pid in pids]
    missing = [pid for pid, t in texts if t is None]
    if missing:
        return {"support": None, "note": f"passage text not found locally: {missing}", "judge_error": True}
    prompt = CITATION_PROMPT.format(answer=answer, passages="\n\n".join(f"[{pid}]\n{t.strip()}" for pid, t in texts))
    last: Exception | None = None
    for _ in range(2):
        try:
            m = re.search(r"\{.*\}", ask(prompt), re.S)
            v = json.loads(m.group(0)) if m else {}
            if v.get("support") in {"supported", "partial", "unsupported"}:
                return v
            last = ValueError(f"citation verdict malformed: {v}")
        except json.JSONDecodeError as exc:
            last = exc
    return {"support": None, "note": f"judge failed: {last}", "judge_error": True}


def v2_entry(row: dict[str, Any], res: dict[str, Any], ask: Callable[[str], str],
             folder: Path | None = None) -> dict[str, Any]:
    """v2.0 fields for one question: abstention flag and the citation scores."""
    out_ = res["output"]
    returned = (res.get("retrieval") or {}).get("returned", [])
    check = res.get("citation_check") or {}
    entry: dict[str, Any] = {
        "abstained": not out_["covered"],
        "returned": returned,
        "cited": out_["citations"],
        "citation_check_ok": check.get("ok"),
        "citation_problems": check.get("problems", []),
        "gold_returned": any(is_gold(p, row) for p in returned) if row.get("passage") else None,
        "gold_cited": any(is_gold(p, row) for p in check.get("verified", [])) if row.get("passage") else None,
        "citation_support": None,
    }
    if out_["covered"] and check.get("verified"):
        entry["citation_support"] = judge_citations(out_["answer"], check["verified"], ask, folder)
    return entry


def declined(r: dict[str, Any]) -> bool:
    """Did the system decline to answer? v1.1 says so itself (covered_in_call=false). A v1 brief has no way to
    say it, so there it means the brief makes no claim about the question. The judge's has_claim is not used
    for v1.1: it counts 'the call did not discuss this' as a claim.
    A run that produced no output (stopped by the budget, invalid output) did not decline: it is counted
    under no_answer instead, in both groups."""
    if no_answer(r):
        return False
    return r["abstained"] if "abstained" in r else not r["verdict"]["has_claim"]


def no_answer(r: dict[str, Any]) -> bool:
    return r.get("status", "ok") != "ok"


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
        "no_answer": {"answerable": sum(no_answer(r) for r in ans), "unanswerable": sum(no_answer(r) for r in una)},
        "abstention_correct": sum(declined(r) for r in una),
        "answered_anyway": sum(not declined(r) and not no_answer(r) for r in una),
        "unanswerable_answers_unsupported": sum(r["verdict"]["grounding"] == "unsupported" for r in una),
        "judge_errors": sum(bool(r["verdict"].get("judge_error")) for r in rows),
        **_v11_extras(rows),
        **_v2_extras(rows),
    }


def _v2_extras(rows: list[dict[str, Any]]) -> dict[str, Any]:
    v2 = [r for r in rows if "citation_check_ok" in r]
    if not v2:
        return {}
    ans = [r for r in v2 if r["answerable"]]
    covered = [r for r in v2 if not r["abstained"]]
    support = Counter((r["citation_support"] or {}).get("support") for r in v2 if r["citation_support"])
    return {
        "citations": {
            "answers_given": len(covered),
            "check_ok": sum(bool(r["citation_check_ok"]) for r in covered),
            "cited_not_returned": sum("cited_not_returned" in r["citation_problems"] for r in v2),
            "uncited_answers": sum("uncited_answer" in r["citation_problems"] for r in v2),
            "support": {k: support.get(k, 0) for k in ("supported", "partial", "unsupported")},
            "support_judge_errors": sum(bool((r["citation_support"] or {}).get("judge_error")) for r in v2),
        },
        "gold_turn_returned": sum(bool(r["gold_returned"]) for r in ans),
        "gold_turn_cited": sum(bool(r["gold_cited"]) for r in ans),
    }


def _v11_extras(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in rows if "citation_check_ok" not in r]  # v2.0 rows have no quote
    quoted = [r for r in rows if r.get("quote_in_call") is not None]
    if not quoted and not any("abstained" in r for r in rows):
        return {}
    return {
        "quotes_given": len(quoted),
        "quotes_found_in_call": sum(r["quote_in_call"] for r in quoted),
    }


# ---------------------------------------------------------------- judge client


def bedrock_judge(model_id: str, max_requests: int = MAX_JUDGE_REQUESTS) -> Callable[[str], str]:
    import boto3  # noqa: PLC0415

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "eu-central-1"))
    used = [0]

    def ask(prompt: str) -> str:
        used[0] += 1
        if used[0] > max_requests:
            raise RuntimeError(f"judge request cap of {max_requests} reached")
        resp = client.converse(modelId=model_id, messages=[{"role": "user", "content": [{"text": prompt}]}],
                               inferenceConfig={"temperature": 0.0, "maxTokens": 400})
        return resp["output"]["message"]["content"][0]["text"]

    return ask


# ---------------------------------------------------------------- main


def kb_docs_sha256(folder: Path | None = None) -> str | None:
    """Same hash as eval/eval_retrieval.py: ties the run to the passages that were ingested."""
    folder = folder or KB_DOCS
    if not folder.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(folder.iterdir()):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_results(out: dict[str, Any], outputs: dict[str, Any], stem: str,
                  folder: Path | None = None) -> tuple[Path, Path]:
    """Write scores to <stem>.json (committed) and the agent's outputs to <stem>.outputs.json.

    The outputs hold text the agent copied from the transcripts, so that file is
    git-ignored (eval/results/*.outputs.json) and stays local. The scores file
    names it, so a local run can be traced back to what the agent said.
    """
    folder = folder or RESULTS
    folder.mkdir(parents=True, exist_ok=True)
    path, outputs_path = folder / f"{stem}.json", folder / f"{stem}.outputs.json"
    scores = {**out, "outputs_file": f"{outputs_path.name} (git-ignored, local only)"}
    path.write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n")
    header = {k: out[k] for k in ("label", "date", "run_at", "target") if k in out}
    outputs_path.write_text(json.dumps({**header, "outputs": outputs}, indent=2, ensure_ascii=False) + "\n")
    return path, outputs_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="show the plan; no model calls")
    ap.add_argument("--limit", type=int, help="score only the first N questions")
    ap.add_argument("--target", choices=list(TARGET_MODULE), default="v1.0",
                    help="v1.0: brief, one run per call. v1.1: question in, answer out, one run per question. "
                         "v2.0: question in, answer from KB passages with checked citations, one run per question")
    ap.add_argument("--label", help="result file prefix (default baseline-<target>)")
    args = ap.parse_args(argv)

    golden = load_golden()[: args.limit]
    args.label = args.label or f"baseline-{args.target}"
    per_question = args.target != "v1.0"
    calls = sorted({(r["ticker"], r["quarter"]) for r in golden})
    judge_id = os.environ.get("JUDGE_MODEL_ID", DEFAULT_JUDGE)
    if args.dry_run:
        print(f"{len(golden)} questions, {len(golden) if per_question else len(calls)} agent runs, {len(golden)} judge calls (cap {MAX_JUDGE_CALLS})")
        print(f"agent MODEL_ID={os.environ.get('MODEL_ID', '(unset)')}  judge={judge_id}")
        if args.target == "v2.0":
            print(f"plus up to {len(golden)} citation-support judge calls; judge requests capped at {MAX_JUDGE_REQUESTS}")
        return 0
    if "MODEL_ID" not in os.environ:
        print("MODEL_ID is not set", file=sys.stderr)
        return 2
    if len(golden) > MAX_JUDGE_CALLS:
        print(f"{len(golden)} questions exceeds the judge cap of {MAX_JUDGE_CALLS}", file=sys.stderr)
        return 2

    if args.target == "v2.0" and not os.environ.get("KNOWLEDGE_BASE_ID"):
        import boto3  # noqa: PLC0415

        stack = boto3.client("cloudformation", region_name=os.environ.get("AWS_REGION", "eu-central-1")).describe_stacks(
            StackName="EarningsResearchAgentKB")["Stacks"][0]
        os.environ["KNOWLEDGE_BASE_ID"] = next(o["OutputValue"] for o in stack["Outputs"]
                                               if o["OutputKey"] == "KnowledgeBaseId")

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
        if args.target == "v2.0" and res["output"]:
            entry.update(v2_entry(row, res, ask))
        elif per_question and res["output"]:
            out_ = res["output"]
            entry["abstained"] = not out_["covered_in_call"]
            entry["quote_in_call"] = quote_in_call(out_["quote"], row) if out_["quote"] else None
        scored.append(entry)
        print(f"judge {row['id']:8} task={verdict['task']:8} grounding={verdict['grounding']:10} claim={verdict['has_claim']}")

    summary = aggregate(scored)
    prompt_src = (ROOT / "agent" / f"{TARGET_MODULE[args.target]}.py").read_text()
    out = {
        "label": args.label,
        "date": date.today().isoformat(),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent_model_id": os.environ["MODEL_ID"],
        "judge_model_id": judge_id,
        "git_head": git("rev-parse", "--short", "HEAD"),
        "git_tag_v1.0": git("rev-parse", "--short", "v1.0"),
        "agent_source_sha256": hashlib.sha256(prompt_src.encode()).hexdigest(),
        "corpus_manifest": "corpus/manifest_v2.json",
        "target": args.target,
        "scored_as": ("v1 EarningsBrief, one brief per call, every question for that call judged against it"
                      if not per_question else {"v1.1": "v1.1 CallAnswer, one run per question",
                                                "v2.0": "v2.0 CitedAnswer from KB passages, one run per question"}[args.target]),
        "summary": summary,
        "agent_runs": {k: {"status": r["status"], "run": r["run"]} for k, r in results.items()},
        **({"kb_docs_sha256": kb_docs_sha256()} if args.target == "v2.0" else {}),
        "questions": scored,
    }
    stem = f"{args.label}-{out['date']}{'-partial' if args.limit else ''}"
    path, outputs_path = write_results(out, {k: r["output"] for k, r in results.items()}, stem)
    print(json.dumps(summary, indent=2))
    print(f"wrote {path.relative_to(ROOT)} (scores, committed) and {outputs_path.relative_to(ROOT)} (agent outputs, git-ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
