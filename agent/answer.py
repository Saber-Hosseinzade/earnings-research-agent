"""v2 (single turn): answer one question from retrieved transcript passages, with checked citations.

v1.1 (ask.py) reads one whole call per question. This entrypoint searches the
Knowledge Base instead, cites the passages it used by ID, and checks those IDs
in code against what the retriever returned in this run (``citations.py``).
ask.py and main.py are untouched, so the v1.0 and v1.1 baselines stay reproducible.

    payload: {"question": "Why did gross margin expand?", "ticker": "AMD", "quarter": "2026Q2"}
             ticker and quarter are optional; when given they pin the search filter in code.
    returns: {"status": "ok" | "stopped" | "invalid_output" | "bad_request" | "blocked" | "guardrail_error",
              "answer": {"covered", "answer", "citations"} | null,
              "citation_check": {"ok", "problems", "verified", "unreturned"} | null,
              "retrieval": {"queries": [...], "returned": [passage ids]},
              "run": {budget counters, latency_ms, model_id, prompts: {name: source, id, version, sha256}},
              "guardrail": {"input": check, "output": check | null} | null,
              "audit": {"turn_id", "written", "error"}}   (only with an audit writer)
    payload may also carry session_id, user_id and role; they go to the audit record only.
    From session.run_turn: history (earlier turns, shown to the model before the question) and
    context (how ticker and quarter were resolved; copied to the result and the audit record).

Guardrail (``guard.py``), when a guard is passed: the question is checked before the
agent runs (advice or prompt attack -> "blocked", no agent run; PII masked), and the
answer after it, with the cited passages as grounding source (advice -> "blocked";
PII masked; low grounding score flagged, answer kept). Without a guard, nothing changes.

Final answer after a budget stop: if the search loop ends on the run budget
with passages in hand, one more model call, with no tools, answers from those
passages (``final_answer``). The loop's searches are not repeated. ``run.final_answer``
says when this happened, so evaluation can count it. With no passages returned,
the run stays ``stopped``: "not discussed" would be wrong when the search failed.

A failed citation check does not hide the answer: it is returned with the flags,
so the caller (UI, evaluation, audit log) decides what to show.
This is a research aid: the answer can be wrong, and it is never investment advice.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from pydantic import BaseModel, Field
from strands import Agent
from strands.models import BedrockModel
from strands.types.exceptions import StructuredOutputException

import audit
import prompts
import session
from budget import RunBudget
from citations import check_citations
from guard import GUARD_ERROR_MESSAGE, PROMPT_ATTACK_MESSAGE, Check, Guard
from retrieval import Ledger, Retriever, bedrock_retriever, make_retrieve_tool

# Prompts live in agent/prompts/*.txt and are published to Bedrock Prompt Management
# (prompts.py, infra/prompt_stack.py). Deployed, the pinned version is loaded; offline, the file.

ADVICE_TOPIC = "topic:InvestmentAdvice"  # infra/guardrail_stack.py
FINAL_MAX_TOKENS = 15_000  # passages from at most two searches; a stopped run can use 40K + this


class CitedAnswer(BaseModel):
    covered: bool = Field(description="False if the retrieved passages do not answer the question")
    answer: str = Field(description="The answer, or one sentence saying the calls did not discuss it")
    citations: list[str] = Field(description="passage_id of each passage the answer relies on; empty if not covered")


def build_agent(budget: RunBudget, retrieve_tool: Any, model: Any | None = None) -> Agent:
    if model is None:
        model = BedrockModel(model_id=os.environ["MODEL_ID"], temperature=0.2)
    return Agent(model=model, system_prompt=prompts.get("answer_system").text, tools=[retrieve_tool],
                 hooks=[budget], callback_handler=None)


def final_answer(prompt: str, ledger: Ledger, model: Any | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """One answer from the passages already returned, with no tools. Own small budget: one answer, one retry."""
    if model is None:
        model = BedrockModel(model_id=os.environ["MODEL_ID"], temperature=0.2)
    passages = "\n\n".join(f"passage_id: {pid}\n{text}" for pid, text in ledger.texts.items())
    budget = RunBudget(max_model_calls=2, max_tool_calls=0, max_total_tokens=FINAL_MAX_TOKENS)
    agent = Agent(model=model, system_prompt=prompts.get("answer_final").text, tools=[], hooks=[budget],
                  callback_handler=None)
    result = agent(f"{prompt}\n\nPassages:\n\n{passages}", structured_output_model=CitedAnswer)
    return result.structured_output.model_dump(), budget.summary()


def run(payload: dict[str, Any], model: Any | None = None, budget: RunBudget | None = None,
        retriever: Retriever | None = None, guard: Guard | None = None,
        audit_writer: audit.Writer | None = None) -> dict[str, Any]:
    """One turn. With an audit writer, every turn (rejected and blocked ones too) gets one record
    (audit.py); result["audit"] says whether it was written."""
    result = _with_context(_run(payload, model, budget, retriever, guard), payload)
    if audit_writer is not None:
        result["audit"] = audit.write(audit_writer, result, payload)
    return result


def _run(payload: dict[str, Any], model: Any | None, budget: RunBudget | None,
         retriever: Retriever | None, guard: Guard | None) -> dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    ticker = str(payload.get("ticker") or "").strip().upper() or None
    quarter = str(payload.get("quarter") or "").strip().upper() or None
    if not question or (quarter and not re.match(r"^\d{4}Q[1-4]$", quarter)) or (quarter and not ticker):
        return {"status": "bad_request", "answer": None, "citation_check": None, "retrieval": None, "run": {}, "guardrail": None,
                "error": "need a question; ticker and quarter (like 2026Q2) are optional, quarter only with a ticker"}

    guard_info: dict[str, Any] | None = None
    if guard is not None:
        gin = guard.check_input(question)
        guard_info = {"input": gin.to_dict(), "output": None}
        history = payload.get("history") or []
        if gin.blocked and gin.blocked_by == [ADVICE_TOPIC] and history:
            # A bare follow-up ("What about Meta?") can read as an advice request on its own. Checked again
            # with the previous question in front of it; real advice ("Should I buy it?") stays advice there.
            retry = guard.check_input(session.follow_up_text(question, history))
            guard_info["input_in_context"] = retry.to_dict()
            if not retry.blocked and not retry.error:
                gin = retry if retry.text else Check()  # masked: the agent sees the masked text in context
        if gin.blocked or gin.error:
            message = (GUARD_ERROR_MESSAGE if not gin.blocked
                       else PROMPT_ATTACK_MESSAGE if "content:PROMPT_ATTACK" in gin.blocked_by else gin.text)
            return {"status": "blocked" if gin.blocked else "guardrail_error",
                    "answer": {"covered": False, "answer": message, "citations": []},
                    "citation_check": None, "retrieval": None, "run": {}, "guardrail": guard_info,
                    "error": gin.error}
        if gin.text:  # PII masked: the agent only ever sees the masked question
            question = gin.text

    ledger = Ledger()
    retrieve_tool = make_retrieve_tool(retriever or bedrock_retriever(), ledger, ticker, quarter)
    budget = budget or RunBudget()
    agent = build_agent(budget, retrieve_tool, model)

    # Earlier turns (session.py) go before the question; a turn without history gets the same prompt as before.
    prompt = "".join([session.history_block(payload.get("history") or []),
                      f"Ticker: {ticker}\n" if ticker else "", f"Quarter: {quarter}\n" if quarter else "",
                      f"Question: {question}"])
    started = time.perf_counter()
    status, answer, check, error = "ok", None, None, None
    try:
        result = agent(prompt, structured_output_model=CitedAnswer)
        answer = result.structured_output.model_dump()
        check = check_citations(answer["citations"], ledger.returned, answer["covered"])
    except StructuredOutputException as exc:
        status = "stopped" if budget.stopped_reason else "invalid_output"
        error = str(exc)[:300]

    run_info = budget.summary()
    run_info["searches_refused"] = len(ledger.refused)
    run_info["final_answer"] = None
    if status == "stopped" and ledger.returned:
        try:
            answer, final_run = final_answer(prompt, ledger, agent.model)
            check = check_citations(answer["citations"], ledger.returned, answer["covered"])
            status, error = "ok", None
            run_info["final_answer"] = {"after": budget.stopped_reason, **final_run}
            run_info["tokens_used"] += final_run["tokens_used"]
        except StructuredOutputException as exc:
            error = f"{error} | final answer failed: {str(exc)[:200]}"

    run_info["latency_ms"] = round((time.perf_counter() - started) * 1000)
    run_info["model_id"] = (agent.model.get_config() or {}).get("model_id")
    run_info["prompts"] = {"answer_system": prompts.get("answer_system").info(),
                           **({"answer_final": prompts.get("answer_final").info()} if run_info["final_answer"] else {})}
    if budget.stopped_reason and status == "ok" and run_info["final_answer"] is None:
        status = "stopped"

    if guard is not None and answer is not None:
        cited = [ledger.texts[pid] for pid in (check or {}).get("verified", []) if pid in ledger.texts]
        gout = guard.check_output(question, answer["answer"], cited if answer["covered"] else [])
        guard_info["output"] = gout.to_dict()
        if gout.blocked:
            answer = {"covered": False, "answer": gout.text, "citations": []}
            status = "blocked"
        elif gout.text:  # PII masked
            answer = {**answer, "answer": gout.text}

    return {"status": status, "question": question, "answer": answer, "citation_check": check,
            "retrieval": {"queries": ledger.queries, "returned": list(ledger.returned)},
            "run": run_info, "guardrail": guard_info, "error": error}


def _with_context(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("context") is not None:
        result["context"] = payload["context"]
    return result
