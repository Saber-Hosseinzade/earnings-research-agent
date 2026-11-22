"""agent/answer.py, retrieval.py and citations.py (v2 single turn), offline."""

import json

import pytest

import answer
import fake_model
from citations import check_citations
from fake_model import FakeModel
from retrieval import Ledger, bedrock_retriever, make_retrieve_tool

TOOL = "retrieve_passages"
PASSAGES = {
    "AMD_2026Q2_t003_p1": "AMD | AMD 2026Q2 | CFO (CFO)\n\nGross margin expanded on data center mix.",
    "AMD_2026Q2_t009": "AMD | AMD 2026Q2 | Analyst (Analyst)\n\nCan you talk about margins?",
}


class FakeRetriever:
    def __init__(self, fail: Exception | None = None):
        self.calls, self.fail = [], fail

    def __call__(self, query, ticker, quarter, k):
        self.calls.append({"query": query, "ticker": ticker, "quarter": quarter, "k": k})
        if self.fail:
            raise self.fail
        return [{"passage_id": pid, "ticker": "AMD", "quarter": "2026Q2", "speaker": "x", "role": "x",
                 "start_seconds": 1, "score": 0.9, "text": text} for pid, text in list(PASSAGES.items())[:k]]


def model_answering(citations, covered=True, tool_args=None, fetches=1, mode="good"):
    fake_model.VALID_BRIEF = {"covered": covered, "answer": "Gross margin expanded on data center mix.",
                              "citations": citations}
    return FakeModel(mode, tool=TOOL, fetches=fetches,
                     tool_args=tool_args or {"query": "gross margin", "ticker": "AMD", "quarter": "2026Q2"})


@pytest.fixture(autouse=True)
def restore_brief(monkeypatch):
    monkeypatch.setattr(fake_model, "VALID_BRIEF", fake_model.VALID_BRIEF)


# --- citation check -------------------------------------------------------------------------------------------

def test_citation_check_passes_returned_ids():
    assert check_citations(["A", "A", "B"], ["A", "B", "C"], covered=True) == {
        "ok": True, "problems": [], "verified": ["A", "B"], "unreturned": []}


def test_citation_check_flags_ids_never_returned():
    out = check_citations(["A", "Z"], ["A"], covered=True)
    assert not out["ok"] and out["problems"] == ["cited_not_returned"] and out["unreturned"] == ["Z"]


def test_citation_check_flags_a_covered_answer_without_a_valid_citation():
    assert check_citations([], ["A"], covered=True)["problems"] == ["uncited_answer"]
    assert check_citations(["Z"], ["A"], covered=True)["problems"] == ["cited_not_returned", "uncited_answer"]


def test_not_covered_with_no_citations_is_fine():
    assert check_citations([], [], covered=False)["ok"]


# --- the tool -------------------------------------------------------------------------------------------------

def _call(tool_fn, **kw):
    return json.loads(tool_fn._tool_func(**kw))  # the undecorated function


def test_pinned_filter_overrides_the_model():
    r, ledger = FakeRetriever(), Ledger()
    t = make_retrieve_tool(r, ledger, pinned_ticker="AMD", pinned_quarter="2026Q2")
    out = _call(t, query="margins", ticker="META", quarter="2025Q4")
    assert out["ok"] and out["filter"] == {"ticker": "AMD", "quarter": "2026Q2"}
    assert r.calls[0]["ticker"] == "AMD" and r.calls[0]["quarter"] == "2026Q2"
    assert list(ledger.returned) == list(PASSAGES)
    assert "text" not in ledger.returned["AMD_2026Q2_t009"]


def test_unpinned_tool_takes_the_models_filter_and_clamps_k():
    r = FakeRetriever()
    _call(make_retrieve_tool(r, Ledger()), query="margins", ticker="amd", quarter="2026q2", k=50)
    assert r.calls[0] == {"query": "margins", "ticker": "AMD", "quarter": "2026Q2", "k": 8}


@pytest.mark.parametrize("kw", [{"query": "  "}, {"query": "m", "quarter": "Q2"}])
def test_bad_requests_never_reach_the_retriever(kw):
    r = FakeRetriever()
    out = _call(make_retrieve_tool(r, Ledger()), **kw)
    assert out["ok"] is False and out["error"]["type"] == "bad_request" and r.calls == []


def test_retriever_errors_become_typed_errors():
    class ThrottlingException(Exception):
        pass

    ledger = Ledger()
    out = _call(make_retrieve_tool(FakeRetriever(fail=ThrottlingException("slow down")), ledger), query="m")
    assert out == {"ok": False, "error": {"type": "retrieve_failed", "message": "slow down", "retryable": True}}
    assert ledger.returned == {}


def test_bedrock_retriever_builds_the_filter(monkeypatch):
    import boto3

    sent = []

    class Client:
        def retrieve(self, **kw):
            sent.append(kw["retrievalConfiguration"]["vectorSearchConfiguration"])
            return {"retrievalResults": [{"content": {"text": "hi"}, "score": 0.5,
                                          "metadata": {"passage_id": "P", "ticker": "AMD", "quarter": "2026Q2",
                                                       "start_seconds": 12.0}}]}

    monkeypatch.setattr(boto3, "client", lambda *a, **kw: Client())
    retrieve = bedrock_retriever(kb_id="KB")
    out = retrieve("q", "AMD", "2026Q2", 3)
    retrieve("q", "AMD", None, 3)
    retrieve("q", None, None, 3)
    assert out[0]["passage_id"] == "P" and out[0]["start_seconds"] == 12 and out[0]["text"] == "hi"
    assert sent[0]["filter"] == {"andAll": [{"equals": {"key": "ticker", "value": "AMD"}},
                                            {"equals": {"key": "quarter", "value": "2026Q2"}}]}
    assert sent[1]["filter"] == {"equals": {"key": "ticker", "value": "AMD"}}
    assert "filter" not in sent[2]


# --- the entrypoint -------------------------------------------------------------------------------------------

def test_answer_with_returned_citations_passes_the_check():
    r = FakeRetriever()
    out = answer.run({"question": "Why did gross margin expand?", "ticker": "amd", "quarter": "2026q2"},
                     model=model_answering(["AMD_2026Q2_t003_p1"]), retriever=r)
    assert out["status"] == "ok" and out["answer"]["covered"] is True
    assert out["citation_check"]["ok"] and out["citation_check"]["verified"] == ["AMD_2026Q2_t003_p1"]
    assert out["retrieval"]["returned"] == list(PASSAGES)
    assert out["run"]["tool_calls"] == 1


def test_an_invented_citation_is_flagged_not_dropped():
    out = answer.run({"question": "Why?", "ticker": "AMD", "quarter": "2026Q2"},
                     model=model_answering(["AMD_2026Q2_t003_p1", "AMD_2025Q4_t001"]), retriever=FakeRetriever())
    assert out["status"] == "ok"
    assert out["citation_check"]["unreturned"] == ["AMD_2025Q4_t001"]
    assert out["answer"]["citations"] == ["AMD_2026Q2_t003_p1", "AMD_2025Q4_t001"]


def test_result_carries_only_the_verified_citations_passages():
    out = answer.run({"question": "Why?", "ticker": "AMD", "quarter": "2026Q2"},
                     model=model_answering(["AMD_2026Q2_t003_p1", "AMD_2025Q4_t001"]), retriever=FakeRetriever())
    assert [p["passage_id"] for p in out["passages"]] == ["AMD_2026Q2_t003_p1"]
    assert out["passages"][0]["text"] == PASSAGES["AMD_2026Q2_t003_p1"] and out["passages"][0]["speaker"] == "x"


def test_not_covered_answer_carries_no_passages():
    out = answer.run({"question": "Why?", "ticker": "AMD", "quarter": "2026Q2"},
                     model=model_answering([], covered=False), retriever=FakeRetriever())
    assert out["passages"] == []


def test_citing_without_searching_fails_the_check():
    out = answer.run({"question": "Why?"}, model=model_answering(["AMD_2026Q2_t003_p1"], fetches=0),
                     retriever=FakeRetriever())
    assert out["retrieval"]["returned"] == []
    assert out["citation_check"]["problems"] == ["cited_not_returned", "uncited_answer"]


@pytest.mark.parametrize("payload", [
    {"ticker": "AMD", "quarter": "2026Q2"},     # no question
    {"question": "  "},
    {"question": "Why?", "ticker": "AMD", "quarter": "Q2"},
    {"question": "Why?", "quarter": "2026Q2"},  # quarter without ticker
])
def test_bad_requests_never_reach_the_model(payload):
    r = FakeRetriever()
    out = answer.run(payload, model=FakeModel("loop", tool=TOOL), retriever=r)
    assert out["status"] == "bad_request" and r.calls == []


def test_a_runaway_model_is_stopped_by_the_budget():
    out = answer.run({"question": "Why?", "ticker": "AMD"}, retriever=FakeRetriever(),
                     model=FakeModel("loop", tool=TOOL, tool_args={"query": "margins"}))
    assert out["status"] == "stopped" and out["answer"] is None and out["citation_check"] is None


# --- search limit, enforced in code ------------------------------------------------------------------------------

def test_the_tool_refuses_a_third_search_and_a_repeated_query():
    r, ledger = FakeRetriever(), Ledger()
    fn = make_retrieve_tool(r, ledger, "AMD", "2026Q2")
    assert _call(fn, query="gross margin")["ok"]
    repeat = _call(fn, query="  Gross   MARGIN ")
    assert repeat["error"]["type"] == "search_limit" and not repeat["error"]["retryable"]
    assert _call(fn, query="data center")["ok"]
    third = _call(fn, query="anything else")
    assert third["error"]["type"] == "search_limit"
    assert [c["query"] for c in r.calls] == ["gross margin", "data center"]  # refusals never reach the retriever
    assert ledger.refused == ["Gross   MARGIN", "anything else"]


def test_a_model_that_keeps_searching_gets_a_final_answer_from_the_passages_it_has():
    fake_model.VALID_BRIEF = {"covered": True, "answer": "Gross margin expanded on data center mix.",
                              "citations": ["AMD_2026Q2_t003_p1"]}
    r = FakeRetriever()
    out = answer.run({"question": "Why did gross margin expand?", "ticker": "AMD", "quarter": "2026Q2"},
                     retriever=r, model=FakeModel("loop_while_tool", tool=TOOL,
                                                  tool_args=[{"query": "margin"}, {"query": "gross margin"},
                                                             {"query": "margin drivers"}]))
    assert out["status"] == "ok" and out["error"] is None
    assert out["answer"]["citations"] == ["AMD_2026Q2_t003_p1"] and out["citation_check"]["ok"]
    assert len(r.calls) == 2                                   # two searches reached the retriever
    assert out["run"]["searches_refused"] >= 1
    assert out["run"]["stopped_reason"] == "max_model_calls"   # the loop still hit the budget ...
    assert out["run"]["final_answer"]["after"] == "max_model_calls"  # ... and the answer came after it
    assert out["run"]["final_answer"]["tool_calls"] == 0


def test_no_final_answer_when_no_passage_was_returned():
    out = answer.run({"question": "Why?", "ticker": "AMD"}, retriever=FakeRetriever(fail=RuntimeError("denied")),
                     model=FakeModel("loop_while_tool", tool=TOOL, tool_args={"query": "margins"}))
    assert out["status"] == "stopped" and out["answer"] is None and out["run"]["final_answer"] is None


# --- abbreviation expansion -----------------------------------------------------------------------------------

@pytest.mark.parametrize("query, sent", [
    ("capex", "capex (capital expenditures)"),
    ("Capex guidance for 2026", "Capex guidance for 2026 (capital expenditures)"),
    ("FCF and EPS", "FCF and EPS (free cash flow; earnings per share)"),
    ("R&D spending", "R&D spending (research and development)"),
    ("SG&A", "SG&A (selling, general and administrative expenses)"),  # not also D&A
    ("ASPs", "ASPs (average selling price)"),
    ("capex, capital expenditures", "capex, capital expenditures"),   # phrase already there
    ("capex and Capex", "capex and Capex (capital expenditures)"),    # phrase added once
    ("gross margin", "gross margin"),
    ("tampa expansion", "tampa expansion"),                           # not inside a word
])
def test_abbreviations_are_spelled_out(query, sent):
    from retrieval import expand_abbreviations
    assert expand_abbreviations(query) == sent


def test_tool_sends_the_expanded_query_and_records_both():
    r, ledger = FakeRetriever(), Ledger()
    make_retrieve_tool(r, ledger, "META", "2026Q3")(query="capex")
    assert r.calls[0]["query"] == "capex (capital expenditures)"
    assert ledger.queries[0]["query"] == "capex" and ledger.queries[0]["sent"] == "capex (capital expenditures)"
    out = json.loads(make_retrieve_tool(r, ledger, "META", "2026Q3")(query="capex"))
    assert out["error"]["type"] == "search_limit"  # a repeat is judged on the model's own words


def test_abbreviation_set_terms_are_known_and_questions_use_them():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    import eval_abbreviations as ea
    from retrieval import ABBREVIATIONS, expand_abbreviations
    rows = [json.loads(line) for line in ea.QUESTIONS.read_text().splitlines() if line.strip()]
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["term"] in ABBREVIATIONS and expand_abbreviations(r["question"]) != r["question"]
    assert ea.topic_pattern("capex").search("capital expenditures were $34 billion")
    assert ea.topic_pattern("R&D").search("our R&D spend") and not ea.topic_pattern("D&A").search("our R&D spend")
