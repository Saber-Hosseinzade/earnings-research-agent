"""eval/run_eval.py, offline: transcript rendering, judge parsing, aggregation."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval", _PATH)
ev = importlib.util.module_from_spec(_spec)
sys.modules["run_eval"] = ev
_spec.loader.exec_module(ev)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_row(monkeypatch):
    """First row of the invented golden set, with the corpus pointed at the invented call."""
    monkeypatch.setattr(ev, "CORPUS_RAW", FIXTURES / "corpus")
    return json.loads((FIXTURES / "golden_set.jsonl").read_text().splitlines()[0])


def test_golden_set_passes_check_golden():
    """eval/check_golden.py on the real golden set (needs corpus/v2/raw, so skipped without it)."""
    if not ev.CORPUS_RAW.exists():
        pytest.skip("raw corpus not present")
    _spec_cg = importlib.util.spec_from_file_location("check_golden", _PATH.parent / "check_golden.py")
    cg = importlib.util.module_from_spec(_spec_cg)
    _spec_cg.loader.exec_module(cg)
    problems, rows = cg.check(ev.GOLDEN, ev.CORPUS_RAW)
    assert problems == {} and len(rows) == len(ev.load_golden())


def test_render_transcript_names_unnamed_speakers_by_role_and_caps(monkeypatch):
    turns = [{"speakerName": None, "speakerRole": "Operator", "text": "Welcome."},
             {"speakerName": "Lisa Su", "speakerRole": "CEO", "text": "x" * 50}]
    text, truncated = ev.render_transcript(turns)
    assert text.startswith("Operator (Operator): Welcome.") and not truncated
    monkeypatch.setattr(ev, "MAX_TRANSCRIPT_CHARS", 40)
    assert ev.render_transcript(turns)[1] is True


def test_parse_judge_accepts_json_inside_prose_and_rejects_bad_values():
    ok = 'Here: {"has_claim": true, "task": "partial", "grounding": "supported", "note": "n"}'
    assert ev.parse_judge(ok)["task"] == "partial"
    with pytest.raises(ValueError):
        ev.parse_judge('{"has_claim": true, "task": "great", "grounding": "supported"}')
    with pytest.raises(ValueError):
        ev.parse_judge("no json here")


def test_judge_row_retries_once_then_records_a_judge_error(monkeypatch):
    row = _fixture_row(monkeypatch)
    replies = iter(["garbage", '{"has_claim": false, "task": "missing", "grounding": "no_claim", "note": "x"}'])
    assert ev.judge_row(row, {"key_points": []}, lambda p: next(replies))["task"] == "missing"
    v = ev.judge_row(row, {"key_points": []}, lambda p: "garbage")
    assert v["judge_error"] is True
    assert ev.judge_row(row, None, lambda p: pytest.fail("no brief, no judge call"))["note"] == "no brief produced"


def _r(answerable, has_claim, task="missing", grounding="no_claim", retrieval=True):
    return {"answerable": answerable, "retrieval": retrieval,
            "verdict": {"has_claim": has_claim, "task": task, "grounding": grounding}}


def test_aggregate_separates_the_dimensions():
    rows = [_r(True, True, "correct", "supported"), _r(True, False), _r(True, True, "partial", "unsupported", False),
            _r(False, False), _r(False, True, grounding="unsupported")]
    s = ev.aggregate(rows)
    assert s["answerable"] == 3 and s["unanswerable"] == 2
    assert s["retrieval_right_call"] == 2
    assert s["task"] == {"correct": 1, "partial": 1, "missing": 1}
    assert s["grounding"] == {"supported": 1, "unsupported": 1, "no_claim": 1}
    assert s["false_abstentions"] == 1
    assert s["abstention_correct"] == 1 and s["answered_anyway"] == 1


def test_quote_in_call_ignores_case_spacing_and_trailing_punctuation(monkeypatch):
    row = _fixture_row(monkeypatch)
    quote = row["passage"]["quote"]
    assert ev.quote_in_call("  " + quote.upper() + ".", row)
    assert not ev.quote_in_call("Management announced a special dividend of five dollars.", row)


def test_aggregate_adds_v11_fields_only_when_present():
    base = [_r(True, True, "correct", "supported")]
    assert "quotes_given" not in ev.aggregate(base)
    rows = [dict(_r(True, True), quote_in_call=True, abstained=False),
            dict(_r(False, False), quote_in_call=None, abstained=True)]
    s = ev.aggregate(rows)
    assert s["quotes_given"] == 1 and s["quotes_found_in_call"] == 1
    # v1.1 declines by its own flag, not by the judge's has_claim
    assert s["abstention_correct"] == 1 and s["answered_anyway"] == 0 and s["false_abstentions"] == 0
    assert ev.aggregate([dict(_r(False, False), abstained=False)])["answered_anyway"] == 1


def test_write_results_keeps_agent_outputs_out_of_the_scores_file(tmp_path):
    out = {"label": "baseline-x", "date": "2026-11-08", "run_at": "t", "target": "v1.1",
           "summary": {"answerable": 1}, "questions": [{"id": "Q1", "verdict": {"task": "correct"}}]}
    outputs = {"Q1": {"answer": "a", "quote": "copied transcript text"}}
    path, outputs_path = ev.write_results(out, outputs, "baseline-x-2026-11-08", tmp_path)
    scores = json.loads(path.read_text())
    assert path.name == "baseline-x-2026-11-08.json" and outputs_path.name == "baseline-x-2026-11-08.outputs.json"
    assert "copied transcript text" not in path.read_text() and scores["summary"] == {"answerable": 1}
    assert scores["outputs_file"].startswith("baseline-x-2026-11-08.outputs.json")
    assert json.loads(outputs_path.read_text())["outputs"] == outputs


def test_outputs_files_are_git_ignored():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(["git", "check-ignore", "-q", "eval/results/any-run-2026-01-01.outputs.json"], cwd=root)
    if probe.returncode == 128:
        pytest.skip("not a git checkout")
    assert probe.returncode == 0
    assert subprocess.run(["git", "check-ignore", "-q", "eval/results/any-run-2026-01-01.json"], cwd=root).returncode == 1


# ---------------------------------------------------------------- v2.0

def test_parse_passage_id_and_gold_match():
    assert ev.parse_passage_id("AMD_2026Q2_t003_p1") == ("AMD", "2026Q2", 3)
    assert ev.parse_passage_id("AMD_2026Q2_t003") == ("AMD", "2026Q2", 3)
    assert ev.parse_passage_id("made up") is None
    row = {"ticker": "AMD", "quarter": "2026Q2", "passage": {"turn": 3}}
    assert ev.is_gold("AMD_2026Q2_t003_p2", row) and not ev.is_gold("AMD_2026Q1_t003", row)
    assert not ev.is_gold("AMD_2026Q2_t003", {"ticker": "AMD", "quarter": "2026Q2", "passage": None})


def _v2_res(covered, cited, returned, check):
    return {"output": {"covered": covered, "answer": "Revenue was $11.5 billion.", "citations": cited},
            "retrieval": {"returned": returned}, "citation_check": check}


def test_v2_entry_scores_citations_and_judges_only_verified_ones(tmp_path):
    (tmp_path / "AMD_2026Q2_t003_p1.txt").write_text("CFO: Revenue was $11.5 billion.")
    row = {"ticker": "AMD", "quarter": "2026Q2", "passage": {"turn": 3}}
    prompts = []
    ask = lambda p: prompts.append(p) or '{"support": "supported", "note": "n"}'  # noqa: E731
    res = _v2_res(True, ["AMD_2026Q2_t003_p1", "AMD_2025Q4_t001"], ["AMD_2026Q2_t009", "AMD_2026Q2_t003_p1"],
                  {"ok": False, "problems": ["cited_not_returned"], "verified": ["AMD_2026Q2_t003_p1"],
                   "unreturned": ["AMD_2025Q4_t001"]})
    e = ev.v2_entry(row, res, ask, tmp_path)
    assert e["gold_returned"] and e["gold_cited"] and e["citation_check_ok"] is False
    assert e["citation_support"]["support"] == "supported"
    assert len(prompts) == 1 and "AMD_2025Q4_t001" not in prompts[0] and "$11.5 billion" in prompts[0]


def test_v2_entry_skips_the_citation_judge_when_declined_or_text_missing(tmp_path):
    row = {"ticker": "AMD", "quarter": "2026Q2", "passage": None}
    no_judge = lambda p: pytest.fail("no citation judge call")  # noqa: E731
    e = ev.v2_entry(row, _v2_res(False, [], ["AMD_2026Q2_t009"], {"ok": True, "problems": [], "verified": []}),
                    no_judge, tmp_path)
    assert e["abstained"] and e["citation_support"] is None and e["gold_returned"] is None
    v = ev.judge_citations("a", ["AMD_2026Q2_t404"], no_judge, tmp_path)
    assert v["judge_error"] and v["support"] is None


def test_aggregate_adds_v2_citation_fields():
    def row(answerable, abstained, ok, problems, gold_ret, gold_cited, support):
        return dict(_r(answerable, not abstained), abstained=abstained, citation_check_ok=ok,
                    citation_problems=problems, gold_returned=gold_ret, gold_cited=gold_cited,
                    citation_support=support and {"support": support})
    rows = [row(True, False, True, [], True, True, "supported"),
            row(True, False, False, ["cited_not_returned"], True, False, "partial"),
            row(True, True, True, [], False, False, None),
            row(False, False, False, ["uncited_answer"], None, None, None)]
    s = ev.aggregate(rows)
    assert s["citations"] == {"answers_given": 3, "check_ok": 1, "cited_not_returned": 1, "uncited_answers": 1,
                              "support": {"supported": 1, "partial": 1, "unsupported": 0}, "support_judge_errors": 0}
    assert s["gold_turn_returned"] == 2 and s["gold_turn_cited"] == 1
    assert s["false_abstentions"] == 1 and s["answered_anyway"] == 1
    assert "quotes_given" not in s
    assert "citations" not in ev.aggregate([_r(True, True)])


def test_a_stopped_run_is_no_answer_not_a_decline():
    stopped = [dict(_r(True, False), status="stopped"), dict(_r(False, False), status="stopped"),
               dict(_r(False, False), status="ok")]
    s = ev.aggregate(stopped)
    assert s["no_answer"] == {"answerable": 1, "unanswerable": 1}
    assert s["false_abstentions"] == 0 and s["abstention_correct"] == 1 and s["answered_anyway"] == 0
