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
