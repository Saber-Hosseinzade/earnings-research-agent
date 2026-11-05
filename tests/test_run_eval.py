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


def test_golden_passages_point_at_real_turns_with_their_quote():
    """Every answerable passage's quote is inside the turn it points to (needs corpus/equibles/raw)."""
    if not ev.CORPUS_RAW.exists():
        pytest.skip("raw corpus not present")
    for row in ev.load_golden():
        if row["answerable"]:
            assert row["passage"]["quote"].lower() in ev.evidence_text(row).lower(), row["id"]


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


def test_judge_row_retries_once_then_records_a_judge_error():
    row = ev.load_golden()[0]
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
