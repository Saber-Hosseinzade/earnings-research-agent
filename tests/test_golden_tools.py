"""eval/check_golden.py, eval/find_passage.py and corpus/calls.json, offline on an invented call."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CORPUS = FIXTURES / "corpus"
GOLDEN = FIXTURES / "golden_set.jsonl"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cg = _load("check_golden", "eval/check_golden.py")
fp = _load("find_passage", "eval/find_passage.py")
ROWS = [json.loads(line) for line in GOLDEN.read_text().splitlines()]
ANSWERABLE, UNANSWERABLE = ROWS


def _with(row, **changes):
    r = copy.deepcopy(row)
    for key, value in changes.items():
        if key.startswith("passage_"):
            r["passage"][key.removeprefix("passage_")] = value
        else:
            r[key] = value
    return r


def test_fixture_golden_set_passes():
    problems, rows = cg.check(GOLDEN, CORPUS)
    assert problems == {} and len(rows) == 2
    assert cg.main(["--golden", str(GOLDEN), "--corpus", str(CORPUS)]) == 0


@pytest.mark.parametrize("changes, expected", [
    ({"passage_turn": 1}, "quote not found in turn 1; found in turn(s) [2]"),
    ({"passage_turn": 9}, "outside 0..4"),
    ({"passage_quote": "Gross margin was 45%"}, "quote not found in turn 2"),
    ({"passage_speaker": "Ana Ortiz"}, "speaker 'Ana Ortiz' != turn's 'Ben Kato'"),
    ({"passage_role": "CEO"}, "role 'CEO' != turn's 'CFO'"),
    ({"passage_start_seconds": 140}, "start_seconds 140 != turn's 141"),
    ({"kind": "opinion"}, "kind 'opinion' is not one of"),
    ({"key_facts": []}, "answerable row has no key_facts"),
    ({"passage_quote": "x" * 201}, "quote is 201 chars"),
])
def test_answerable_row_problems_are_reported(changes, expected):
    errs = cg.check_row(_with(ANSWERABLE, **changes), CORPUS)
    assert any(expected in e for e in errs), errs


def test_quote_that_matches_several_turns_is_rejected():
    row = _with(ANSWERABLE, passage_turn=1, passage_speaker="Ana Ortiz", passage_role="CEO",
                passage_start_seconds=30, passage_quote="Thank you")
    assert any("also in turn(s) [0, 3]" in e for e in cg.check_row(row, CORPUS))


def test_quote_check_ignores_case_spacing_and_curly_quotes():
    row = _with(ANSWERABLE, passage_quote="GROSS  margin was 41%,\ndown one point")
    assert cg.check_row(row, CORPUS) == []


def test_unanswerable_row_is_rejected_when_the_call_covers_the_topic():
    row = _with(UNANSWERABLE, absence_check={"pattern": "factory", "hits": 0})
    errs = cg.check_row(row, CORPUS)
    assert any("3 hit(s)" in e for e in errs) and any("recomputed 3" in e for e in errs)
    assert any("passage null" in e for e in cg.check_row(_with(UNANSWERABLE, passage={}), CORPUS))
    assert any("not a valid regex" in e for e in cg.check_row(
        _with(UNANSWERABLE, absence_check={"pattern": "(", "hits": 0}), CORPUS))


def test_missing_fields_missing_call_and_duplicate_ids(tmp_path):
    assert cg.check_row({"id": "x"}, CORPUS)[0].startswith("missing fields:")
    assert "no transcript file DEMO_2026Q2.json" in cg.check_row(_with(ANSWERABLE, quarter="2026Q2"), CORPUS)[0]
    golden = tmp_path / "g.jsonl"
    golden.write_text(GOLDEN.read_text() + json.dumps(ANSWERABLE) + "\nnot json\n")
    problems, _ = cg.check(golden, CORPUS)
    assert problems["DEMO-01"] == ["duplicate id"] and "not valid JSON" in problems["line 4"][0]
    assert cg.main(["--golden", str(golden), "--corpus", str(CORPUS)]) == 1


def test_find_passage_prints_a_passage_that_passes_the_check(capsys):
    assert fp.main(["DEMO", "2026Q1", r"\$1\.2 billion", "--corpus", str(CORPUS)]) == 0
    passage = json.loads(capsys.readouterr().out.strip())["passage"]
    assert passage == {"turn": 2, "speaker": "Ben Kato", "role": "CFO", "start_seconds": 141,
                       "quote": "Revenue grew 18% year over year to $1.2 billion."}
    assert cg.check_row(_with(ANSWERABLE, passage=passage), CORPUS) == []


def test_find_passage_absent_and_list(capsys):
    assert fp.main(["DEMO", "2026Q1", "--absent", "dividend|buyback", "--corpus", str(CORPUS)]) == 0
    assert json.loads(capsys.readouterr().out) == {"absence_check": {"pattern": "dividend|buyback", "hits": 0}}
    assert fp.main(["DEMO", "2026Q1", "--absent", "porto", "--corpus", str(CORPUS)]) == 1
    capsys.readouterr()
    assert fp.main(["DEMO", "2026Q1", "--list", "--corpus", str(CORPUS)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5 and "Chris Lee (Analyst)" in lines[3] and " -s " in lines[3]
    assert fp.main(["DEMO", "2026Q9", "x", "--corpus", str(CORPUS)]) == 2


def test_sentence_around_keeps_decimals():
    text = "Revenue was $11.5 billion. Margin was 56%."
    assert fp.sentence_around(text, text.index("11.5")) == "Revenue was $11.5 billion."
    assert fp.sentence_around(text, text.index("56")) == "Margin was 56%."


def test_calls_file_defines_the_corpus(tmp_path):
    pc = sys.modules.get("pull_corpus") or _load("pull_corpus", "scripts/pull_corpus.py")
    f = tmp_path / "calls.json"
    f.write_text(json.dumps({"frozen": "2026-01-01", "calls": [
        {"ticker": "DEMO", "label": "2026Q1", "fiscal_year": 2026, "fiscal_quarter": 1}]}))
    calls, frozen = pc.load_calls(f)
    assert frozen == "2026-01-01" and [(c.ticker, c.label, c.fiscal_year, c.fiscal_quarter) for c in calls] == [
        ("DEMO", "2026Q1", 2026, 1)]
