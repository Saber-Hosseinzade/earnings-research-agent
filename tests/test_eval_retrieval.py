"""eval/eval_retrieval.py, offline: hit rule, metrics and the two modes, with a fake retriever."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_retrieval", ROOT / "eval" / "eval_retrieval.py")
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)

ROW = {"id": "X-01", "ticker": "AMD", "quarter": "2026Q2", "answerable": True, "kind": "fact",
       "question": "q", "passage": {"turn": 3}}


def hit(pid, ticker="AMD", quarter="2026Q2", turn=3, score=0.9):
    return {"passage_id": pid, "ticker": ticker, "quarter": quarter, "turn": turn, "score": score}


def test_hit_needs_same_call_and_turn_and_counts_any_part():
    results = [hit("a", quarter="2026Q1"), hit("b", turn=4), hit("AMD_2026Q2_t003_p2")]
    assert er.first_hit_rank(ROW, results) == 3
    assert er.first_hit_rank(ROW, results[:2]) is None


def test_summarise_recall_and_mrr():
    s = er.summarise([1, 2, None, 6])
    assert s["recall@1"] == {"hits": 1, "rate": 0.25}
    assert s["recall@3"]["hits"] == 2 and s["recall@5"]["hits"] == 2 and s["recall@10"]["hits"] == 3
    assert s["mrr"] == round((1 + 1 / 2 + 1 / 6) / 4, 3)


def test_score_filters_only_in_filtered_mode_and_skips_unanswerable():
    calls = []

    def fake(question, ticker, quarter, k):
        calls.append((ticker, quarter))
        return [hit("x", quarter="2026Q1"), hit("y")] if ticker is None else [hit("y")]

    golden = [ROW, {**ROW, "id": "X-99", "answerable": False}]
    per_q, summary = er.score(golden, fake)
    assert calls == [(None, None), ("AMD", "2026Q2")]
    assert per_q[0]["unfiltered"]["rank"] == 2 and per_q[0]["filtered"]["rank"] == 1
    assert per_q[0]["unfiltered"]["top1_right_call"] is False
    assert summary["skipped_unanswerable"] == 1
    assert summary["unfiltered"]["top1_right_call"] == 0
    assert "text" not in str(per_q)  # IDs and scores only: results file is committed


def test_kb_docs_hash_changes_with_content(tmp_path):
    (tmp_path / "a.txt").write_text("one")
    h1 = er.kb_docs_sha256(tmp_path)
    (tmp_path / "a.txt").write_text("two")
    assert er.kb_docs_sha256(tmp_path) != h1
    assert er.kb_docs_sha256(tmp_path / "missing") is None
