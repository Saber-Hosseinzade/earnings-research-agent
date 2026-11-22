"""scripts/pull_corpus.py, offline: label check, key rotation, manifest."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pull_corpus.py"
_spec = importlib.util.spec_from_file_location("pull_corpus", _PATH)
pc = importlib.util.module_from_spec(_spec)
sys.modules["pull_corpus"] = pc  # dataclasses look the module up here
_spec.loader.exec_module(pc)

CRDO_Q1 = pc.Call("CRDO", "2027Q1", 1, 2027)


def seg(text):
    return {"speaker": "Operator", "title": "", "content": text}


@pytest.mark.parametrize("text, expected", [
    ("Welcome to the AMD third quarter 2026 earnings call.", (3, 2026)),
    ("Credo's first quarter of fiscal 2027 conference call.", (1, 2027)),
    ("Our Q4 2025 results.", (4, 2025)),
    ("Results for Q1 FY27.", (1, 2027)),
    ("Our fiscal 2027 first quarter call.", (1, 2027)),
])
def test_stated_period_reads_common_openings(text, expected):
    assert pc.stated_period([seg(text)])[:2] == expected


def test_label_check_flags_a_mismatch_and_unknown():
    assert pc.label_check(CRDO_Q1, [seg("fourth quarter of fiscal 2026")])[0] == "mismatch"
    assert pc.label_check(CRDO_Q1, [seg("Good afternoon, everyone.")]) == ("unknown", None)


def _replies(monkeypatch, bodies):
    """Make http_get return these bodies in order, recording the key used."""
    used = []

    def fake(call, key):
        used.append(key)
        return json.dumps(bodies.pop(0)).encode(), None

    monkeypatch.setattr(pc, "http_get", fake)
    return used


def test_daily_cap_rotates_to_next_key(monkeypatch):
    ok = {"symbol": "CRDO", "quarter": "2027Q1", "transcript": [seg("hi")]}
    used = _replies(monkeypatch, [{"Information": "limit is 25 requests per day"}, ok])
    pool = pc.KeyPool(["k1", "k2"])
    raw, err = pc.fetch(CRDO_Q1, pool, pause_s=0)
    assert err is None and json.loads(raw) == ok
    assert used == ["k1", "k2"] and pool.requests == [1, 1]


def test_all_keys_capped_is_a_gap_not_a_crash(monkeypatch):
    cap = {"Information": "limit is 25 requests per day"}
    _replies(monkeypatch, [cap, cap])
    raw, err = pc.fetch(CRDO_Q1, pc.KeyPool(["k1", "k2"]), pause_s=0)
    assert raw is None and err["type"] == "rate_limited"


def test_empty_transcript_is_a_gap(monkeypatch):
    _replies(monkeypatch, [{"symbol": "CRDO", "quarter": "2027Q1", "transcript": []}])
    raw, err = pc.fetch(CRDO_Q1, pc.KeyPool(["k1"]), pause_s=0)
    assert raw is None and err["type"] == "not_found"


def test_manifest_entry_hashes_the_raw_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "RAW_DIR", tmp_path)
    monkeypatch.setattr(pc, "ROOT", tmp_path)
    raw = json.dumps({"transcript": [seg("first quarter of fiscal 2027"), seg("abc")]}).encode()
    e = pc.manifest_entry(CRDO_Q1, raw, "2026-11-05T00:00:00+00:00")
    assert e["segments"] == 2 and e["chars"] == len("first quarter of fiscal 2027") + 3
    assert e["label_check"] == "match"
    assert len(e["sha256"]) == 64 and e["file"] == "CRDO_2027Q1.json"


def test_corpus_is_24_distinct_calls():
    assert len({(c.ticker, c.label) for c in pc.CORPUS}) == 24


def test_env_file_fills_unset_vars_only(tmp_path, monkeypatch):
    env = tmp_path / ".env.local"
    env.write_text('# comment\n\nexport ALPHAVANTAGE_API_KEYS="k1,k2"\nSET_IN_SHELL=file\nnot a pair\n')
    monkeypatch.delenv("ALPHAVANTAGE_API_KEYS", raising=False)
    monkeypatch.setenv("SET_IN_SHELL", "shell")
    pc.load_env_file(env)
    assert pc.os.environ["ALPHAVANTAGE_API_KEYS"] == "k1,k2"
    assert pc.os.environ["SET_IN_SHELL"] == "shell"


def test_missing_env_file_is_fine(tmp_path):
    pc.load_env_file(tmp_path / "absent")
