"""agent/prompts.py, offline: file and Prompt Management sources, no AWS."""

import pytest

import answer
import fake_model
import prompts
from test_answer import FakeRetriever, model_answering


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    prompts.get.cache_clear()
    monkeypatch.setattr(fake_model, "VALID_BRIEF", fake_model.VALID_BRIEF)
    yield
    prompts.get.cache_clear()


class FakeAgentClient:
    def __init__(self, text="deployed text", fail=None):
        self.text, self.fail, self.calls = text, fail, []

    def get_prompt(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        return {"id": kw["promptIdentifier"], "version": kw["promptVersion"], "defaultVariant": "default",
                "variants": [{"name": "default", "templateType": "TEXT",
                              "templateConfiguration": {"text": {"text": self.text}}}]}


def test_without_ids_the_file_is_used():
    p = prompts.get("answer_system")
    assert p.source == "file" and p.text == (prompts.PROMPTS_DIR / "answer_system.txt").read_text()
    assert p.info() == {"source": "file", "id": None, "version": None, "sha256": p.sha256[:12]}


def test_ids_set_means_bedrock_with_the_pinned_version(monkeypatch):
    monkeypatch.setenv("ANSWER_SYSTEM_PROMPT_ID", "PID123")
    monkeypatch.setenv("ANSWER_SYSTEM_PROMPT_VERSION", "2")
    client = FakeAgentClient()
    real = prompts.from_bedrock
    monkeypatch.setattr(prompts, "from_bedrock", lambda name, pid, ver: real(name, pid, ver, client))
    p = prompts.get("answer_system")
    assert (p.source, p.prompt_id, p.version, p.text) == ("bedrock", "PID123", "2", "deployed text")
    assert client.calls == [{"promptIdentifier": "PID123", "promptVersion": "2"}]
    prompts.get("answer_system")
    assert len(client.calls) == 1  # cached per process


def test_bedrock_failure_raises_rather_than_falling_back(monkeypatch):
    with pytest.raises(RuntimeError, match="throttled"):
        prompts.from_bedrock("answer_system", "PID", "1", FakeAgentClient(fail=RuntimeError("throttled")))


def test_bedrock_without_ids_is_an_error(monkeypatch):
    monkeypatch.setenv("PROMPT_SOURCE", "bedrock")
    with pytest.raises(RuntimeError, match="ANSWER_SYSTEM_PROMPT_ID"):
        prompts.get("answer_system")


def test_prompt_source_file_overrides_ids(monkeypatch):
    monkeypatch.setenv("PROMPT_SOURCE", "file")
    monkeypatch.setenv("ANSWER_SYSTEM_PROMPT_ID", "PID123")
    assert prompts.get("answer_system").source == "file"


def test_empty_template_is_an_error():
    with pytest.raises(RuntimeError, match="no text template"):
        prompts.from_bedrock("answer_system", "PID", "1", FakeAgentClient(text=""))


def test_run_records_which_prompt_it_used():
    out = answer.run({"question": "Why did gross margin expand?", "ticker": "AMD", "quarter": "2026Q2"},
                     model=model_answering(["AMD_2026Q2_t003_p1"]), retriever=FakeRetriever())
    used = out["run"]["prompts"]
    assert list(used) == ["answer_system"]  # answer_final only when the final-answer call ran
    assert used["answer_system"]["source"] == "file"
    assert used["answer_system"]["sha256"] == prompts.get("answer_system").sha256[:12]
