"""Load the agent's prompts: from Bedrock Prompt Management (deployed) or from agent/prompts/ (offline).

The files in agent/prompts/ are the source of truth: they are reviewed in git, and
infra/prompt_stack.py publishes each one as a Prompt Management prompt with a pinned,
numbered version. The runtime loads that pinned version, so every run can say which
prompt version produced it.

Where a prompt comes from (per process, cached):
- PROMPT_SOURCE=bedrock, or unset with <NAME>_PROMPT_ID set -> GetPrompt(id, version).
  A failure raises: the runtime never falls back to a file silently, which would hide
  a difference between what is deployed and what is in git.
- PROMPT_SOURCE=file, or unset without an ID -> agent/prompts/<name>.txt (tests, local runs,
  evaluation of earlier baselines).

Env per prompt, e.g. for "answer_system": ANSWER_SYSTEM_PROMPT_ID, ANSWER_SYSTEM_PROMPT_VERSION.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    source: str  # "bedrock" | "file"
    prompt_id: str | None = None
    version: str | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def info(self) -> dict[str, Any]:
        """What a run record says about the prompt it used (no text)."""
        return {"source": self.source, "id": self.prompt_id, "version": self.version, "sha256": self.sha256[:12]}


def _env(name: str, suffix: str) -> str | None:
    return os.environ.get(f"{name.upper()}_PROMPT_{suffix}")


def from_file(name: str) -> Prompt:
    return Prompt(name, (PROMPTS_DIR / f"{name}.txt").read_text(), "file")


def from_bedrock(name: str, prompt_id: str, version: str, client: Any = None) -> Prompt:
    if client is None:
        import boto3

        client = boto3.client("bedrock-agent")
    resp = client.get_prompt(promptIdentifier=prompt_id, promptVersion=version)
    variants = resp.get("variants") or []
    default = resp.get("defaultVariant")
    variant = next((v for v in variants if v.get("name") == default), variants[0] if variants else None)
    text = ((variant or {}).get("templateConfiguration") or {}).get("text", {}).get("text")
    if not text:
        raise RuntimeError(f"prompt {prompt_id} version {version} has no text template")
    return Prompt(name, text, "bedrock", prompt_id, str(resp.get("version", version)))


@lru_cache(maxsize=None)
def get(name: str) -> Prompt:
    source = os.environ.get("PROMPT_SOURCE") or ("bedrock" if _env(name, "ID") else "file")
    if source == "file":
        return from_file(name)
    if source != "bedrock":
        raise ValueError(f"PROMPT_SOURCE must be 'bedrock' or 'file', not {source!r}")
    prompt_id, version = _env(name, "ID"), _env(name, "VERSION")
    if not prompt_id or not version:
        raise RuntimeError(f"PROMPT_SOURCE=bedrock needs {name.upper()}_PROMPT_ID and {name.upper()}_PROMPT_VERSION")
    return from_bedrock(name, prompt_id, version)
