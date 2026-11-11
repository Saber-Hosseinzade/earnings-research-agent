"""A scripted Strands model provider, so the agent loop can be tested offline."""

from __future__ import annotations

import itertools
import json
from typing import Any, AsyncGenerator

from strands.models.model import Model

TOOL = "get_earnings_call_transcript"

VALID_BRIEF = {
    "ticker": "IBM",
    "quarter": "2024Q4",
    "key_points": ["Revenue grew."],
    "reasons": ["Software demand."],
    "assumptions": ["Stable currency."],
    "speakers_cited": ["CEO"],
    "transcript_truncated": False,
}

_ids = itertools.count()


def _tool_call(name: str, args: dict[str, Any], tokens: int) -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{next(_ids)}", "name": name}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": tokens, "outputTokens": 50, "totalTokens": tokens + 50},
                      "metrics": {"latencyMs": 1}}},
    ]


class FakeModel(Model):
    """mode="good": ``fetches`` data tool calls (default one), then a valid structured answer.
    mode="loop": calls the data tool forever and never answers.
    """

    def __init__(self, mode: str = "good", tokens_per_call: int = 500, fetches: int = 1,
                 tool: str = TOOL, tool_args: dict[str, Any] | list[dict[str, Any]] | None = None) -> None:
        self.mode = mode
        self.tool = tool
        # One args dict for every data call, or a list used in order (last one repeats).
        self.tool_args = tool_args if tool_args is not None else {"ticker": "IBM", "quarter": "2024Q4"}
        self.tokens_per_call = tokens_per_call
        self.fetches = fetches
        self.calls = 0

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {"model_id": f"fake-{self.mode}"}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):  # pragma: no cover
        raise NotImplementedError
        yield  # noqa: unreachable — makes this an async generator

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncGenerator[dict, None]:
        self.calls += 1
        names = [s["name"] for s in (tool_specs or [])]
        output_tool = next((n for n in names if n != self.tool), None)

        if self.mode == "loop" or self.calls <= self.fetches:
            args = self.tool_args
            if isinstance(args, list):
                args = args[min(self.calls, len(args)) - 1]
            events = _tool_call(self.tool, args, self.tokens_per_call)
        else:
            events = _tool_call(output_tool, VALID_BRIEF, self.tokens_per_call)
        for event in events:
            yield event
