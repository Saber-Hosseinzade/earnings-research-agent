"""Run budget: the agent counts its own rounds, tool calls and tokens, and stops.

Two levels, on purpose:

* **Soft stop — tool calls.** Past ``max_tool_calls`` each further tool call is
  cancelled with a message telling the model to answer with what it has. The
  model still gets to write its answer.
* **Hard stop — model calls and tokens.** Past ``max_model_calls``, or once the
  tokens used plus the projected input of the next call would exceed
  ``max_total_tokens``, every further model call is cancelled. The loop ends
  and ``stopped_reason`` says why. This is the ceiling that cannot be argued
  with: it does not depend on the model obeying a message.

Counts reset at the start of each invocation, so the budget is per request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from strands.hooks import (
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)


@dataclass
class RunBudget(HookProvider):
    max_model_calls: int = 6
    max_tool_calls: int = 3
    max_total_tokens: int = 40_000

    model_calls: int = field(default=0, init=False)
    tool_calls: int = field(default=0, init=False)
    tokens_used: int = field(default=0, init=False)
    stopped_reason: str | None = field(default=None, init=False)
    tools_cancelled: int = field(default=0, init=False)
    _tokens_at_start: int = field(default=0, init=False)

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self._reset)
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(BeforeToolCallEvent, self._before_tool)

    # accumulated_usage spans every request this Agent has served, so the
    # per-request figure is measured against a snapshot taken at the start.
    @staticmethod
    def _total_tokens(agent: Any) -> int:
        return int(agent.event_loop_metrics.accumulated_usage.get("totalTokens", 0))

    def _reset(self, event: BeforeInvocationEvent) -> None:
        self.model_calls = 0
        self.tool_calls = 0
        self.tokens_used = 0
        self.tools_cancelled = 0
        self.stopped_reason = None
        self._tokens_at_start = self._total_tokens(event.agent)

    def _before_model(self, event: BeforeModelCallEvent) -> None:
        self.tokens_used = self._total_tokens(event.agent) - self._tokens_at_start
        if self.stopped_reason is None:
            projected = event.projected_input_tokens or 0
            if self.model_calls >= self.max_model_calls:
                self.stopped_reason = "max_model_calls"
            elif self.tokens_used + projected > self.max_total_tokens:
                self.stopped_reason = "max_total_tokens"
        if self.stopped_reason is not None:
            event.cancel = f"STOPPED by run budget: {self.stopped_reason}"
            return
        self.model_calls += 1

    def _before_tool(self, event: BeforeToolCallEvent) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            self.tools_cancelled += 1
            event.cancel_tool = (
                f"Tool budget reached ({self.max_tool_calls} calls). "
                "Do not call tools again. Answer now with the information you already have."
            )

    def summary(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "tools_cancelled": self.tools_cancelled,
            "tokens_used": self.tokens_used,
            "stopped_reason": self.stopped_reason,
            "limits": {
                "max_model_calls": self.max_model_calls,
                "max_tool_calls": self.max_tool_calls,
                "max_total_tokens": self.max_total_tokens,
            },
        }
