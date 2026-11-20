"""Spend caps, enforced in the runtime: turns per session and turns per day (UTC), across all users.

The run budget (budget.py) caps one turn's model calls and tokens; it does not cap how many turns
are run. These caps do. They live here, not in the UI, so they hold for any caller of the runtime
and survive a UI restart.

Counted: every chat turn and every reviewer decision (an approval fetches and calls the model).
Not counted: listing requests and checking a request's status (no model call, no fetch).
Counted before the turn runs, so a turn that fails still counts; a capped turn runs nothing.

Both counters move in one DynamoDB transaction: either both go up or neither does.
Items expire by TTL (expires_ttl) a day after their period ends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

TURNS_PER_SESSION = 30
TURNS_PER_DAY = 100

SESSION_MESSAGE = ("This conversation has reached its limit of {limit} questions. "
                   "Start a new conversation to continue.")
DAY_MESSAGE = "The assistant has reached its daily limit of {limit} questions. It will answer again after midnight UTC."


@dataclass(frozen=True)
class Limits:
    per_session: int = TURNS_PER_SESSION
    per_day: int = TURNS_PER_DAY

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Limits":
        environ = os.environ if environ is None else environ
        return cls(int(environ.get("TURNS_PER_SESSION") or TURNS_PER_SESSION),
                   int(environ.get("TURNS_PER_DAY") or TURNS_PER_DAY))


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    cap: str | None = None  # "session" | "day" when refused
    limit: int | None = None
    error: str | None = None  # the counter store failed

    @property
    def message(self) -> str | None:
        if self.allowed:
            return None
        if self.error:
            return "The assistant could not check its usage limits, so it did not run. Please try again shortly."
        return (SESSION_MESSAGE if self.cap == "session" else DAY_MESSAGE).format(limit=self.limit)

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "cap": self.cap, "limit": self.limit, "error": self.error}


class Counter(Protocol):
    def take(self, keys: list[tuple[str, int, int]]) -> str | None:
        """Add 1 to every key whose count is below its limit, all or none. keys: (key, limit, expires_ttl).
        Returns None if all went up, else the first key that was at its limit."""


def _keys(session_id: str | None, limits: Limits, now: datetime) -> list[tuple[str, int, int]]:
    day = now.astimezone(timezone.utc).date()
    day_end = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(days=1)
    keys = [(f"day#{day.isoformat()}", limits.per_day, int((day_end + timedelta(days=1)).timestamp()))]
    if session_id:
        # A session can outlive a day; its counter expires with the session memory (7 days, runtime_stack.py).
        keys.append((f"session#{session_id}", limits.per_session, int((now + timedelta(days=8)).timestamp())))
    return keys


def take(counter: Counter, session_id: str | None, limits: Limits | None = None,
         now: datetime | None = None) -> Verdict:
    """Count one turn, or refuse it. A counter failure refuses (fails closed): an uncounted turn costs money."""
    limits = limits or Limits()
    try:
        hit = counter.take(_keys(session_id, limits, now or datetime.now(timezone.utc)))
    except Exception as exc:  # noqa: BLE001
        return Verdict(False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
    if hit is None:
        return Verdict(True)
    if hit.startswith("session#"):
        return Verdict(False, "session", limits.per_session)
    return Verdict(False, "day", limits.per_day)


class InMemoryCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def take(self, keys: list[tuple[str, int, int]]) -> str | None:
        for key, limit, _ in keys:
            if self.counts.get(key, 0) >= limit:
                return key
        for key, _, _ in keys:
            self.counts[key] = self.counts.get(key, 0) + 1
        return None


class DynamoCounter:
    """One item per counter: {counter, n, expires_ttl}. The table's TTL attribute is expires_ttl."""

    def __init__(self, table: str | None = None, client: Any = None, region: str | None = None):
        self.table = table or os.environ["USAGE_TABLE"]
        if client is None:
            import boto3
            client = boto3.client("dynamodb", region_name=region or os.environ.get("AWS_REGION"))
        self.client = client

    def take(self, keys: list[tuple[str, int, int]]) -> str | None:
        items = [{"Update": {
            "TableName": self.table, "Key": {"counter": {"S": key}},
            "UpdateExpression": "ADD n :one SET expires_ttl = if_not_exists(expires_ttl, :ttl)",
            "ConditionExpression": "attribute_not_exists(n) OR n < :limit",
            "ExpressionAttributeValues": {":one": {"N": "1"}, ":limit": {"N": str(limit)}, ":ttl": {"N": str(ttl)}},
        }} for key, limit, ttl in keys]
        try:
            self.client.transact_write_items(TransactItems=items)
        except self.client.exceptions.TransactionCanceledException as exc:
            reasons = exc.response.get("CancellationReasons") or []
            for (key, _, _), reason in zip(keys, reasons):
                if reason.get("Code") == "ConditionalCheckFailed":
                    return key
            raise
        return None
