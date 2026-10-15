"""Agent-side wrapper around the Lambda tool.

The Lambda owns validation, timeouts and upstream error mapping. This wrapper
only invokes it and turns an invocation failure into the same typed-error shape,
so the model always sees one result format.
"""

from __future__ import annotations

import json
import os

import boto3
from botocore.config import Config
from strands import tool

# The Lambda's own HTTP timeout is below its configured timeout, so this read
# timeout only fires if the invocation itself hangs.
_lambda = boto3.client(
    "lambda",
    config=Config(read_timeout=30, connect_timeout=5, retries={"max_attempts": 2, "mode": "standard"}),
)


@tool
def get_earnings_call_transcript(ticker: str, quarter: str) -> str:
    """Fetch an excerpt of a company's earnings-call transcript.

    Returns JSON. On success: ok=true, the transcript text as "Speaker (Title): content"
    lines, and truncated=true if it was cut to fit a length budget. On failure: ok=false
    with error.type one of bad_request, not_found, rate_limited, timeout, upstream_error,
    and error.retryable. Do not retry an error that is not retryable.

    Args:
        ticker: Stock ticker symbol, e.g. "IBM".
        quarter: Fiscal quarter in the form YYYYQn, e.g. "2024Q4".
    """
    try:
        resp = _lambda.invoke(
            FunctionName=os.environ["TOOL_FUNCTION_NAME"],
            InvocationType="RequestResponse",
            Payload=json.dumps({"ticker": ticker, "quarter": quarter}).encode(),
        )
        payload = resp["Payload"].read().decode()
        if resp.get("FunctionError"):
            return json.dumps(
                {"ok": False, "error": {"type": "tool_crashed", "message": payload[:300], "retryable": False}}
            )
        return payload
    except Exception as exc:  # network, throttling, permissions
        return json.dumps(
            {"ok": False, "error": {"type": "invoke_failed", "message": str(exc)[:300], "retryable": False}}
        )
