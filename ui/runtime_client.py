"""Calls the AgentCore Runtime (agent/serve.py) with InvokeAgentRuntime, server-side.

Credentials come from the environment: locally the sandbox profile, deployed the ECS task role, which may
invoke this one runtime and nothing else (infra). Never from the browser.
"""

from __future__ import annotations

import json
import os
from typing import Any

# A turn runs a few model calls and up to two searches; an approval also fetches a transcript.
READ_TIMEOUT_S = 120


class RuntimeClient:
    def __init__(self, runtime_arn: str | None = None, client: Any = None, region: str | None = None) -> None:
        self.runtime_arn = runtime_arn or os.environ["AGENT_RUNTIME_ARN"]
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client("bedrock-agentcore", region_name=region or os.environ.get("AWS_REGION"),
                                  config=Config(read_timeout=READ_TIMEOUT_S, retries={"max_attempts": 1}))
        self.client = client

    def call(self, payload: dict[str, Any], session_id: str) -> dict[str, Any]:
        """One invocation. Never raises: a failure comes back as status "ui_error" with the reason in "error"."""
        try:
            resp = self.client.invoke_agent_runtime(
                agentRuntimeArn=self.runtime_arn, runtimeSessionId=session_id, qualifier="DEFAULT",
                contentType="application/json", accept="application/json",
                payload=json.dumps(payload).encode("utf-8"))
            body = resp["response"].read()
            out = json.loads(body)
        except Exception as exc:  # noqa: BLE001 - shown to the user as one plain sentence (view.FAILED)
            return {"status": "ui_error", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        if not isinstance(out, dict):
            return {"status": "ui_error", "error": f"unexpected response: {str(out)[:200]}"}
        return out
