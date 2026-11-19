"""Calls to the tools behind the AgentCore Gateway, over MCP.

The Gateway (infra/gateway_stack.py) exposes two Lambda-backed tools over MCP:

    passages___search_passages    Knowledge Base search (search_lambda/handler.py)
    transcripts___get_transcript  one call from the transcript API (tool_lambda/handler.py)

MCP is the transport, not the control point. The model never sees these tools: it sees
``retrieve_passages`` (retrieval.py), whose code pins the ticker and quarter, enforces the
search limit, expands abbreviations and records every returned passage for the citation check
*before and after* calling ``gateway_retriever``. The fetch is called only by
``approvals.decide()`` after a reviewer approves. Both functions here have the same signatures
as the direct versions (``retrieval.bedrock_retriever``, ``approvals.lambda_fetcher``), so the
rest of the agent does not know which one it runs on.

Inbound auth is AWS IAM: each HTTP request is signed with SigV4 using the caller's credentials
(deployed: the runtime role, which may invoke this one Gateway). Strands' MCPClient has no IAM
signing built in, so ``SigV4Auth`` below is a small ``httpx.Auth``.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import timedelta
from typing import Any, Callable

import httpx

SERVICE = "bedrock-agentcore"
SEARCH_TOOL = "passages___search_passages"
FETCH_TOOL = "transcripts___get_transcript"
CALL_TIMEOUT_S = 40  # above the transcript Lambda's 30 s, so a slow fetch ends as the Lambda's typed error


class GatewayError(Exception):
    """A tool call that failed, in the typed-error shape every tool result uses."""

    def __init__(self, kind: str, message: str, retryable: bool = False):
        super().__init__(f"{kind}: {message}")
        self.kind, self.message, self.retryable = kind, message, retryable

    def as_result(self) -> dict[str, Any]:
        return {"ok": False, "error": {"type": self.kind, "message": self.message[:300], "retryable": self.retryable}}


class SigV4Auth(httpx.Auth):
    """Sign each request to the Gateway with SigV4 (service ``bedrock-agentcore``).

    Credentials are resolved per request, so refreshed role credentials are picked up.
    Only host, content type and body are signed; headers the transport adds later are sent unsigned.
    """

    def __init__(self, region: str, session: Any = None):
        import boto3

        self.region = region
        self.session = session or boto3.Session()

    def auth_flow(self, request: httpx.Request):
        from botocore.auth import SigV4Auth as BotoSigV4
        from botocore.awsrequest import AWSRequest

        headers = {"host": request.url.host}
        if request.headers.get("content-type"):
            headers["content-type"] = request.headers["content-type"]
        aws = AWSRequest(method=request.method, url=str(request.url), data=request.content, headers=headers)
        BotoSigV4(self.session.get_credentials().get_frozen_credentials(), SERVICE, self.region).add_auth(aws)
        for name in ("Authorization", "X-Amz-Date", "X-Amz-Security-Token"):
            if name in aws.headers:
                request.headers[name] = aws.headers[name]
        yield request


def parse_result(result: Any) -> dict[str, Any]:
    """An MCP tool result -> the tool's own JSON (``{"ok": ...}``), or GatewayError.

    The Gateway returns a Lambda target's return value as text content. ``isError`` means the
    call failed before or inside the Lambda (e.g. the function raised), not a typed tool error.
    """
    get = result.get
    texts = [c.get("text", "") for c in (get("content") or []) if isinstance(c, dict) and "text" in c]
    if get("isError") or get("status") == "error":
        raise GatewayError("gateway_error", " ".join(texts) or "the Gateway reported an error", retryable=False)
    structured = get("structuredContent")
    if isinstance(structured, dict) and "ok" in structured:
        return structured
    for text in texts:
        try:
            value = json.loads(text)
        except ValueError:
            continue
        if isinstance(value, str):  # a JSON string holding JSON
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, dict):
            return value
    raise GatewayError("gateway_error", "the tool result held no JSON object", retryable=False)


class GatewayClient:
    """One MCP session per process, opened on first use and reopened after a transport failure."""

    def __init__(self, url: str | None = None, region: str | None = None,
                 client_factory: Callable[[], Any] | None = None):
        self.url = url or os.environ["GATEWAY_URL"]
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "eu-central-1"
        self._factory = client_factory or self._mcp_client
        self._client: Any = None
        self._lock = threading.Lock()

    def _mcp_client(self) -> Any:
        from strands.tools.mcp import MCPClient

        return MCPClient(url=self.url, auth_provider=SigV4Auth(self.region),
                         application_name="earnings-research-agent")

    def _started(self) -> Any:
        with self._lock:
            if self._client is None:
                client = self._factory()
                client.start()
                self._client = client
            return self._client

    def _reset(self) -> None:
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                client.stop(None, None, None)
            except Exception:  # noqa: BLE001 - closing a broken session must not hide the real error
                pass

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one tool; return its JSON. Raises GatewayError for a failure outside the tool."""
        try:
            result = self._started().call_tool_sync(f"gw-{uuid.uuid4().hex[:12]}", tool, arguments,
                                                    read_timeout_seconds=timedelta(seconds=CALL_TIMEOUT_S))
        except GatewayError:
            raise
        except Exception as exc:  # connection, auth, timeout: the session may be broken, open a new one next time
            self._reset()
            text = f"{type(exc).__name__}: {exc}"
            raise GatewayError("gateway_unavailable", text, retryable="Throttl" in text or "imeout" in text) from exc
        # Strands returns a transport failure (timeout, 403, broken stream) as status=error *without*
        # isError, which only a server-reported tool error sets. Treat it like the exception above.
        if result.get("status") == "error" and not result.get("isError"):
            self._reset()
            text = " ".join(c.get("text", "") for c in result.get("content") or [] if isinstance(c, dict))
            raise GatewayError("gateway_unavailable", text or "the MCP call failed",
                               retryable="Throttl" in text or "imeout" in text)
        return parse_result(result)


_shared: GatewayClient | None = None


def shared_client() -> GatewayClient:
    global _shared
    if _shared is None:
        _shared = GatewayClient()
    return _shared


def gateway_retriever(client: GatewayClient | None = None):
    """Retriever (retrieval.Retriever) backed by the Gateway's ``search_passages`` tool."""

    def retrieve(query: str, ticker: str | None, quarter: str | None, k: int) -> list[dict[str, Any]]:
        args: dict[str, Any] = {"query": query, "k": k}
        if ticker:
            args["ticker"] = ticker
        if quarter:
            args["quarter"] = quarter
        out = (client or shared_client()).call(SEARCH_TOOL, args)
        if not out.get("ok"):
            err = out.get("error") or {}
            raise GatewayError(err.get("type", "search_failed"), err.get("message", ""), bool(err.get("retryable")))
        return out["passages"]

    return retrieve


def gateway_fetcher(client: GatewayClient | None = None) -> Callable[[str, str], dict[str, Any]]:
    """Fetcher (approvals) backed by the Gateway's ``get_transcript`` tool. Never raises: errors come back typed."""

    def fetch(ticker: str, quarter: str) -> dict[str, Any]:
        try:
            return (client or shared_client()).call(FETCH_TOOL, {"ticker": ticker, "quarter": quarter})
        except GatewayError as exc:
            return exc.as_result()

    return fetch
