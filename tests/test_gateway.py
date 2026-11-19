"""MCP Gateway path, offline: the search Lambda, result parsing, the client, and the agent's choice of path.

No network: the MCP client is replaced by a fake with the same call_tool_sync/start/stop surface,
returning results in the shape Strands' MCPClient returns them.
"""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

import approvals
import gateway
import retrieval
from gateway import FETCH_TOOL, SEARCH_TOOL, GatewayClient, GatewayError, SigV4Auth, parse_result
from retrieval import Ledger, make_retrieve_tool

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("search_handler", ROOT / "search_lambda" / "handler.py")
search_handler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search_handler)

KB_RESPONSE = {"retrievalResults": [
    {"content": {"text": "Data center revenue grew."}, "score": 0.81234567,
     "metadata": {"passage_id": "AMD_2026Q2_t003_p1", "ticker": "AMD", "quarter": "2026Q2",
                  "speaker": "Lisa Su", "role": "CEO", "start_seconds": 312.0}},
    {"content": {"text": "Gross margin was 54%."}, "score": 0.7,
     "metadata": {"passage_id": "AMD_2026Q2_t007_p1", "ticker": "AMD", "quarter": "2026Q2",
                  "speaker": "Jean Hu", "role": "CFO"}},
]}


class FakeKB:
    def __init__(self, response=KB_RESPONSE, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.response


def text_result(value, **extra):
    return {"status": "success", "toolUseId": "x", "content": [{"text": json.dumps(value)}], **extra}


class FakeMCP:
    """Stands in for strands MCPClient: start/stop/call_tool_sync."""

    def __init__(self, results):
        self.results = list(results)
        self.calls, self.started, self.stopped = [], 0, 0

    def start(self):
        self.started += 1
        return self

    def stop(self, *args):
        self.stopped += 1

    def call_tool_sync(self, tool_use_id, name, arguments=None, read_timeout_seconds=None):
        self.calls.append((name, arguments))
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client_with(*results):
    fake = FakeMCP(results)
    return GatewayClient(url="https://gw.example/mcp", region="eu-central-1", client_factory=lambda: fake), fake


# --- search Lambda -------------------------------------------------------------------------------

def test_search_lambda_filters_on_ticker_and_quarter(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_ID", "KB1")
    kb = FakeKB()
    out = search_handler.lambda_handler({"query": "margins", "ticker": "amd", "quarter": "2026q2", "k": 3}, client=kb)
    assert out["ok"] and [p["passage_id"] for p in out["passages"]] == ["AMD_2026Q2_t003_p1", "AMD_2026Q2_t007_p1"]
    cfg = kb.calls[0]["retrievalConfiguration"]["vectorSearchConfiguration"]
    assert cfg["numberOfResults"] == 3
    assert cfg["filter"] == {"andAll": [{"equals": {"key": "ticker", "value": "AMD"}},
                                        {"equals": {"key": "quarter", "value": "2026Q2"}}]}
    assert kb.calls[0]["knowledgeBaseId"] == "KB1"


@pytest.mark.parametrize("event, kind", [
    ({}, "bad_request"),
    ({"query": "x", "ticker": "not a ticker"}, "bad_request"),
    ({"query": "x", "quarter": "Q2"}, "bad_request"),
    ({"query": "x", "k": "many"}, "bad_request"),
    ({"query": "x" * 1001}, "bad_request"),
])
def test_search_lambda_rejects_bad_input_before_searching(monkeypatch, event, kind):
    monkeypatch.setenv("KNOWLEDGE_BASE_ID", "KB1")
    kb = FakeKB()
    out = search_handler.lambda_handler(event, client=kb)
    assert out["ok"] is False and out["error"]["type"] == kind and not kb.calls


def test_search_lambda_clamps_k(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_ID", "KB1")
    kb = FakeKB()
    search_handler.lambda_handler({"query": "x", "k": 50}, client=kb)
    assert kb.calls[0]["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"] == 8


def test_search_lambda_returns_throttling_as_a_retryable_typed_error(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_ID", "KB1")

    class ThrottlingException(Exception):
        response = {"Error": {"Code": "ThrottlingException"}}

    out = search_handler.lambda_handler({"query": "x"}, client=FakeKB(exc=ThrottlingException("slow down")))
    assert out == {"ok": False, "error": {"type": "rate_limited", "message": out["error"]["message"], "retryable": True}}


def test_search_lambda_and_direct_retriever_map_results_the_same():
    """The direct path (local runs) and the Gateway path must hand the agent identical passages."""
    import boto3

    real_client = boto3.client
    try:
        boto3.client = lambda *a, **k: FakeKB()
        via_direct = retrieval.bedrock_retriever(kb_id="KB1")("margins", "AMD", "2026Q2", 5)
    finally:
        boto3.client = real_client
    assert via_direct == search_handler.to_passages(KB_RESPONSE)


# --- parsing MCP results -------------------------------------------------------------------------

def test_parse_result_reads_the_tools_json_from_text_content():
    assert parse_result(text_result({"ok": True, "passages": []})) == {"ok": True, "passages": []}


def test_parse_result_prefers_structured_content():
    r = text_result({"ignored": 1}, structuredContent={"ok": True, "passages": [1]})
    assert parse_result(r) == {"ok": True, "passages": [1]}


def test_parse_result_unwraps_json_inside_a_json_string():
    r = {"status": "success", "content": [{"text": json.dumps(json.dumps({"ok": True}))}]}
    assert parse_result(r) == {"ok": True}


def test_parse_result_raises_on_a_server_reported_error():
    with pytest.raises(GatewayError) as e:
        parse_result({"status": "error", "isError": True, "content": [{"text": "Lambda raised"}]})
    assert e.value.kind == "gateway_error"


def test_parse_result_raises_when_no_json_came_back():
    with pytest.raises(GatewayError):
        parse_result({"status": "success", "content": [{"text": "not json"}]})


# --- client --------------------------------------------------------------------------------------

def test_client_opens_one_session_and_reuses_it():
    client, fake = client_with(text_result({"ok": True, "passages": []}), text_result({"ok": True, "passages": []}))
    client.call(SEARCH_TOOL, {"query": "a"})
    client.call(SEARCH_TOOL, {"query": "b"})
    assert fake.started == 1 and len(fake.calls) == 2


def test_a_transport_failure_reported_as_a_result_resets_the_session():
    client, fake = client_with({"status": "error", "content": [{"text": "Tool execution failed: 403 Forbidden"}]},
                               text_result({"ok": True, "passages": []}))
    with pytest.raises(GatewayError) as e:
        client.call(SEARCH_TOOL, {"query": "a"})
    assert e.value.kind == "gateway_unavailable" and fake.stopped == 1
    client.call(SEARCH_TOOL, {"query": "a"})
    assert fake.started == 2


def test_an_exception_resets_the_session_and_timeouts_are_retryable():
    client, fake = client_with(TimeoutError("read timeout"))
    with pytest.raises(GatewayError) as e:
        client.call(SEARCH_TOOL, {"query": "a"})
    assert e.value.kind == "gateway_unavailable" and e.value.retryable and fake.stopped == 1


# --- retriever and fetcher over the Gateway ------------------------------------------------------

def test_gateway_retriever_sends_only_the_filters_it_has():
    client, fake = client_with(text_result({"ok": True, "passages": search_handler.to_passages(KB_RESPONSE)}))
    passages = gateway.gateway_retriever(client)("margins", "AMD", None, 5)
    assert fake.calls == [(SEARCH_TOOL, {"query": "margins", "k": 5, "ticker": "AMD"})]
    assert passages[0]["passage_id"] == "AMD_2026Q2_t003_p1"


def test_retrieve_tool_over_the_gateway_keeps_pinning_ledger_and_search_limit():
    """The safeguards run in the agent: pinned values win, IDs are recorded, a third search never reaches MCP."""
    ok = text_result({"ok": True, "passages": search_handler.to_passages(KB_RESPONSE)})
    client, fake = client_with(ok, ok)
    ledger = Ledger()
    tool = make_retrieve_tool(gateway.gateway_retriever(client), ledger, "AMD", "2026Q2")
    json.loads(tool._tool_func(query="capex", ticker="META", quarter="2025Q4"))
    json.loads(tool._tool_func(query="margins"))
    third = json.loads(tool._tool_func(query="guidance"))
    assert [c[1]["ticker"] for c in fake.calls] == ["AMD", "AMD"]  # the model's META never left the agent
    assert fake.calls[0][1]["query"] == "capex (capital expenditures)"  # expanded in the agent
    assert "AMD_2026Q2_t003_p1" in ledger.returned
    assert third["error"]["type"] == "search_limit" and len(fake.calls) == 2


def test_a_typed_search_error_keeps_its_retryable_flag_in_the_tool_result():
    client, _ = client_with(text_result({"ok": False, "error": {"type": "rate_limited", "message": "slow",
                                                                  "retryable": True}}))
    out = json.loads(make_retrieve_tool(gateway.gateway_retriever(client), Ledger())._tool_func(query="x"))
    assert out["error"] == {"type": "retrieve_failed", "message": out["error"]["message"], "retryable": True}


def test_gateway_fetcher_returns_the_lambdas_result_and_never_raises():
    fetched = {"ok": True, "ticker": "NVDA", "quarter": "2026Q3", "transcript": "x"}
    client, fake = client_with(text_result(fetched), TimeoutError("read timeout"))
    fetch = gateway.gateway_fetcher(client)
    assert fetch("NVDA", "2026Q3") == fetched
    assert fake.calls == [(FETCH_TOOL, {"ticker": "NVDA", "quarter": "2026Q3"})]
    failed = fetch("NVDA", "2026Q3")
    assert failed["ok"] is False and failed["error"]["type"] == "gateway_unavailable"


# --- which path the agent takes ------------------------------------------------------------------

def test_with_gateway_url_set_search_and_fetch_go_through_the_gateway(monkeypatch):
    monkeypatch.setenv("GATEWAY_URL", "https://gw.example/mcp")
    monkeypatch.setattr(gateway, "_shared", None)
    assert retrieval.default_retriever().__qualname__.startswith("gateway_retriever")
    assert approvals.default_fetcher().__qualname__.startswith("gateway_fetcher")


def test_without_gateway_url_local_runs_use_the_direct_path(monkeypatch):
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    monkeypatch.setenv("KNOWLEDGE_BASE_ID", "KB1")
    assert retrieval.default_retriever().__qualname__.startswith("bedrock_retriever")
    assert approvals.default_fetcher().__qualname__.startswith("lambda_fetcher")


# --- SigV4 ---------------------------------------------------------------------------------------

def test_sigv4_auth_signs_for_bedrock_agentcore():
    import boto3

    session = boto3.Session(aws_access_key_id="AKIDEXAMPLE", aws_secret_access_key="secret",
                            aws_session_token="token", region_name="eu-central-1")
    request = httpx.Request("POST", "https://gw.example/mcp", json={"jsonrpc": "2.0", "method": "tools/list"})
    signed = next(SigV4Auth("eu-central-1", session).auth_flow(request))
    auth = signed.headers["Authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert "/eu-central-1/bedrock-agentcore/aws4_request" in auth
    assert signed.headers["X-Amz-Security-Token"] == "token" and "X-Amz-Date" in signed.headers
