# Earnings Research Agent

A Strands Agents agent, deployed on Amazon Bedrock AgentCore Runtime, that reads a company's earnings call through a Lambda-backed market-data tool and returns the key points, reasons and assumptions as schema-validated JSON.

> **Status:** work in progress.
>
> **Not investment advice.** This is a research aid. It produces no buy/sell recommendations, and LLM-generated financial analysis can be wrong.

## Architecture

*Diagram to come — `docs/architecture.png`.*

```
caller ──> AgentCore Runtime (Strands agent, Bedrock model)      agent/
                │   run budget hook · structured output
                └──> Lambda tool ──> Alpha Vantage API           tool_lambda/
                         │
                         └── API key from Secrets Manager        infra/ (CDK)
```

## Safeguards

| Safeguard | Where | Why |
|---|---|---|
| HTTP timeout derived from the Lambda's remaining time, minus a margin | `tool_lambda/handler.py` | A slow upstream returns a typed `timeout` error to the agent instead of the Lambda timing out |
| Upstream errors mapped to typed, retryable-or-not results | `tool_lambda/handler.py` | Alpha Vantage reports rate limits with HTTP 200 and a message; those must not reach the model as data |
| Transcript truncated to a character budget | `tool_lambda/handler.py` | Bounds the tokens a single tool call can add to context |
| Soft stop: tool calls past the limit are cancelled with an instruction to answer | `agent/budget.py` | Lets the model finish with what it has |
| Hard stop: model calls and tokens; further model calls are cancelled | `agent/budget.py` | A ceiling that doesn't depend on the model obeying a message |
| Output validated against a Pydantic schema | `agent/schema.py` | A malformed answer fails loudly instead of parsing into something plausible |
| Least-privilege invoke policy for the agent | `infra/stack.py` | The runtime can invoke this one function and nothing else |
| No API keys in code, template or repo | Secrets Manager | — |

The budget is tested end to end with a scripted model that calls the tool forever (`tests/test_agent_budget.py`).

## Run it

**Prerequisites:** Python 3.10+, Node.js (for the CDK CLI), AWS CLI with a profile for your own account, an [Alpha Vantage](https://www.alphavantage.co/) API key, and access to a Bedrock model in your chosen region.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                 # offline, no AWS needed
```

**1. Deploy the tool**

```bash
cd infra
npx cdk bootstrap                         # once per account and region
npx cdk deploy
aws secretsmanager put-secret-value --secret-id <SecretArn> --secret-string '<your Alpha Vantage key>'
```

**2. Run the agent locally** against the deployed tool and Bedrock:

```bash
cd agent
export MODEL_ID=<Bedrock model or inference-profile id>
export TOOL_FUNCTION_NAME=<ToolFunctionName output>
python run_local.py IBM 2024Q4
```

**3. Deploy to AgentCore Runtime** — *to come.*

**4. Tear down** — *to come.*

## Measurements

*Tokens, rounds and latency per run — to come.*

## License

MIT
