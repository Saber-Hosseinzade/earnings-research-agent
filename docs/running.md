# Running it in your own AWS account

The README describes what the assistant does. This page is for reproducing it.

**Prerequisites:** Python 3.12, Node.js 22 (CDK CLI), AWS CLI with a profile for your own account, access to a Bedrock model in your region (the stacks assume an EU inference profile and Amazon Nova 2 Lite), and a key for an earnings-call transcript API (see [Corpus](#corpus)). The dev container in `.devcontainer/` provides the tools.

## Offline tests

No AWS needed. A scripted fake model drives the agent, and CDK assertions check the IAM policies.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

The same suite runs in GitHub Actions on every push and pull request (`.github/workflows/tests.yml`).

## Corpus

The assistant answers over a fixed set of 24 earnings calls listed in `corpus/calls.json`. Transcripts are not in the repo; `corpus/manifest_v2.json` holds the sha256 of each, so a fresh pull can be checked against it.

```bash
cp .env.local.example .env.local          # set TRANSCRIPT_API_KEY and TRANSCRIPT_API_BASE
python scripts/pull_corpus_v2.py          # writes corpus/v2/raw/ (git-ignored)
make corpus-terms                         # words from the calls that are not tickers or companies
```

All provider-specific code is in `scripts/transcript_source.py`. To use another provider or other calls, adapt that file and `corpus/calls.json`, and write your own golden set: [golden_set.md](golden_set.md).

## Deploy

Stacks that stay deployed cost cents a month. The runtime and the UI are deployed for a test or demo and destroyed afterwards.

```bash
scripts/enable_transaction_search.sh                 # once per account: traces; wait for ACTIVE
cd infra
npx cdk bootstrap                                    # once per account and region
npx cdk deploy EarningsResearchAgentTool EarningsResearchAgentKB EarningsResearchAgentGateway \
  EarningsResearchAgentGuardrail EarningsResearchAgentPrompts EarningsResearchAgentAudit EarningsResearchAgentAuth
aws secretsmanager put-secret-value --secret-id <SecretArn> \
  --secret-string '{"api_key": "<your key>", "base_url": "<the API base URL>"}'
cd .. && make kb-sync                                # passages into the Knowledge Base

scripts/build_agent.sh                               # before every runtime deploy
cd infra && npx cdk deploy EarningsResearchAgentRuntime --exclusively
cd .. && scripts/deploy_ui.sh                        # hosted UI; prints its URL
```

Users are created in the Cognito user pool by an administrator (no self sign-up) and added to the `viewer` or `reviewer` group.

**Tear down** the UI first, then the runtime:

```bash
cd infra
npx cdk destroy EarningsResearchAgentUi --exclusively
npx cdk destroy EarningsResearchAgentRuntime --exclusively
```

Hosted UI details:
- **Image:** built by CodeBuild in your account (ARM64); no local Docker.
- **Endpoint:** AWS-managed domain and TLS. The id changes on every create, so the script adds the new URL to Cognito and deploys again. A fresh deploy takes about 14 minutes.
- **Cost:** about $0.065 an hour while up.

## Run locally

```bash
make ui-dev                                          # the chat UI without sign-in, against the deployed runtime
make ui                                              # with Cognito sign-in (once: python scripts/ui_auth_config.py)
```

## Evaluate

```bash
python eval/run_eval.py --target v2.0 --guardrail --label my-run    # golden set, 34 questions
python eval/run_multiturn.py --label my-run                         # 17 scripted conversations
python eval/eval_retrieval.py                                        # retrieval only
```

Scores go to `eval/results/<label>-<date>.json` (committed). Agent outputs go to a git-ignored `.outputs.json` next to them.

## v1

The v1 agent (one structured brief per call, tag `v1.0`) is still in `agent/main.py`, so its baseline stays reproducible:

```bash
cd agent && MODEL_ID=eu.amazon.nova-2-lite-v1:0 TOOL_FUNCTION_NAME=<ToolFunctionName output> python run_local.py IBM 2024Q4
```
