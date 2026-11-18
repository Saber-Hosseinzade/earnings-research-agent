#!/usr/bin/env bash
# Builds the code bundle for AgentCore Runtime into build/agent. CDK zips that
# folder and uploads it, so run this before `cdk deploy`.
#
# The runtime is Linux on arm64 with Python 3.12, which may not match this
# machine. So we ask uv for wheels built for that target, and --only-binary
# makes the build fail rather than compile something for the wrong platform.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build/agent"

# Start clean, so files removed from the agent can't linger in the bundle.
rm -rf "$OUT"
mkdir -p "$OUT"

uv pip install \
  --quiet \
  --target "$OUT" \
  --python-version 3.12 \
  --python-platform aarch64-manylinux2014 \
  --only-binary :all: \
  --requirement "$ROOT/agent/requirements.txt"

# Only the modules the runtime imports; run_local.py, ask.py and the eval code are for local use.
# serve.py (v2) is the deployed entrypoint; main.py and its modules (v1) are kept so it can be redeployed.
cp "$ROOT"/agent/{serve,answer,audit,budget,citations,guard,prompts,retrieval,session,main,schema,tools}.py "$OUT"/
# session.py resolves follow-ups against the corpus call list and company names.
cp "$ROOT"/corpus/calls.json "$OUT"/calls.json
# prompts.py reads these files only when PROMPT_SOURCE=file; deployed, it loads the pinned
# Bedrock versions. Shipped so the sha256 in run.prompts can be compared with the repo.
cp -r "$ROOT"/agent/prompts "$OUT"/prompts

du -sh "$OUT"
