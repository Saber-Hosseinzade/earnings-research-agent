#!/usr/bin/env bash
# Dev container setup. VS Code runs this once, after the container is created
# (postCreateCommand in devcontainer.json).
#
# Stop at the first failure, so a half-built environment can't pass as finished:
#   -e  exit when a command fails
#   -u  treat an unset variable as an error
#   -o pipefail  a failure anywhere in a pipeline fails the whole pipeline
set -euo pipefail

# The virtualenv lives outside the bind-mounted workspace, so a .venv created
# on the host can't collide with it. PATH points here (see devcontainer.json).
VENV=/home/vscode/.venv

# Docker creates the config volumes as root, and ~/.config with them.
sudo chown -R vscode:vscode /home/vscode/.claude /home/vscode/.config/gh
sudo chown vscode:vscode /home/vscode/.config

# Python tooling: agent and CDK dependencies, pytest, and uv.
python -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r requirements-dev.txt

# AgentCore CLI: builds and deploys the agent to AgentCore Runtime. It uses uv
# to package the Python code. Pinned, so a rebuild gets the same version.
npm install -g @aws/agentcore@1.0.1

# The CLI sends anonymous usage analytics by default. Opt out.
agentcore config telemetry.enabled false
