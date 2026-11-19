import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "agent"), str(ROOT / "tool_lambda")]

# Clients are created at import time; give them a region and no real creds.
# boto3 looks up a named profile even with keys set, and fails if it is missing.
os.environ.pop("AWS_PROFILE", None)
# Without GATEWAY_URL the agent picks the direct search and fetch; tests inject fakes either way.
os.environ.pop("GATEWAY_URL", None)
# Prompts come from agent/prompts/ in tests, never from a deployed Prompt Management version.
for _k in [k for k in os.environ if k == "PROMPT_SOURCE" or "_PROMPT_" in k]:
    os.environ.pop(_k)
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("TOOL_FUNCTION_NAME", "fake-tool")
