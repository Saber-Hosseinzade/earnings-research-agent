"""Run the agent once from the command line, no server.

    cd agent
    MODEL_ID=... TOOL_FUNCTION_NAME=... python run_local.py IBM 2024Q4
"""

import json
import sys

from main import run

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python run_local.py TICKER YYYYQn")
    print(json.dumps(run({"ticker": sys.argv[1], "quarter": sys.argv[2]}), indent=2))
