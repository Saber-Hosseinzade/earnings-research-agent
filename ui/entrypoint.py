"""Container entrypoint: write .streamlit/secrets.toml from the environment, then run Streamlit.

ECS injects AUTH_CLIENT_SECRET and AUTH_COOKIE_SECRET from Secrets Manager; the other AUTH_* values are
plain environment (infra/ui_stack.py). Streamlit reads sign-in settings only from secrets.toml, so they are
written there (owner-only), and the two secrets are removed from the environment Streamlit runs with.
Without every AUTH_* value, no file is written and the app says sign-in is not configured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

KEYS = {"redirect_uri": "AUTH_REDIRECT_URI", "cookie_secret": "AUTH_COOKIE_SECRET", "client_id": "AUTH_CLIENT_ID",
        "client_secret": "AUTH_CLIENT_SECRET", "server_metadata_url": "AUTH_METADATA_URL"}
SECRET_ENV = ("AUTH_CLIENT_SECRET", "AUTH_COOKIE_SECRET")
PORT = "8501"


def toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def secrets_toml(environ: dict[str, str]) -> str | None:
    values = {key: environ.get(env, "") for key, env in KEYS.items()}
    if not all(values.values()):
        return None
    return "[auth]\n" + "".join(f"{key} = {toml_str(value)}\n" for key, value in values.items())


def streamlit_command() -> list[str]:
    return ["streamlit", "run", "app.py", "--server.port", PORT, "--server.address", "0.0.0.0",
            "--server.headless", "true", "--browser.gatherUsageStats", "false"]


def main() -> None:
    text = secrets_toml(dict(os.environ))
    if text:
        path = Path(__file__).resolve().with_name(".streamlit") / "secrets.toml"
        path.parent.mkdir(exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(text)
    else:
        missing = [env for env in KEYS.values() if not os.environ.get(env)]
        print(f"sign-in not configured, missing: {', '.join(missing)}", file=sys.stderr)
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    os.execvpe("streamlit", streamlit_command(), env)


if __name__ == "__main__":
    main()
