"""Write ui/.streamlit/secrets.toml for local runs with Cognito sign-in (infra/auth_stack.py).

    AWS_PROFILE=sandbox python scripts/ui_auth_config.py

Reads the auth stack's outputs and the app client's secret, and writes the [auth] section Streamlit's
OIDC sign-in needs. The file is git-ignored and written owner-only; the client secret is never printed.
A cookie_secret already in the file is kept, so existing sign-ins stay valid.
"""

from __future__ import annotations

import os
import secrets
import sys
import tomllib
from pathlib import Path

STACK = "EarningsResearchAgentAuth"
OUT = Path(__file__).resolve().parents[1] / "ui" / ".streamlit" / "secrets.toml"
# Streamlit fills in the port it runs on, so one file serves 8501 and 8502 (both are callback URLs in cdk.json).
REDIRECT_URI = "http://localhost:{port}/oauth2callback"


def toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(client_id: str, client_secret: str, metadata_url: str, cookie_secret: str,
           redirect_uri: str = REDIRECT_URI) -> str:
    return "\n".join([
        "# Written by scripts/ui_auth_config.py. Git-ignored: holds the Cognito app client secret.",
        "[auth]",
        f"redirect_uri = {toml_str(redirect_uri)}",
        f"cookie_secret = {toml_str(cookie_secret)}",
        f"client_id = {toml_str(client_id)}",
        f"client_secret = {toml_str(client_secret)}",
        f"server_metadata_url = {toml_str(metadata_url)}",
        "",
    ])


def existing_cookie_secret(path: Path) -> str | None:
    try:
        return tomllib.loads(path.read_text()).get("auth", {}).get("cookie_secret")
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return None


def main() -> int:
    import boto3

    region = os.environ.get("AWS_REGION", "eu-central-1")
    cfn = boto3.client("cloudformation", region_name=region)
    outputs = {o["OutputKey"]: o["OutputValue"]
               for o in cfn.describe_stacks(StackName=STACK)["Stacks"][0].get("Outputs", [])}
    client = boto3.client("cognito-idp", region_name=region).describe_user_pool_client(
        UserPoolId=outputs["UserPoolId"], ClientId=outputs["ClientId"])["UserPoolClient"]
    cookie_secret = existing_cookie_secret(OUT) or secrets.token_urlsafe(32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(render(outputs["ClientId"], client["ClientSecret"], outputs["MetadataUrl"], cookie_secret))
    os.chmod(OUT, 0o600)
    print(f"wrote {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT} "
          f"(client {outputs['ClientId'][:6]}…, secret not shown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
