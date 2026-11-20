"""UI sign-in: role from the ID token (ui/view.py), the Cognito stack, and the local secrets file. Offline."""

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ui"), str(ROOT / "infra")]

import view  # noqa: E402
from auth_stack import EarningsAuthStack  # noqa: E402

_spec = importlib.util.spec_from_file_location("ui_auth_config", ROOT / "scripts" / "ui_auth_config.py")
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

ISSUER = "https://cognito-idp.eu-central-1.amazonaws.com/eu-central-1_abc"
NOW = 1_800_000_000
USER = {"is_logged_in": True, "iss": ISSUER, "sub": "uuid-1", "email": "a@example.com", "exp": NOW + 60,
        "cognito:groups": ["viewer"]}


# --- role from the ID token ----------------------------------------------------------------------------------------

def test_viewer_and_reviewer_from_groups_reviewer_wins():
    assert view.login_identity(USER, ISSUER, NOW) == view.Identity("uuid-1", "viewer", "a@example.com")
    both = {**USER, "cognito:groups": ["viewer", "reviewer"]}
    assert view.login_identity(both, ISSUER, NOW).role == "reviewer"


def test_user_id_is_the_sub_not_the_email():
    who = view.login_identity(USER, ISSUER, NOW)
    assert who.user_id == "uuid-1" and view.chat_payload("q", who)["user_id"] == "uuid-1"


@pytest.mark.parametrize("groups", [[], None, ["admin"], "Viewer"])
def test_no_group_no_access(groups):
    with pytest.raises(view.NoAccess):
        view.login_identity({**USER, "cognito:groups": groups}, ISSUER, NOW)


def test_expired_token_means_sign_in_again():
    with pytest.raises(view.SessionExpired):
        view.login_identity({**USER, "exp": NOW - 1}, ISSUER, NOW)
    with pytest.raises(view.SessionExpired):
        view.login_identity({k: v for k, v in USER.items() if k != "exp"}, ISSUER, NOW)


@pytest.mark.parametrize("user", [{**USER, "is_logged_in": False}, {**USER, "iss": "https://evil.example"}])
def test_not_signed_in_or_other_issuer_refused(user):
    with pytest.raises(view.LoginRequired):
        view.login_identity(user, ISSUER, NOW)


def test_no_issuer_configured_refuses():
    with pytest.raises(view.LoginRequired):
        view.login_identity(USER, "", NOW)


def test_issuer_from_metadata_url():
    assert view.issuer_from_metadata_url(ISSUER + "/.well-known/openid-configuration") == ISSUER
    assert view.issuer_from_metadata_url("https://example.com/other") == ""


# --- Cognito stack -------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def template():
    app = cdk.App()
    return Template.from_stack(EarningsAuthStack(
        app, "Auth", domain_prefix="earnings-research-test", callback_urls=["http://localhost:8501/oauth2callback"],
        logout_urls=["http://localhost:8501/"]))


def test_nobody_signs_themselves_up(template):
    (pool,) = template.find_resources("AWS::Cognito::UserPool").values()
    assert pool["Properties"]["AdminCreateUserConfig"]["AllowAdminCreateUserOnly"] is True
    assert pool["Properties"]["UserPoolTier"] == "LITE"


def test_two_groups(template):
    names = sorted(g["Properties"]["GroupName"] for g in template.find_resources("AWS::Cognito::UserPoolGroup").values())
    assert names == ["reviewer", "viewer"]


def test_client_is_confidential_code_flow_with_short_id_tokens(template):
    (client,) = template.find_resources("AWS::Cognito::UserPoolClient").values()
    p = client["Properties"]
    assert p["GenerateSecret"] is True
    assert p["AllowedOAuthFlows"] == ["code"] and p["AllowedOAuthFlowsUserPoolClient"] is True
    assert sorted(p["AllowedOAuthScopes"]) == ["email", "openid", "profile"]
    assert p["CallbackURLs"] == ["http://localhost:8501/oauth2callback"]
    assert p["IdTokenValidity"] == 480 and p["TokenValidityUnits"]["IdToken"] == "minutes"
    assert p["PreventUserExistenceErrors"] == "ENABLED"


def test_outputs_carry_no_secret(template):
    outputs = json.dumps(template.find_outputs("*"))
    assert "ClientSecret" not in outputs


# --- local secrets file --------------------------------------------------------------------------------------------

def test_secrets_file_is_valid_toml_for_streamlit():
    text = cfg.render("cid", 'se"cret', ISSUER + "/.well-known/openid-configuration", "cookie")
    auth = tomllib.loads(text)["auth"]
    assert auth == {"redirect_uri": "http://localhost:{port}/oauth2callback", "cookie_secret": "cookie",
                    "client_id": "cid", "client_secret": 'se"cret',
                    "server_metadata_url": ISSUER + "/.well-known/openid-configuration"}


def test_cookie_secret_is_kept(tmp_path):
    f = tmp_path / "secrets.toml"
    f.write_text(cfg.render("c", "s", "m", "keep-me"))
    assert cfg.existing_cookie_secret(f) == "keep-me"
    assert cfg.existing_cookie_secret(tmp_path / "missing.toml") is None


def test_secrets_file_is_git_ignored():
    assert "ui/.streamlit/secrets.toml" in (ROOT / ".gitignore").read_text().splitlines()
