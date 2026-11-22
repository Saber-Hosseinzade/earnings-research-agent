"""Login for the UI: a Cognito user pool with two groups, viewer and reviewer.

Own stack, kept deployed: no hourly charge (Lite plan), and its lifetime is not the UI's.

    cdk deploy EarningsResearchAgentAuth
    python scripts/ui_auth_config.py      # writes ui/.streamlit/secrets.toml (git-ignored) for local runs

The UI signs in with Streamlit's OIDC login (authorization code flow, PKCE) against the hosted UI,
and reads the role from the ID token's cognito:groups claim, server-side (ui/view.py). Nobody can
sign themselves up: users are created by an admin and added to a group. A user in no group gets no access.

viewer    chats
reviewer  chats, and approves or declines out-of-corpus fetches (agent/approvals.py)
"""

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

GROUPS = {
    "viewer": "Chat with the research assistant",
    "reviewer": "Chat, and approve or decline out-of-corpus transcript fetches",
}
# Streamlit keeps the login in a signed cookie for 30 days, with the groups as they were at sign-in. The UI
# refuses an ID token past its expiry, so a removed user or a changed group takes effect within this time.
ID_TOKEN_VALIDITY = Duration.hours(8)


class EarningsAuthStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, domain_prefix: str, callback_urls: list[str],
                 logout_urls: list[str], ui_hostname: str = "", **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        if ui_hostname:  # the hosted UI's endpoint (ui_stack.py), known after its first deploy
            callback_urls = [*callback_urls, f"https://{ui_hostname}/oauth2callback"]
            logout_urls = [*logout_urls, f"https://{ui_hostname}/"]

        pool = cognito.UserPool(
            self,
            "Users",
            user_pool_name="earnings-research-agent",
            feature_plan=cognito.FeaturePlan.LITE,
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            sign_in_case_sensitive=False,
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            password_policy=cognito.PasswordPolicy(min_length=14, require_symbols=False),
            mfa=cognito.Mfa.OPTIONAL,
            mfa_second_factor=cognito.MfaSecondFactor(otp=True, sms=False),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            # Only demo users live here; recreating them is cheaper than an orphaned pool.
            removal_policy=RemovalPolicy.DESTROY,
        )
        for name, description in GROUPS.items():
            cognito.CfnUserPoolGroup(self, f"Group{name.capitalize()}", user_pool_id=pool.user_pool_id,
                                     group_name=name, description=description)

        domain = pool.add_domain(
            "HostedUi",
            cognito_domain=cognito.CognitoDomainOptions(domain_prefix=domain_prefix),
            managed_login_version=cognito.ManagedLoginVersion.CLASSIC_HOSTED_UI,
        )

        client = pool.add_client(
            "Ui",
            user_pool_client_name="streamlit-ui",
            generate_secret=True,  # confidential client: the secret stays on the UI server
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL, cognito.OAuthScope.PROFILE],
                callback_urls=callback_urls,
                logout_urls=logout_urls,
            ),
            supported_identity_providers=[cognito.UserPoolClientIdentityProvider.COGNITO],
            id_token_validity=ID_TOKEN_VALIDITY,
            access_token_validity=ID_TOKEN_VALIDITY,
            refresh_token_validity=Duration.days(1),
            prevent_user_existence_errors=True,
        )

        # For the hosted UI container (ui_stack.py). The client secret is copied in by scripts/deploy_ui.sh
        # (scripts/ui_auth_config.py --to-secrets-manager), not by CDK: CDK's way reads it with a helper Lambda
        # whose log group would never expire. Until then the secret holds a placeholder and sign-in fails.
        self.client_secret = secretsmanager.Secret(
            self, "UiClientSecret", secret_name="earnings-research-agent/ui-client-secret",
            description="Cognito app client secret for the hosted UI (copied in by scripts/deploy_ui.sh)",
            removal_policy=RemovalPolicy.DESTROY)
        self.cookie_secret = secretsmanager.Secret(
            self, "UiCookieSecret", secret_name="earnings-research-agent/ui-cookie-secret",
            description="Signs the hosted UI's login cookie",
            generate_secret_string=secretsmanager.SecretStringGenerator(password_length=48, exclude_punctuation=True),
            removal_policy=RemovalPolicy.DESTROY)

        issuer = f"https://cognito-idp.{self.region}.amazonaws.com/{pool.user_pool_id}"
        CfnOutput(self, "UserPoolId", value=pool.user_pool_id)
        CfnOutput(self, "ClientId", value=client.user_pool_client_id)
        CfnOutput(self, "MetadataUrl", value=f"{issuer}/.well-known/openid-configuration")
        CfnOutput(self, "HostedUiUrl", value=domain.base_url())
        CfnOutput(self, "ClientSecretName", value=self.client_secret.secret_name)
        self.user_pool, self.client = pool, client
        self.metadata_url = f"{issuer}/.well-known/openid-configuration"
