"""What the UI shows and sends, without Streamlit: identity, payloads, and how a runtime result is rendered.

Kept free of Streamlit so it can be tested offline (tests/test_ui.py). app.py only lays it out.

Trust: the runtime takes user_id and role from the payload (agent/serve.py). That is safe only because
the UI server is the one caller allowed to invoke the runtime (its task role, infra), and the UI sets
both from the login on the server, never from anything the browser sends.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

VIEWER, REVIEWER = "viewer", "reviewer"
ROLES = (VIEWER, REVIEWER)

DISCLAIMER = ("Research aid over a fixed set of earnings-call transcripts. Answers can be wrong: check the cited "
              "passages. Not investment advice: it does not say whether to buy, sell or hold anything.")

# Statuses that mean the turn did not produce an answer; the user gets one plain sentence, details stay in the log.
FAILED = {
    "stopped": "The assistant ran out of its per-question budget before it could answer. Try a narrower question.",
    "invalid_output": "The assistant could not produce a valid answer this time. Please try again.",
    "guardrail_error": "The safety check could not run, so the assistant did not answer. Please try again shortly.",
    "bad_request": "That question could not be sent. Please rephrase it.",
    "ui_error": "The assistant could not be reached. Please try again shortly.",
}


class LoginRequired(RuntimeError):
    pass


class SessionExpired(LoginRequired):
    """The ID token behind the login has expired: sign in again (groups are re-read then)."""


class NoAccess(LoginRequired):
    """Signed in, but in neither group."""


@dataclass(frozen=True)
class Identity:
    user_id: str
    role: str
    display: str = ""  # shown in the UI only; the audit log gets user_id


METADATA_SUFFIX = "/.well-known/openid-configuration"


def issuer_from_metadata_url(url: str) -> str:
    return url.removesuffix(METADATA_SUFFIX) if url.endswith(METADATA_SUFFIX) else ""


def login_identity(user: dict[str, Any], issuer: str, now: float | None = None) -> Identity:
    """Identity from st.user: the ID token's claims, verified by Streamlit at sign-in (signature, issuer,
    audience) and kept in a cookie signed with the server's secret, so the browser cannot change them.

    Checked again here: the issuer is our user pool, the token has not expired (the cookie outlives it by
    weeks, and its groups are a snapshot from sign-in), and the user is in a group. reviewer wins over viewer.
    user_id is the Cognito ``sub``, not the email, so the audit log holds no address.
    """
    if not user.get("is_logged_in"):
        raise LoginRequired("not signed in")
    if not issuer or user.get("iss") != issuer:
        raise LoginRequired("signed in with an unexpected identity provider")
    try:
        exp = float(user.get("exp") or 0)
    except (TypeError, ValueError):
        exp = 0
    if exp <= (time.time() if now is None else now):
        raise SessionExpired("your sign-in has expired")
    groups = user.get("cognito:groups") or []
    if isinstance(groups, str):
        groups = [groups]
    role = REVIEWER if REVIEWER in groups else VIEWER if VIEWER in groups else None
    if role is None or not user.get("sub"):
        raise NoAccess("your account has no access yet: an admin must add it to the viewer or reviewer group")
    return Identity(str(user["sub"]), role, str(user.get("email") or ""))


def dev_identity(environ: dict[str, str] | None = None) -> Identity:
    """Local development only: identity from UI_DEV_USER / UI_DEV_ROLE when UI_AUTH=dev.

    Refused inside a container on ECS, so a deployed UI cannot run without a login.
    """
    environ = os.environ if environ is None else environ
    if environ.get("UI_AUTH") != "dev":
        raise LoginRequired("no login configured (set UI_AUTH=dev with UI_DEV_USER and UI_DEV_ROLE for local runs)")
    if environ.get("ECS_CONTAINER_METADATA_URI_V4") or environ.get("ECS_CONTAINER_METADATA_URI"):
        raise LoginRequired("UI_AUTH=dev is for local runs only; refused on ECS")
    user, role = environ.get("UI_DEV_USER", "").strip(), environ.get("UI_DEV_ROLE", "").strip()
    if not user or role not in ROLES:
        raise LoginRequired(f"UI_DEV_USER must be set and UI_DEV_ROLE one of {', '.join(ROLES)}")
    return Identity(user, role)


def new_session_id() -> str:
    """A runtime session id (AgentCore needs at least 33 characters). One per conversation."""
    return f"ui-{uuid.uuid4().hex}"


def chat_payload(question: str, who: Identity) -> dict[str, Any]:
    return {"question": question.strip(), "user_id": who.user_id, "role": who.role}


def action_payload(action: str, who: Identity, **fields: Any) -> dict[str, Any]:
    return {"action": action, "user_id": who.user_id, "role": who.role, **fields}


def markdown_safe(text: str) -> str:
    """Streamlit markdown reads $...$ as LaTeX; earnings answers are full of dollar amounts."""
    return text.replace("$", "\\$")


def quarter_label(quarter: str | None) -> str:
    """'2026Q2' -> 'Q2 2026'. Fiscal quarter as the company labels it."""
    if not quarter or len(quarter) != 6 or quarter[4] != "Q":
        return quarter or ""
    return f"Q{quarter[5]} {quarter[:4]}"


@dataclass
class Citation:
    passage_id: str
    label: str
    text: str


@dataclass
class Turn:
    """One assistant message as shown: the text, short notes under it, and the cited passages."""

    text: str
    status: str
    notes: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    request_id: str | None = None  # an out-of-corpus fetch request this turn created or found
    detail: str | None = None  # why a turn failed (error text), shown collapsed


def _speaker(p: dict[str, Any]) -> str:
    speaker, role = (p.get("speaker") or "").strip(), (p.get("role") or "").strip()
    return f"{speaker} ({role})" if speaker and role and role.lower() != speaker.lower() else speaker or role


def passage_body(text: str) -> str:
    """A Knowledge Base passage starts with a header line for retrieval ("Company | AMD 2026Q2 | Speaker (Role)",
    scripts/build_kb_docs.py), then a blank line. The label shows that already; show what was said."""
    head, sep, body = text.strip().partition("\n\n")
    return body.strip() if sep and "\n" not in head and head.count(" | ") >= 2 else text.strip()


def citation(p: dict[str, Any]) -> Citation:
    parts = [f"{p.get('ticker', '')} {quarter_label(p.get('quarter'))}".strip(), _speaker(p)]
    return Citation(p.get("passage_id", ""), " · ".join(x for x in parts if x), passage_body(p.get("text") or ""))


def render(result: dict[str, Any]) -> Turn:
    """A runtime result (serve.handle: a chat turn or a decision's result) as one message."""
    status = str(result.get("status") or "ui_error")
    answer = result.get("answer") or {}
    if status in FAILED:
        return Turn(FAILED[status], status, detail=str(result.get("error") or "") or None)

    turn = Turn(str(answer.get("answer") or "").strip() or FAILED["invalid_output"], status)
    approval = result.get("approval") or {}
    if approval.get("request_id"):
        turn.request_id = approval["request_id"]
    if status != "ok":
        return turn  # fixed replies (clarify, out of corpus, approval, blocked, capped): the text says it all

    if answer.get("covered"):
        turn.citations = [citation(p) for p in result.get("passages") or []]
        ctx = result.get("context") or {}
        if ctx.get("ticker") and ctx.get("quarter"):
            turn.notes.append(f"From the {ctx['ticker']} {quarter_label(ctx['quarter'])} call.")
    check = result.get("citation_check") or {}
    if answer.get("covered") and check and not check.get("ok"):
        turn.notes.append("Citation check failed: some cited passages were not returned by the search. "
                          "Treat this answer with care.")
    out = ((result.get("guardrail") or {}).get("output") or {})
    if out.get("grounding_flagged"):
        turn.notes.append("Low grounding score: parts of this answer may not be supported by the cited passages.")
    return turn


def render_fetched(result: dict[str, Any], ticker: str | None, quarter: str | None) -> Turn:
    """The answer from an approved out-of-corpus fetch (agent/approvals.py): a quote, not cited passages."""
    if result.get("status") == "fetch_failed" or not result.get("answer"):
        return Turn("The transcript could not be fetched, so there is no answer.", "fetch_failed")
    turn = Turn(str(result["answer"]).strip(), str(result.get("status") or "ok"))
    if result.get("quote") and result.get("quote_in_transcript"):
        label = " · ".join(x for x in (f"{ticker or ''} {quarter_label(quarter)}".strip(),
                                       str(result.get("speaker") or "")) if x)
        turn.citations = [Citation("", label, str(result["quote"]))]
    elif result.get("quote"):
        turn.notes.append("The quoted sentence was not found in the fetched transcript. Treat this answer with care.")
    if result.get("call_date"):
        # Fiscal quarters differ from calendar quarters for some companies: the call date says which call it was.
        day = str(result["call_date"])[:10]  # an ISO timestamp from the transcript API
        turn.notes.append(f"Fetched call held on {day} ({ticker} fiscal {quarter_label(quarter)}).")
    if result.get("truncated"):
        turn.notes.append("The fetched transcript was cut to its first part.")
    return turn


def request_status_text(result: dict[str, Any]) -> tuple[str, Turn | None]:
    """A request_status result: one status line, and the answer once the request was approved and answered."""
    req = result.get("request") if result.get("status") == "ok" else None
    if not req:
        return "That request could not be found.", None
    ticker, quarter = req.get("ticker"), req.get("quarter")
    what = f"{ticker} {quarter_label(quarter)}"
    status = str(req.get("status"))
    if status in ("answered", "failed") and req.get("result"):
        return f"Fetch of {what} was approved.", render_fetched(req["result"], ticker, quarter)
    line = {"pending": f"Fetch of {what} is waiting for a reviewer.",
            "declined": f"Fetch of {what} was declined.",
            "approved": f"Fetch of {what} was approved; the answer is on its way."}.get(status, f"Fetch of {what}: {status}.")
    return line, None
