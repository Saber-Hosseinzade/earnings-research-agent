"""Streamlit chat UI for the earnings research agent.

    cd ui && uv sync --frozen
    AGENT_RUNTIME_ARN=... AWS_PROFILE=sandbox AWS_REGION=eu-central-1 \\
      UI_AUTH=dev UI_DEV_USER=me UI_DEV_ROLE=viewer .venv/bin/streamlit run app.py

Every request goes to the AgentCore Runtime (agent/serve.py) from this server; the browser never holds
AWS credentials. Identity and role are set here, from the login, never from the browser (view.py).
Spend caps, guardrails, the citation check and the audit log all run in the runtime.

The answer arrives whole, not streamed (issue #35).
"""

from __future__ import annotations

import streamlit as st

import view
from runtime_client import RuntimeClient

st.set_page_config(page_title="Earnings call research", layout="centered")


@st.cache_resource
def runtime() -> RuntimeClient:
    return RuntimeClient()


def identity() -> view.Identity:
    try:
        return view.dev_identity()
    except view.LoginRequired as exc:
        st.error(f"Sign-in is not configured: {exc}")
        st.stop()


def show_turn(turn: view.Turn) -> None:
    st.markdown(view.markdown_safe(turn.text))
    for note in turn.notes:
        st.caption(view.markdown_safe(note))
    if turn.citations:
        with st.expander(f"Sources ({len(turn.citations)})"):
            for c in turn.citations:
                st.markdown(f"**{view.markdown_safe(c.label)}**")
                st.markdown("> " + view.markdown_safe(c.text).replace("\n", "\n> "))
    if turn.detail:
        with st.expander("Details"):
            st.code(turn.detail, language=None)


def new_conversation() -> None:
    st.session_state.session_id = view.new_session_id()
    st.session_state.messages = []


who = identity()
if "session_id" not in st.session_state:
    new_conversation()
    st.session_state.my_requests = []

with st.sidebar:
    st.subheader("Earnings call research")
    st.caption(view.DISCLAIMER)
    st.caption(f"Signed in as **{who.user_id}** ({who.role})")
    st.button("New conversation", on_click=new_conversation, width="stretch")
    page = st.radio("View", ["Chat", "Fetch requests"], label_visibility="collapsed") \
        if who.role == view.REVIEWER else "Chat"


def chat_page() -> None:
    st.title("Ask about an earnings call")
    st.caption(view.DISCLAIMER)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                show_turn(m["turn"])
            else:
                st.markdown(view.markdown_safe(m["text"]))

    question = st.chat_input("e.g. Why did AMD's gross margin change in Q2 2026?")
    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(view.markdown_safe(question))
        with st.chat_message("assistant"):
            with st.spinner("Searching the transcripts…"):
                result = runtime().call(view.chat_payload(question, who), st.session_state.session_id)
            turn = view.render(result)
            show_turn(turn)
        st.session_state.messages.append({"role": "assistant", "turn": turn})
        if turn.request_id and turn.request_id not in st.session_state.my_requests:
            st.session_state.my_requests.append(turn.request_id)

    if st.session_state.my_requests:
        with st.expander(f"Your fetch requests ({len(st.session_state.my_requests)})"):
            if st.button("Check status"):
                for rid in st.session_state.my_requests:
                    out = runtime().call(view.action_payload("request_status", who, request_id=rid),
                                         st.session_state.session_id)
                    line, turn = view.request_status_text(out)
                    st.markdown(f"**{rid[:8]}** · {view.markdown_safe(line)}")
                    if turn:
                        show_turn(turn)


def requests_page() -> None:
    st.title("Fetch requests")
    st.caption("A viewer asked about a company or quarter outside the corpus. Approving runs one transcript fetch "
               "from the external source and answers that one question; nothing is added to the corpus.")
    st.button("Refresh")
    out = runtime().call(view.action_payload("list_requests", who), st.session_state.session_id)
    if out.get("status") != "ok":
        st.error(view.render(out).text if out.get("status") in view.FAILED else str(out.get("error")))
        return
    if not out["requests"]:
        st.info("No pending requests.")
    for req in out["requests"]:
        rid = req["request_id"]
        with st.container(border=True):
            st.markdown(f"**{req['ticker']} {view.quarter_label(req['quarter'])}** · asked by {req['requested_by']}")
            st.markdown("> " + view.markdown_safe(str(req.get("question") or "")))
            st.caption(f"Requested {req['created_at']} · expires {req['expires_at']} · {rid[:8]}")
            own = req["requested_by"] == who.user_id
            if own:
                st.caption("You made this request, so another reviewer has to decide it.")
            approve, decline = st.columns(2)
            for col, decision in ((approve, "approve"), (decline, "decline")):
                if col.button(decision.capitalize(), key=f"{decision}-{rid}", disabled=own, width="stretch"):
                    with st.spinner("Fetching and answering…" if decision == "approve" else "Declining…"):
                        res = runtime().call(view.action_payload("decide", who, request_id=rid, decision=decision),
                                             st.session_state.session_id)
                    st.session_state.last_decision = (rid, res)
                    st.rerun()  # list again, without the decided request
    if "last_decision" in st.session_state:
        rid, res = st.session_state.last_decision
        st.divider()
        st.markdown(f"**Last decision** · {rid[:8]} · {view.markdown_safe(str(res.get('message') or res.get('status')))}")
        req = res.get("request") or {}
        if res.get("result"):
            show_turn(view.render_fetched(res["result"], req.get("ticker"), req.get("quarter")))
        elif res.get("status") in view.FAILED:
            show_turn(view.render(res))


if page == "Chat":
    chat_page()
else:
    requests_page()
