"""Account sign-in UI. Provider credentials stay in the separate sign-in window."""

import hashlib
import secrets
import time
from pathlib import Path

import requests
import streamlit as st
from streamlit.components.v1 import declare_component

from dashboard.access import admit_login

browser_binding = declare_component(
    "account_browser_binding", path=str(Path(__file__).with_name("account_binding"))
)


def managed_login(base, operator_token):
    # A forwarded gateway URL cannot grant a session: the returning browser must
    # possess both its dashboard nonce and the ticket delivered after authentication.
    ticket = st.query_params.get("rb_ticket")
    flow = st.query_params.get("rb_flow")
    if ticket or flow:
        if not ticket or not flow:
            st.query_params.clear()
            st.error("Start sign-in from this dashboard in the same browser.")
            return
        binding = browser_binding(mode="redeem", flow=flow, key="return_" + str(flow), default=None)
        if binding is None:
            st.caption("Signing you in...")
            st.stop()
        verifier = binding.get("verifier", "")
        st.query_params.clear()
        if not verifier or not ticket or not flow:
            st.error("Start sign-in from this dashboard in the same browser.")
        else:
            try:
                response = requests.post(
                    base + "/auth/redeem",
                    json={"flow": flow, "verifier": verifier, "ticket": ticket},
                    timeout=(10, 20),
                )
                response.raise_for_status()
                data = response.json()
                st.session_state.clear()
                st.session_state.update(
                    authenticated=True,
                    authenticated_at=time.time(),
                    api_token=data["token"],
                    account_user=data["user_id"],
                )
                st.rerun()
            except requests.RequestException:
                st.error("Sign-in could not be verified. Start again.")
    st.title("RAG Bench")
    st.subheader("Your experiments, in one place")
    st.caption("Sign in or create an account to get started.")
    pending = st.session_state.get("account_flow")
    if pending and time.time() - pending["started_at"] > 600:
        st.session_state.pop("account_flow", None)
    if "account_flow" not in st.session_state:
        if not admit_login():
            st.error("Too many sign-in attempts. Try again in one minute.")
            return
        verifier = secrets.token_urlsafe(32)
        try:
            response = requests.post(
                base + "/api/auth/start",
                json={"challenge": hashlib.sha256(verifier.encode()).hexdigest()},
                headers={"Authorization": "Bearer " + operator_token},
                timeout=(10, 15),
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.account_flow = {
                **data,
                "verifier": verifier,
                "started_at": time.time(),
            }
        except requests.RequestException:
            st.error("Sign-in is temporarily unavailable. Please try again.")
            if st.button("Try again", width="stretch"):
                st.rerun()
    if pending := st.session_state.get("account_flow"):
        # The native component survives Streamlit Cloud's cookie forwarding.
        # It returns only a browser-held nonce, never an API/provider credential.
        binding = browser_binding(
            mode="prepare",
            flow=pending["flow"],
            verifier=pending["verifier"],
            key="prepare_" + pending["flow"],
            default=None,
        )
        if binding is None:
            st.caption("Preparing sign-in...")
            return
        if not binding.get("ready"):
            st.error("Allow site storage in your browser, then reload to sign in.")
            return
        st.link_button(
            "Continue with GitHub",
            pending["url"] + "&method=github",
            type="primary",
            width="stretch",
        )
        st.link_button("Continue with email", pending["url"] + "&method=email", width="stretch")
        st.caption("New here? GitHub creates your account automatically.")
    st.caption("Your experiments stay private to your account.")


def sign_out(base):
    token = st.session_state.get("api_token", "")
    if token.startswith("rb_session_"):
        try:
            response = requests.post(
                base + "/api/auth/logout",
                headers={"Authorization": "Bearer " + token},
                timeout=(5, 10),
            )
            if response.status_code not in {200, 401}:
                st.error("Sign-out could not be confirmed. Please try again.")
                return
        except requests.RequestException:
            st.error("Sign-out could not be confirmed. Please try again.")
            return
    st.session_state.clear()
    st.rerun()


def api_key_controls(api):
    if not st.session_state.get("account_user") or st.session_state.account_user == "owner":
        return
    with st.expander("API & CLI access"):
        st.caption(
            "Optional: create a 30-day key for scripts. Creating a new key replaces your previous key."
        )
        if st.button("Create access key", width="stretch"):
            try:
                data = api("POST", "/api/access-key").json()
                st.code(data["key"], language=None)
                st.caption("Copy it now. It won’t be shown again.")
            except Exception:
                st.error("Access key creation failed. Please try again.")
        if st.button("Revoke access key", width="stretch"):
            try:
                api("DELETE", "/api/access-key")
                st.success("Access key revoked.")
            except Exception:
                st.error("Revocation could not be confirmed. Please try again.")
