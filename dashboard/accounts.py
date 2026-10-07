"""Account sign-in UI. Provider credentials stay in the separate sign-in window."""

import hashlib
import secrets
import time

import requests
import streamlit as st

from dashboard.access import admit_login


def managed_login(base, operator_token):
    st.title("RAG Bench")
    st.caption("Compare retrieval. Understand the evidence.")
    st.subheader("Welcome to your workspace")
    st.write(
        "Sign in with GitHub or your email and password. New here? Create an account on the sign-in page."
    )
    if "account_flow" not in st.session_state and st.button(
        "Continue to sign in", type="primary", width="stretch"
    ):
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
            st.rerun()
        except requests.RequestException:
            st.error("Sign-in is temporarily unavailable. Please try again.")
    if pending := st.session_state.get("account_flow"):
        st.link_button(
            "Sign in with GitHub or email", pending["url"], type="primary", width="stretch"
        )
        st.caption(
            "Sign-in opens in a new tab. Keep this dashboard tab open; your workspace will appear here when you’re done."
        )

        @st.fragment(run_every="5s")
        def poll():
            if time.time() - pending["started_at"] > 600:
                st.session_state.pop("account_flow", None)
                st.warning("Sign-in expired. Start again.")
                st.rerun()
            try:
                response = requests.post(
                    base + "/auth/redeem",
                    json={"flow": pending["flow"], "verifier": pending["verifier"]},
                    timeout=(5, 10),
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.clear()
                    st.session_state.update(
                        authenticated=True,
                        authenticated_at=time.time(),
                        api_token=data["token"],
                        account_user=data["user_id"],
                    )
                    st.rerun()
                elif response.status_code != 202:
                    st.session_state.pop("account_flow", None)
                    st.error("Sign-in expired or was interrupted. Start again.")
            except requests.RequestException:
                st.caption("Waiting for a connection to the sign-in service…")

        poll()
        if st.button("Start again", width="stretch"):
            st.session_state.pop("account_flow", None)
            st.rerun()
    st.caption(
        "Your account keeps experiments private. Email verification is required. Forgot your password? Use the reset option on the sign-in page."
    )


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
