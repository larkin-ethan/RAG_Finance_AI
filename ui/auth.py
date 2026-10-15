"""Login gate for the private deployment. A no-op when no password hash is configured.

Session state lives server-side per browser tab (Streamlit websocket session);
there's no long-lived cookie to steal. Refreshing the page or idling past the
timeout means logging in again.
"""

from __future__ import annotations

import os

import streamlit as st

from finrag.security.auth import IdleSession, LoginThrottle, verify_password

PASSWORD_HASH = os.environ.get("APP_PASSWORD_HASH", "")
IDLE_MINUTES = float(os.environ.get("SESSION_IDLE_MINUTES", "30"))


@st.cache_resource
def _throttle() -> LoginThrottle:
    return LoginThrottle()  # one per server process, shared by all tabs


def require_login() -> None:
    if not PASSWORD_HASH:
        return
    session: IdleSession | None = st.session_state.get("auth_session")
    if session and not session.expired():
        session.touch()
        if st.sidebar.button("Log out"):
            del st.session_state["auth_session"]
            st.rerun()
        return
    st.session_state.pop("auth_session", None)

    st.title("🔒 FinRAG")
    if session:
        st.info("Signed out after inactivity.")
    locked = _throttle().seconds_locked()
    if locked:
        st.error(f"Too many failed attempts. Try again in {int(locked) + 1} seconds.")
        st.stop()
    with st.form("login"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            ok = verify_password(PASSWORD_HASH, password)
            _throttle().record(ok)
            if ok:
                st.session_state["auth_session"] = IdleSession(IDLE_MINUTES)
                st.rerun()
            st.error("Incorrect password.")
    st.stop()
