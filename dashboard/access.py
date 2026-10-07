"""Process-wide login throttling; state is shared across Streamlit sessions."""

import time
from collections import deque
from threading import Lock

import streamlit as st


@st.cache_resource
def login_state():
    return Lock(), deque()


def admit_login():
    lock, attempts = login_state()
    now = time.monotonic()
    with lock:
        while attempts and attempts[0] < now - 60:
            attempts.popleft()
        if len(attempts) >= 20:
            return False
        attempts.append(now)
        return True
