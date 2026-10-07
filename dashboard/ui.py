"""Small presentation helpers shared by the dashboard pages."""

import math
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

MODEL_LABELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM L6",
    "BAAI/bge-small-en-v1.5": "BGE Small",
}
STATUS_LABELS = {
    "queued": "Queued",
    "running": "Running",
    "completed": "Complete",
    "partial": "Needs retry",
    "failed": "Failed",
    "cancelled": "Cancelled",
}
STATUS_COLORS = {
    "queued": "blue",
    "running": "violet",
    "completed": "green",
    "partial": "orange",
    "failed": "red",
    "cancelled": "gray",
}


def apply_styles():
    st.html(Path(__file__).with_name("styles.css"))


def header(eyebrow, title, description):
    st.title(title)
    st.caption(description)


def status_badge(status):
    st.badge(STATUS_LABELS.get(status, status.title()), color=STATUS_COLORS.get(status, "gray"))


def step_status(label, detail, ready):
    state = "ready" if ready else "pending"
    path = '<path d="m4 8 3 3 5-6"/>' if ready else '<circle cx="8" cy="8" r="4"/>'
    icon = f'<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">{path}</svg>'
    st.html(
        f'<div class="step-status {state}" role="group" aria-label="{escape(label)}: {state}"><span class="step-icon" aria-hidden="true">{icon}</span>'
        f"<div><strong>{escape(label)}</strong><span>{escape(detail)}</span></div></div>"
    )


def format_score(value):
    return f"{value:.3f}" if value is not None and math.isfinite(value) else "—"


def format_date(value):
    try:
        return (
            datetime.fromisoformat(value)
            .astimezone(ZoneInfo("Europe/Paris"))
            .strftime("%d %b %Y · %H:%M %Z")
        )
    except (ValueError, TypeError):
        return value or "Unknown date"


def config_description(config):
    context_k = config.get("context_k", config.get("top_k", 3))
    candidate_k = config.get("candidate_k", context_k)
    model = MODEL_LABELS.get(config.get("embedding_model"), "Saved configuration")
    return model, context_k, candidate_k
