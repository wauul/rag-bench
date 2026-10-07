"""Only backend-approved private project links; no tracing credentials on the dashboard."""

from urllib.parse import urlsplit

import streamlit as st


def trace_link(api, workflow, record_id):
    try:
        url = api("GET", f"/api/observability/{workflow}/{record_id}").json().get("url")
        if url:
            parsed = urlsplit(url)
            if parsed.scheme == "https" and parsed.hostname in {
                "cloud.langfuse.com",
                "us.cloud.langfuse.com",
            }:
                st.link_button("Open private Langfuse trace", url)
                st.caption("Requires project access. Link availability does not confirm ingestion.")
    except RuntimeError:
        pass  # Dashboard remains usable during staged backend upgrades.
