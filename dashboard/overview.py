"""Bounded, owner-scoped workspace overview; viewing never runs an evaluation."""

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.charts import COLORS, LABELS, chart, finite, metric_chart, theme
from dashboard.ui import STATUS_LABELS, format_date, header, status_badge


def overview_page(api, navigate, open_run, connected):
    title, action = st.columns([3, 1], vertical_alignment="center")
    with title:
        header("", "Experiment overview", "Your retrieval experiments, at a glance.")
    action.button(
        "New benchmark",
        icon=":material/add:",
        type="primary",
        width="stretch",
        on_click=navigate,
        args=("Upload / Setup",),
    )
    if not connected:
        st.warning("Backend connection pending. Connect the API to see your saved experiments.")
        empty_workspace(navigate)
        return
    response = api("GET", "/api/runs", params={"limit": 50, "include_summary": True}).json()
    runs = response["runs"]
    st.caption(
        f"Your latest {len(runs)} benchmarks · private to your access key · times in Paris"
        + (" · older runs in History" if response.get("next_offset") is not None else "")
    )
    if not runs:
        empty_workspace(navigate)
        return
    with st.container(key="overview_metrics"):
        columns = st.columns(4)
        columns[0].metric(
            "Benchmarks", len(runs), help="Latest 50 saved benchmarks; not all-time totals."
        )
        columns[1].metric("Complete", sum(r["status"] == "completed" for r in runs))
        columns[2].metric("Processed answers", sum(r.get("completed") or 0 for r in runs))
        profiles = [r["profile_summary"] for r in runs if r.get("profile_summary") is not None]
        columns[3].metric(
            "Recorded HTTP attempts",
            sum(p.get("groq", {}).get("http_requests", 0) for p in profiles) if profiles else "—",
            help=f"Recorded provider HTTP attempts across {len(profiles)}/{len(runs)} benchmarks. Missing profiles excluded. Not remaining quota or dollar spend.",
        )
    canvas, recent = st.columns([2.2, 1], gap="large")
    with canvas, st.container(border=True, key="overview_comparison"):
        st.subheader("Compare configurations")
        choices = {r["id"]: r for r in runs}
        first_measured = next((r["id"] for r in runs if r.get("summary")), runs[0]["id"])
        if st.session_state.get("overview_run") not in choices:
            st.session_state.overview_run = first_measured
        run_select, metric_select = st.columns([2, 1])
        selected = run_select.selectbox(
            "Benchmark",
            list(choices),
            key="overview_run",
            format_func=lambda rid: f"{format_date(choices[rid]['created_at'])} · {rid[:8]}",
        )
        current = choices[selected]
        summaries = current.get("summary") or []
        metric = metric_select.selectbox(
            "Metric", list(LABELS), format_func=LABELS.get, key="overview_metric"
        )
        if any(finite(s.get(metric)) for s in summaries):
            chart(metric_chart(summaries, metric), key="overview_quality")
            if current["status"] != "completed" or not all(s.get("complete") for s in summaries):
                st.caption(
                    "Partial results · available scores only. No overall winner is declared."
                )
            else:
                st.caption(
                    "Judge estimates · compare runs only when inputs and measurement conditions match."
                )
        else:
            st.info(
                "No measured scores for this benchmark yet. Open the run to check its progress."
            )
        st.button(
            "Explore this benchmark",
            icon=":material/arrow_forward:",
            on_click=open_run,
            args=(selected, "Results"),
            width="stretch",
        )
    with recent, st.container(border=True, key="recent_runs"):
        st.subheader("Recent benchmarks")
        for run in runs[:3]:
            status_badge(run["status"])
            st.caption(format_date(run["created_at"]))
            # Native text rendering keeps saved configuration names out of HTML.
            st.text(" · ".join(c["name"] for c in run.get("configurations", [])) or run["id"][:8])
            st.progress(
                min(1.0, max(0.0, (run.get("completed") or 0) / max(1, run.get("total") or 0))),
                text=f"{run.get('completed') or 0}/{run.get('total') or 0} processed",
            )
            st.button(
                "Open", key=f"overview_open_{run['id']}", on_click=open_run, args=(run["id"], "Run")
            )
        st.button("View all benchmarks", on_click=navigate, args=("History",), width="stretch")
    activity, coverage = st.columns([2.2, 1], gap="large")
    with activity, st.container(border=True):
        st.subheader("Benchmark activity")
        records = [
            {"Date": r["created_at"], "Status": STATUS_LABELS.get(r["status"], r["status"])}
            for r in runs
        ]
        frame = pd.DataFrame(records)
        frame["Date"] = (
            pd.to_datetime(frame["Date"], utc=True, errors="coerce")
            .dt.tz_convert("Europe/Paris")
            .dt.date
        )
        frame = (
            frame.dropna(subset=["Date"])
            .groupby(["Date", "Status"])
            .size()
            .reset_index(name="Benchmarks")
        )
        figure = px.bar(
            frame, x="Date", y="Benchmarks", color="Status", color_discrete_map=status_colors()
        )
        figure.update_yaxes(dtick=1, title="Benchmarks created")
        chart(theme(figure, 280), key="overview_activity")
        st.caption(
            "Run creation dates, grouped by their current status. Scope: latest 50 benchmarks."
        )
    with coverage, st.container(border=True):
        st.subheader("Run outcomes")
        counts = Counter(STATUS_LABELS.get(r["status"], r["status"]) for r in runs)
        figure = go.Figure(
            go.Bar(
                x=list(counts.values()),
                y=list(counts),
                orientation="h",
                marker_color=[status_colors().get(s, "#858ca3") for s in counts],
                text=list(counts.values()),
                textposition="auto",
                hovertemplate="%{y}: %{x}<extra></extra>",
            )
        )
        figure.update_xaxes(dtick=1, title="Benchmarks")
        figure.update_yaxes(autorange="reversed")
        chart(theme(figure, 280), key="overview_outcomes")


def status_colors():
    return {
        "Complete": COLORS[1],
        "Running": COLORS[0],
        "Queued": "#858ca3",
        "Needs retry": COLORS[2],
        "Failed": COLORS[3],
        "Cancelled": "#858ca3",
    }


def empty_workspace(navigate):
    with st.container(border=True, key="empty_workspace"):
        st.subheader("Your first comparison starts here")
        st.write("Add documents and reference answers, then compare two retrieval configurations.")
        columns = st.columns(3)
        for column, title, caption in zip(
            columns,
            ["Documents", "Reference answers", "Configurations"],
            [
                "Your knowledge base",
                "Questions with expected answers",
                "Two approaches, same inputs",
            ],
        ):
            column.markdown(f"**{title}**")
            column.caption(caption)
        st.button("Build a benchmark", on_click=navigate, args=("Upload / Setup",), type="primary")
    st.caption("Charts appear when measurements are saved. Running an evaluation uses Groq quota.")
