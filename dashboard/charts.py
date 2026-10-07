"""Shared visual grammar and honest projections of saved measurements."""

import math
from html import escape
from textwrap import wrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COLORS = ["#655ce0", "#169b8a", "#ed9760", "#bc69a9"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
LABELS = dict(
    zip(METRICS, ["Faithfulness", "Answer relevancy", "Context precision", "Context recall"])
)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def axis_label(name):
    return "<br>".join(escape(line) for line in wrap(name, width=20))


def theme(figure, height=360):
    figure.update_layout(
        template="plotly_white",
        height=height,
        colorway=COLORS,
        font={"family": "sans-serif", "size": 12, "color": "#626a83"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 45, "l": 12, "r": 16, "b": 15},
        legend={"orientation": "h", "y": 1.16, "x": 0, "title_text": ""},
        hoverlabel={"bgcolor": "#20243b", "font_color": "white"},
    )
    figure.update_xaxes(gridcolor="#eceef5", zerolinecolor="#e2e5f0", automargin=True)
    figure.update_yaxes(gridcolor="#eceef5", zerolinecolor="#e2e5f0", automargin=True)
    return figure


def chart(figure, key=None):
    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={"displaylogo": False, "scrollZoom": False},
    )


def metric_chart(summaries, metric):
    valid = [s for s in summaries if finite(s.get(metric))]
    figure = go.Figure()
    # Keep each configuration's color stable even when a measurement is missing.
    for i, summary in enumerate(summaries):
        if summary not in valid:
            continue
        name = summary["name"]
        figure.add_trace(
            go.Bar(
                x=[summary[metric]],
                y=[summary.get("configuration_id", name)],
                orientation="h",
                name=name,
                marker_color=COLORS[i % len(COLORS)],
                text=[f"{summary[metric]:.3f}"],
                textposition="auto",
                customdata=[name],
                hovertemplate="%{customdata}<br>Score: %{x:.3f}<extra></extra>",
            )
        )
    figure.update_layout(showlegend=False, bargap=0.42)
    lower = min([0.0, *[s[metric] for s in valid]])
    figure.update_xaxes(range=[lower - 0.04, 1.05], title="Mean score · higher is better")
    figure.update_yaxes(
        autorange="reversed",
        title=None,
        tickmode="array",
        tickvals=[s.get("configuration_id", s["name"]) for s in valid],
        ticktext=[axis_label(s["name"]) for s in valid],
    )
    return theme(figure, 300)


def question_heatmap(run, metric):
    configurations = run["configurations"]
    count = len(run.get("questions", [])) or max(
        (r["question_index"] + 1 for r in run.get("rows", [])), default=0
    )
    lookup = {
        (r["configuration_id"], r["question_index"]): r.get("scores", {}).get(metric)
        for r in run.get("rows", [])
    }
    values = [
        [value if finite(value := lookup.get((c["id"], q))) else None for q in range(count)]
        for c in configurations
    ]
    lower = min([0.0, *[v for row in values for v in row if v is not None]])
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=[f"Q{i + 1}" for i in range(count)],
            y=[c["id"] for c in configurations],
            customdata=[[c["name"] for _ in range(count)] for c in configurations],
            zmin=lower,
            zmax=1,
            colorscale=[[0, "#eeecff"], [0.5, "#9b94eb"], [1, "#5147b8"]],
            xgap=4,
            ygap=6,
            hoverongaps=False,
            hovertemplate="%{customdata} · %{x}<br>Score: %{z:.3f}<extra></extra>",
            colorbar={"title": "Score", "thickness": 10},
        )
    )
    figure.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=[c["id"] for c in configurations],
        ticktext=[axis_label(c["name"]) for c in configurations],
    )
    return theme(figure, max(240, len(configurations) * 55 + 95))


def latency_frame(run):
    names = {c["id"]: c["name"] for c in run["configurations"]}
    return pd.DataFrame(
        [
            {"Configuration": names[r["configuration_id"]], "Seconds": r["latency_seconds"]}
            for r in run.get("rows", [])
            if r.get("answer") and finite(r.get("latency_seconds"))
        ],
        columns=["Configuration", "Seconds"],
    )


def latency_chart(run):
    frame = latency_frame(run)
    color_map = {c["name"]: COLORS[i % 4] for i, c in enumerate(run["configurations"])}
    figure = px.box(
        frame,
        x="Configuration",
        y="Seconds",
        color="Configuration",
        points="all",
        color_discrete_map=color_map,
    )
    figure.update_layout(showlegend=False)
    figure.update_yaxes(title="Generation + scoring time (s)", rangemode="tozero")
    return theme(figure)


def tradeoff_chart(run):
    latencies = latency_frame(run)
    figure = go.Figure()
    if latencies.empty:
        return theme(figure)
    for i, summary in enumerate(run.get("summary", [])):
        seconds = latencies.loc[latencies.Configuration == summary["name"], "Seconds"]
        if (
            run.get("status") != "completed"
            or not summary.get("complete")
            or not finite(summary.get("overall"))
            or len(seconds) != summary.get("expected_questions")
        ):
            continue
        figure.add_trace(
            go.Scatter(
                x=[seconds.mean()],
                y=[summary["overall"]],
                mode="markers",
                name=summary["name"],
                marker={"size": 18, "color": COLORS[i % 4]},
                hovertemplate="Mean processing time: %{x:.2f}s<br>Quality: %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_xaxes(
        title="Mean generation + scoring time (s) · lower is better", rangemode="tozero"
    )
    figure.update_yaxes(title="Complete overall quality · higher is better", rangemode="tozero")
    return theme(figure)


def phase_chart(summary, names, stages):
    records = [
        {
            "Configuration": names.get(cid, cid),
            "Phase": stages.get(stage, stage),
            "Seconds": seconds,
        }
        for cid, values in summary.get("configurations", {}).items()
        for stage, seconds in values.get("timings", {}).items()
        if finite(seconds)
    ]
    frame = pd.DataFrame(records, columns=["Configuration", "Phase", "Seconds"])
    figure = px.bar(
        frame,
        x="Seconds",
        y="Configuration",
        color="Phase",
        orientation="h",
        color_discrete_sequence=COLORS + ["#858ca3", "#aaa5e9", "#70bcb1"],
    )
    figure.update_xaxes(title="Recorded phase time (s)")
    figure.update_yaxes(title=None)
    return theme(figure, 340)
