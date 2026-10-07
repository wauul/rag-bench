"""Render saved timings and resource measurements without ML dependencies."""

import json

import pandas as pd
import streamlit as st

from backend.profiling_report import profile_csv_rows, spans, summarize_profile
from dashboard.charts import chart, phase_chart

STAGES = {
    "provider_check": "Groq readiness check",
    "chunking": "Chunking",
    "embedding_model_load": "Load retriever",
    "indexing": "Index documents",
    "retrieval": "Retrieve passages",
    "reranker_load": "Load reranker",
    "reranking": "Rerank passages",
    "judge_model_load": "Load judge embeddings",
    "judge_setup": "Set up judge",
    "generation": "Generate answers",
    "scoring": "Score answers",
    "index_reuse": "Reuse saved index",
    "orchestration_setup": "Prepare graph",
    "passage_selection": "Select final passages",
}


def number(value, suffix="", decimals=1):
    return "—" if value is None else f"{value:,.{decimals}f}{suffix}"


def performance_metrics(profile):
    summary = profile.get("summary") or summarize_profile(profile)
    usage = summary["groq"]
    first, second = st.columns(2)
    first.metric(
        "Indexing time",
        number(summary["timings"].get("indexing"), "s"),
        help="Embedding document chunks and writing the vector index. Model loading and chunking are separate.",
    )
    second.metric(
        "Peak backend RAM",
        number(summary.get("rss_peak_mb"), " MiB"),
        help="Sampled total backend process RSS, including models, indexes, Python and other allocations.",
    )
    first, second = st.columns(2)
    first.metric(
        "Groq HTTP requests",
        f"{usage['http_requests']:,}",
        help="Includes readiness checks, generation, judge calls and individual SDK retry attempts.",
    )
    second.metric(
        "Scoring time",
        number(summary["timings"].get("scoring"), "s"),
        help="Sum of metric attempts, including quota waits and retries. Generation is measured separately.",
    )
    return summary


def performance_panel(run):
    st.header("Understand the cost of your benchmark.")
    profile = run.get("profiling")
    if not profile:
        st.info(
            "Performance profiling was not recorded for this run. New runs record timings, memory and Groq usage automatically."
        )
        return
    summary = performance_metrics(profile)
    names = {c["id"]: c["name"] for c in run["configurations"]}
    if summary.get("configurations"):
        st.subheader("Where the time goes")
        chart(phase_chart(summary, names, STAGES), key="performance_phases")
        st.caption(
            "Cumulative recorded phase time across attempts. Phase durations may overlap; this is not wall-clock run duration. Unrecorded phases are omitted."
        )
    usage = summary["groq"]
    st.caption(
        "Includes every recorded attempt and retry. Timings are elapsed wall time; RAM is sampled every 100 ms and at phase boundaries. Values update at saved checkpoints."
    )
    if profile.get("coverage") != "full":
        st.warning(
            "This run began before profiling was enabled. Only work performed since profiling was enabled is measured."
        )
    if any(a["status"] == "interrupted" for a in profile["attempts"]):
        st.warning(
            "A worker was interrupted. Saved timings and usage are lower bounds; in-flight responses and memory peaks after the last checkpoint may be missing."
        )
    unresolved = (
        usage["http_requests"]
        - usage["http_successes"]
        - usage["http_failures"]
        - usage["transport_errors"]
    )
    st.subheader("Groq usage")
    left, middle, right = st.columns(3)
    left.metric(
        "Reported input tokens",
        number(usage["prompt_tokens"] if usage.get("prompt_token_reports") else None, decimals=0),
    )
    middle.metric(
        "Reported output tokens",
        number(
            usage["completion_tokens"] if usage.get("completion_token_reports") else None,
            decimals=0,
        ),
    )
    right.metric(
        "Reported total tokens",
        number(usage["total_tokens"] if usage.get("total_token_reports") else None, decimals=0),
    )
    st.caption(
        f"{usage['usage_reports']} of {usage['http_successes']} successful responses include complete token usage · "
        f"{usage['http_failures']} HTTP failures ({usage['rate_limit_responses']} rate limits) · "
        f"{usage['transport_errors']} transport errors · {unresolved} requests without a recorded outcome. "
        "Token totals use provider reports; failed requests without usage are not assumed free. This is usage for this run, not your account's remaining quota."
    )
    names = {c["id"]: c["name"] for c in run["configurations"]}
    configs = []
    for cid, item in summary["configurations"].items():
        configs.append(
            {
                "Configuration": names.get(cid, cid),
                "Indexing (s)": item["timings"].get("indexing"),
                "Generation (s)": item["timings"].get("generation"),
                "Scoring (s)": item["timings"].get("scoring"),
                "Peak process RSS (MiB)": item.get("rss_peak_mb"),
                "Groq requests": item["groq"]["http_requests"],
                "Reported tokens": item["groq"]["total_tokens"]
                if item["groq"].get("total_token_reports")
                else None,
            }
        )
    if configs:
        st.subheader("Compare configurations")
        st.dataframe(pd.DataFrame(configs), hide_index=True, width="stretch")
    graph_rows = [r for r in run["rows"] if "orchestration_seconds" in r]
    if graph_rows:
        st.caption(
            f"Graph framework and checkpoint overhead: {sum(r['orchestration_seconds'] for r in graph_rows):.3f}s across recorded attempts. "
            "Measured outside graph nodes; node persistence remains inside node time. Interrupted processes can leave lower bounds."
        )
        st.caption(
            "Saved indexes and passages may be reused on retry. Model-load phases disclose repeated loads; download cache warmth is not inferred. Compare equivalent cold or warm conditions."
        )
    entries = list(spans(profile))
    phases = [
        {"Phase": STAGES.get(stage, stage), "Elapsed (s)": value}
        for stage, value in summary["timings"].items()
    ]
    if phases:
        st.subheader("Time by phase")
        st.dataframe(pd.DataFrame(phases), hide_index=True, width="stretch")
    scoring = [e for e in entries if e["stage"] == "scoring"]
    if scoring:
        st.subheader("Scoring latency by metric")
        frame = pd.DataFrame(scoring)
        frame["Metric"] = frame["metric"].str.replace("_", " ").str.title()
        grouped = frame.groupby("Metric")["seconds"]
        table = grouped.agg(
            Attempts="count", **{"Total (s)": "sum", "Mean (s)": "mean", "Median (s)": "median"}
        )
        table["95th percentile (s)"] = grouped.quantile(0.95)
        st.dataframe(table, width="stretch")
        st.caption(
            "Includes successful, failed and interrupted metric attempts. Each metric's Groq requests and tokens are retained in the phase export."
        )
    loads = [e for e in entries if e.get("model")]
    if loads:
        with st.expander("Model loading and memory"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Attempt": e["attempt"],
                            "Configuration": names.get(e.get("configuration_id")),
                            "Phase": STAGES.get(e["stage"], e["stage"]),
                            "Model": e["model"],
                            "Load (s)": e["seconds"],
                            "RSS before (MiB)": e["rss_start_mb"],
                            "RSS after (MiB)": e["rss_end_mb"],
                            "RSS change (MiB)": e["rss_delta_mb"],
                            "Sampled peak (MiB)": e["rss_peak_mb"],
                        }
                        for e in loads
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "RSS changes include library and allocator effects; they do not isolate model weights. First use can include model downloads. Compare attempts on the same hardware and with the same cache conditions."
            )
    with st.expander("Recorded attempts"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        k: a.get(k)
                        for k in (
                            "number",
                            "status",
                            "started_at",
                            "finished_at",
                            "seconds",
                            "rss_peak_mb",
                            "memory_samples",
                        )
                    }
                    for a in profile["attempts"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.json([a.get("environment", {}) for a in profile["attempts"]])
    left, right = st.columns(2)
    csv_rows = profile_csv_rows(profile)
    for entry in csv_rows:
        entry["configuration_name"] = names.get(entry.get("configuration_id"))
    # Protect user-controlled names in spreadsheet downloads, as the API export does.
    frame = pd.DataFrame(csv_rows).map(
        lambda v: (
            "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
        )
    )
    left.download_button(
        "Download performance CSV",
        frame.to_csv(index=False),
        f"rag-bench-profile-{run['id']}.csv",
        "text/csv",
        width="stretch",
    )
    right.download_button(
        "Download performance JSON",
        json.dumps(profile, indent=2, allow_nan=False),
        f"rag-bench-profile-{run['id']}.json",
        "application/json",
        width="stretch",
    )
