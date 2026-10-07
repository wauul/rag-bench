"""Plan review and observed experiment comparisons in the existing workspace."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.ui import MODEL_LABELS


def optimization_page(api):
    st.title("Optimize configuration")
    st.write(
        "Compare a bounded set of retrieval settings, then check the selected configuration on questions kept aside."
    )
    history = api("GET", "/api/optimizations").json()["experiments"]
    # Stable labels prevent polling/status changes from invalidating the selection.
    choices = {r["id"]: f"{r['id'][:8]} · {r['plan']['baseline']['name']}" for r in history}
    if st.session_state.get("optimization_choice") not in {None, *choices}:
        st.session_state.optimization_choice = None
    selected = st.selectbox(
        "Experiment",
        [None, *choices],
        format_func=lambda v: choices.get(v, "Create a new experiment"),
        key="optimization_choice",
    )
    if selected:
        experiment_panel(api, selected)
        return
    inputs = api("GET", "/api/optimizations/inputs").json()
    configs = [c for c in inputs["configurations"] if c.get("engine") == "langgraph"]
    if not inputs["documents"] or not inputs["test_sets"] or not configs:
        st.info(
            "Upload documents and reference questions in Setup, and save a configuration using LangChain + LangGraph. Then return here."
        )
        return
    with st.form("optimization_plan"):
        docs = {
            d["id"]: f"{d['name']} · {d['pages']} pages · {d['id'][:8]}"
            for d in inputs["documents"]
        }
        tests = {t["id"]: f"{t['name']} · {t['count']} questions" for t in inputs["test_sets"]}
        baselines = {c["id"]: c["name"] for c in configs}
        document_id = st.selectbox("Document set", list(docs), format_func=docs.get)
        test_id = st.selectbox("Reference-question set", list(tests), format_func=tests.get)
        baseline_id = st.selectbox(
            "Baseline configuration", list(baselines), format_func=baselines.get
        )
        objective = st.radio(
            "Selection objective",
            ["quality", "latency"],
            format_func=lambda v: (
                "Best quality" if v == "quality" else "Lowest latency above a quality threshold"
            ),
        )
        threshold = st.slider("Minimum quality for the latency objective", 0.0, 1.0, 0.7, 0.05)
        col1, col2 = st.columns(2)
        with col1:
            trials = st.number_input(
                "Maximum trials, including baseline and held-out checks", 4, 20, 6
            )
        with col2:
            requests = st.number_input(
                "Maximum provider HTTP attempts, including retries", 1, 10000, 300
            )
        with st.expander("Retrieval search space"):
            st.caption(
                "Valid combinations are sampled deterministically. Overlap must be smaller than chunk size; candidate count must cover the final passages."
            )
            sizes = st.multiselect(
                "Chunk sizes in tokens", list(range(32, 241)), default=[128, 192, 240]
            )
            overlaps = st.multiselect("Overlap in tokens", list(range(240)), default=[0, 32])
            models = st.multiselect(
                "Embedding models",
                list(MODEL_LABELS),
                default=list(MODEL_LABELS),
                format_func=MODEL_LABELS.get,
            )
            candidates = st.multiselect("Retrieved candidates", list(range(1, 41)), default=[3, 8])
            contexts = st.multiselect("Final passages", list(range(1, 9)), default=[1, 3])
            rerank = st.multiselect(
                "Reranking",
                [False, True],
                default=[False, True],
                format_func=lambda v: "Enabled" if v else "Disabled",
            )
            seed = st.number_input("Split and sampling seed", 0, 2**32 - 1, 42)
        exploratory = st.checkbox(
            "For fewer than six questions, I acknowledge exploratory tuning without an independent held-out check."
        )
        st.caption(
            "Planning uses no provider quota. Generation, judging and retries share the request limit. No automatic upgrade or monetary estimate."
        )
        submitted = st.form_submit_button("Create plan for review", type="primary")
    if submitted:
        body = {
            "document_set_id": document_id,
            "test_set_id": test_id,
            "baseline_configuration_id": baseline_id,
            "objective": objective,
            "quality_threshold": threshold,
            "max_trials": trials,
            "max_requests": requests,
            "seed": seed,
            "exploratory_acknowledged": exploratory,
            "space": {
                "chunk_size": sizes,
                "overlap": overlaps,
                "embedding_model": models,
                "candidate_k": candidates,
                "context_k": contexts,
                "rerank": rerank,
            },
        }
        record = api("POST", "/api/optimizations", json=body).json()
        st.session_state.optimization_pending = record["id"]
        st.rerun()


def experiment_panel(api, identity):
    record = api("GET", "/api/optimizations/" + identity).json()
    plan, usage = record["plan"], record["usage"]
    split, request = plan["split"], plan["request"]
    st.subheader("Review plan" if record["status"] == "planned" else "Experiment progress")
    stages = {
        "validate": "Checking frozen inputs",
        "execute_tuning": "Comparing tuning results",
        "select_candidate": "Selecting from complete tuning results",
        "evaluate_heldout": "Checking held-out questions",
        "persist_report": "Saving comparison report",
    }
    st.write(
        f"Status: **{record['status'].replace('_', ' ')}** · {'Report ready' if record['status'] == 'completed' else stages.get(record['stage'], record['stage'])}"
    )
    st.write(
        f"{len(plan['candidates'])} candidate configurations + baseline · Tuning questions: {len(split['tuning'])} · Held-out questions: {len(split['heldout'])}"
    )
    st.write(f"Maximum {request['max_trials']} trials · {record['max_requests']} HTTP attempts")
    st.caption(f"Estimated attempts: {record['estimated_requests']}. {record['estimate_note']}")
    st.write(
        "Selection: "
        + (
            "best quality"
            if request["objective"] == "quality"
            else f"lowest serving latency with quality ≥ {request['quality_threshold']:.2f}"
        )
    )
    st.caption(plan["selection_policy"])
    configs = [plan["baseline"], *plan["candidates"]]
    st.dataframe(pd.DataFrame(configs).drop(columns=["engine"]), hide_index=True, width="stretch")
    for limitation in record["limitations"]:
        st.caption(limitation)
    with st.expander("Frozen conditions and question identities"):
        st.json(
            {
                "provenance": plan["provenance"],
                "split": split,
                "dataset_fingerprint": plan["dataset_fingerprint"],
                "plan_fingerprint": record["plan_fingerprint"],
                "amendments": record["amendments"],
                "invalid_combinations_skipped": record["rejected_combinations"],
            }
        )
    st.progress(
        min(usage["requests_reserved"] / record["max_requests"], 1),
        text=f"{usage['requests_reserved']} / {record['max_requests']} provider attempts reserved · {len(record['trials'])} trials scheduled",
    )
    st.caption(
        f"Provider-reported tokens: {usage['total_tokens']:,} from {usage['token_reports']} usage reports. Missing responses and crash reservations remain charged to the request budget."
    )
    if trial := record.get("current_trial"):
        st.caption(
            f"Current trial: {trial['stage']} · {trial.get('scored') or 0}/{trial['total']} questions fully scored"
        )
    status = record["status"]
    if status == "planned":
        if st.button("Approve plan and start experiments", type="primary"):
            api(
                "POST",
                f"/api/optimizations/{identity}/start",
                json={"plan_fingerprint": record["plan_fingerprint"]},
            )
            st.rerun()
    elif status in {"queued", "running", "cancelling"}:
        if status == "cancelling":
            st.info(
                "Stopping new work. An in-flight provider request may finish and its usage will be retained."
            )
        elif st.button("Cancel experiment"):
            api("POST", f"/api/optimizations/{identity}/cancel")
            st.rerun()
        st.button("Refresh progress", on_click=lambda: None)
        poll_experiment(api, identity)
    elif status in {"failed", "cancelled", "budget_exhausted"}:
        st.caption(
            "Resume retries only eligible missing work. The split, selected candidate and consumed budget stay fixed."
        )
        if st.button(
            "Resume missing work", disabled=usage["requests_reserved"] >= record["max_requests"]
        ):
            api(
                "POST",
                f"/api/optimizations/{identity}/resume",
                json={"plan_fingerprint": record["plan_fingerprint"]},
            )
            st.rerun()
        with st.expander("Increase the request budget"):
            with st.form("optimization_amendment"):
                new_limit = st.number_input(
                    "New maximum HTTP attempts",
                    record["max_requests"] + 1,
                    10001,
                    min(record["max_requests"] + 100, 10001),
                )
                reason = st.text_input("Reason for this recorded budget amendment")
                if st.form_submit_button("Record budget increase"):
                    api(
                        "POST",
                        f"/api/optimizations/{identity}/budget",
                        json={"max_requests": new_limit, "reason": reason},
                    )
                    st.rerun()
    if record.get("selection"):
        st.subheader(record["selection"]["outcome"])
        st.caption(
            f"Selected on tuning questions: {record['selection']['selected'] or 'none'}. Held-out results never change this selection."
        )
    comparison(record.get("tuning_results", []), "Tuning results")
    comparison(
        record.get("heldout_results", []),
        "Held-out check",
        "Not run: exploratory tuning has no independent held-out split."
        if split.get("exploratory")
        else "No saved results yet.",
    )
    st.download_button(
        "Export experiment report",
        json.dumps(record, indent=2),
        file_name=f"ragbench-optimization-{identity}.json",
        mime="application/json",
    )
    observed = {r["configuration_id"] for r in record.get("tuning_results", []) if r["eligible"]}
    if observed:
        with st.form("optimization_save"):
            config_id = st.selectbox(
                "Observed configuration to save", [c["id"] for c in configs if c["id"] in observed]
            )
            name = st.text_input("New configuration name", "Observed retrieval configuration")
            if st.form_submit_button("Save as configuration"):
                api(
                    "POST",
                    f"/api/optimizations/{identity}/configuration",
                    json={"configuration_id": config_id, "name": name},
                )
                st.success(
                    "Saved a new configuration. Existing configurations and defaults are unchanged."
                )


@st.fragment(run_every="4s")
def poll_experiment(api, identity):
    record = api("GET", "/api/optimizations/" + identity).json()
    if record["status"] not in {"queued", "running", "cancelling"}:
        st.rerun(scope="app")
    trial = record.get("current_trial", {})
    st.caption(
        f"Live: {trial.get('stage', 'Preparing trial')} · {record['usage']['requests_reserved']} attempts reserved"
    )


def comparison(results, title, empty_message="No saved results yet."):
    st.subheader(title)
    if not results:
        st.caption(empty_message)
        return
    rows = [
        {
            "Configuration": r["configuration_id"],
            "Status": r["status"],
            "Eligible": r["eligible"],
            "Quality": r["quality"],
            "Serving seconds/question": r["latency_seconds"],
            "Indexing seconds": r["timings"].get("indexing", 0),
            "Generation seconds": r["timings"].get("generation", 0),
            "Evaluation seconds": r["timings"].get("scoring", 0),
            "HTTP attempts": r["usage"].get("http_requests", 0),
            "Reported tokens": r["usage"].get("total_tokens")
            if r["usage"].get("total_token_reports")
            else None,
            "Failed rows": r["failures"],
            "Index": r["index_cache"],
        }
        for r in results
    ]
    frame = pd.DataFrame(rows)
    primary = [
        "Configuration",
        "Quality",
        "Serving seconds/question",
        "HTTP attempts",
        "Reported tokens",
    ]
    frame = frame[primary + [column for column in frame.columns if column not in primary]]
    st.dataframe(frame, hide_index=True, width="stretch")
    complete = frame[frame["Eligible"]].dropna(subset=["Quality", "Serving seconds/question"])
    if not complete.empty:
        figure = px.scatter(
            complete,
            x="Serving seconds/question",
            y="Quality",
            color="Configuration",
            hover_data=["HTTP attempts", "Reported tokens"],
        )
        figure.update_layout(font={"family": "sans-serif"}, margin={"t": 15, "b": 20})
        figure.update_yaxes(range=[-0.25, 1])
        figure.update_xaxes(rangemode="tozero")
        st.plotly_chart(figure, width="stretch")
    with st.expander("Metric coverage and trial references"):
        st.json(results)
