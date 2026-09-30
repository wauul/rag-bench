"""Presentation only: ingestion, configuration validation and evaluations live in FastAPI."""
import math
import os
from html import escape
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from dashboard.ui import (MODEL_LABELS, STATUS_LABELS, apply_styles, config_description,
                          format_date, format_score, header, status_badge, step_status)

load_dotenv()
st.set_page_config(page_title="RAG Bench", page_icon="◈", layout="wide")


def setting(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except FileNotFoundError:
        return os.getenv(name, default)


BASE = setting("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TOKEN = setting("API_TOKEN")
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
MODELS = list(MODEL_LABELS)
LABELS = {m: m.replace("_", " ").title() for m in METRICS}
COLORS = ["#655ce0", "#169b8a", "#ed9760", "#bc69a9"]


def api(method, path, **kwargs):
    if not BASE:
        raise RuntimeError("Backend deployment is pending. Set BACKEND_URL in Streamlit secrets once the API is ready.")
    response = requests.request(method, BASE + path, timeout=(10, 120),
        headers={"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}, **kwargs)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
            if isinstance(detail, list):
                detail = "; ".join(f"{item.get('loc', ['Input'])[-1]}: {item.get('msg', 'Invalid value')}" for item in detail)
        except ValueError:
            detail = response.text[:300]
        raise RuntimeError(f"Backend {response.status_code}: {detail}")
    return response


def navigate(destination):
    st.session_state.page = destination


def open_saved_run(run_id, destination):
    st.session_state.run_id = run_id
    st.session_state.results_run_id = run_id
    navigate(destination)


def launch_demo():
    try:
        demo = api("POST", "/api/demo").json()
        st.session_state.update(document_set_id=demo["document_set_id"], test_set_id=demo["test_set_id"],
                                configs=demo["configurations"], question_count=len(demo["questions"]),
                                document_info="Harbor handbook")
        run = api("POST", "/api/runs", json={k: demo[k] for k in
            ["document_set_id", "test_set_id", "configuration_ids"]}).json()
        open_saved_run(run["id"], "Run")
    except (requests.RequestException, RuntimeError) as exc:
        st.session_state.ui_error = str(exc)


def launch_run():
    try:
        run = api("POST", "/api/runs", json={"document_set_id": st.session_state.document_set_id,
            "test_set_id": st.session_state.test_set_id, "configuration_ids": [c["id"] for c in st.session_state.configs]}).json()
        open_saved_run(run["id"], "Run")
    except (requests.RequestException, RuntimeError) as exc:
        st.session_state.ui_error = str(exc)


def run_action(run_id, action):
    try:
        api("POST", f"/api/runs/{run_id}/{action}")
        if action == "retry":
            open_saved_run(run_id, "Run")
    except (requests.RequestException, RuntimeError) as exc:
        st.session_state.ui_error = str(exc)


def setup_ready():
    return all(st.session_state.get(k) for k in ["document_set_id", "test_set_id"]) and 2 <= len(st.session_state.get("configs", [])) <= 4


def setup_progress():
    cols = st.columns(3)
    count = st.session_state.get("question_count")
    with cols[0]:
        step_status("Documents", st.session_state.get("document_info", "Upload your knowledge base"), bool(st.session_state.get("document_set_id")))
    with cols[1]:
        step_status("Reference answers", f"{count} questions saved" if count else "Add a question set", bool(st.session_state.get("test_set_id")))
    with cols[2]:
        configs = len(st.session_state.get("configs", []))
        step_status("Configurations", f"{configs} of 2–4 selected", 2 <= configs <= 4)


def clear_draft():
    st.session_state.pop("config_draft", None)
    st.session_state.draft_revision = st.session_state.get("draft_revision", 0) + 1


def edit_configuration(config):
    st.session_state.config_draft = config.copy()
    st.session_state.draft_revision = st.session_state.get("draft_revision", 0) + 1


def remove_configuration(config_id):
    st.session_state.configs = [c for c in st.session_state.configs if c["id"] != config_id]
    if st.session_state.get("config_draft", {}).get("id") == config_id:
        clear_draft()


def save_configuration(values, editing_id=None):
    configs = st.session_state.setdefault("configs", [])
    if not values["name"].strip():
        raise RuntimeError("Give this configuration a name.")
    if any(c["id"] != editing_id and c["name"].strip().casefold() == values["name"].strip().casefold() for c in configs):
        raise RuntimeError("Choose a different name so the configurations are easy to compare.")
    if values["overlap"] >= values["chunk_size"]:
        raise RuntimeError("Overlap must be smaller than the chunk size.")
    if values["candidate_k"] < values["context_k"]:
        raise RuntimeError("Retrieval candidates must be at least as many as final passages.")
    if editing_id is None and len(configs) >= 4:
        raise RuntimeError("You can compare up to four configurations. Remove one to add another.")
    result = api("POST", "/api/configurations", json=values).json()
    if editing_id:
        st.session_state.configs = [result if c["id"] == editing_id else c for c in configs]
    else:
        configs.append(result)
    clear_draft()


def add_preset(model):
    try:
        rerank = model == MODELS[1]
        save_configuration({"name": "BGE + reranking" if rerank else "MiniLM baseline",
            "embedding_model": model, "chunk_size": 192, "overlap": 32,
            "context_k": 3, "candidate_k": 20 if rerank else 3, "rerank": rerank})
    except (requests.RequestException, RuntimeError) as exc:
        st.session_state.ui_error = str(exc)


def submit_configuration(key, editing_id):
    """Read the submitted form before rendering the next draft."""
    values = {field: st.session_state[f"{key}_{field}"] for field in
              ["name", "embedding_model", "chunk_size", "overlap", "context_k", "candidate_k", "rerank"]}
    if not values["rerank"]:
        values["candidate_k"] = values["context_k"]
    try:
        save_configuration(values, editing_id)
        st.session_state.ui_notice = "Configuration updated." if editing_id else "Configuration added."
    except (requests.RequestException, RuntimeError) as exc:
        st.session_state.ui_error = str(exc)


def configuration_cards():
    configs = st.session_state.get("configs", [])
    if not configs:
        return
    for start in range(0, len(configs), 2):
        columns = st.columns(2)
        for col, config in zip(columns, configs[start:start + 2]):
            with col, st.container(border=True, key=f"config_card_{config['id']}"):
                model, context_k, candidate_k = config_description(config)
                st.html(f'<div class="config-name">{escape(config["name"])}</div>'
                    f'<div class="config-model">{escape(model)}</div>'
                    f'<p class="config-spec">{config["chunk_size"]} token chunks · {config["overlap"]} overlap<br>'
                    f'{context_k} final passages · {candidate_k} candidates</p>')
                st.badge("Reranking on" if config.get("rerank") else "Vector retrieval", color="violet" if config.get("rerank") else "gray")
                left, right = st.columns(2)
                left.button("Edit", key=f"edit_{config['id']}", on_click=edit_configuration, args=(config,), width="stretch")
                right.button("Remove", key=f"remove_{config['id']}", on_click=remove_configuration, args=(config["id"],), type="tertiary", width="stretch")


def configuration_builder():
    draft = st.session_state.get("config_draft", {})
    editing = draft.get("id")
    revision = st.session_state.get("draft_revision", 0)
    context_default = draft.get("context_k", draft.get("top_k", 3))
    key = f"draft_{revision}"
    with st.expander("Edit configuration" if editing else "Customize a configuration", expanded=bool(editing) or not st.session_state.configs):
        rerank = st.checkbox("Cross-encoder reranking", value=draft.get("rerank", False), key=key + "_rerank",
            help="Score a larger pool of candidates, then keep the strongest passages.")
        with st.form(key + "_form"):
            left, right = st.columns(2)
            left.text_input("Name", value=draft.get("name", f"Configuration {len(st.session_state.configs) + 1}"), key=key + "_name")
            right.selectbox("Embedding model", MODELS, index=MODELS.index(draft.get("embedding_model", MODELS[0])), format_func=lambda m: MODEL_LABELS[m], key=key + "_embedding_model")
            left, right = st.columns(2)
            left.number_input("Chunk tokens", 32, 240, draft.get("chunk_size", 192), key=key + "_chunk_size", help="Size of each passage. Smaller chunks are more focused; larger chunks keep more context.")
            right.number_input("Overlap", 0, 239, draft.get("overlap", 32), key=key + "_overlap", help="Tokens repeated between neighboring passages. Must be smaller than the chunk size.")
            left, right = st.columns(2)
            left.number_input("Final passages", 1, 8, context_default, key=key + "_context_k", help="Passages used to generate and score each answer.")
            right.number_input("Retrieval candidates", 1, 40, draft.get("candidate_k", 20 if not draft else context_default), key=key + "_candidate_k",
                disabled=not rerank, help="The reranker chooses final passages from this pool.")
            st.form_submit_button("Save changes" if editing else "Add configuration", type="primary",
                disabled=not editing and len(st.session_state.configs) >= 4,
                on_click=submit_configuration, args=(key, editing))
        if editing:
            st.button("Cancel editing", on_click=clear_draft, type="tertiary")


def setup():
    header("WORKSPACE / SETUP", "Build your benchmark.", "Bring your documents, add reference answers, and compare retrieval configurations on the same questions.")
    if not BASE:
        st.warning("Backend connection pending. Evaluations become available once the API is configured.")
    with st.container(key="sample_card"):
        left, right = st.columns([3, 1], vertical_alignment="center")
        with left:
            st.html('<div class="sample-meta"><span>SAMPLE BENCHMARK</span><span>10 questions</span><span>2 configurations</span></div>'
                '<h2 class="sample-title">Get a feel for the results.</h2>'
                '<p class="sample-copy">Try the Harbor handbook with ready-made reference answers and a retrieval comparison.</p>')
        with right:
            st.button("Run sample benchmark", type="primary", on_click=launch_demo, width="stretch")
            st.caption("Uses Groq quota · takes several minutes")
    setup_progress()
    left, right = st.columns(2)
    with left, st.container(border=True, key="document_card"):
        st.header("1. Add your documents")
        st.caption("PDF, TXT or Markdown · up to 10 files · 5 MB each")
        files = st.file_uploader("Knowledge base", type=["pdf", "txt", "md"], accept_multiple_files=True,
            help="PDFs need extractable text. Use UTF-8 for TXT and Markdown.")
        if st.button("Save documents", disabled=not files, width="stretch"):
            with st.spinner("Saving documents…"):
                result = api("POST", "/api/documents", files=[("files", (f.name, f.getvalue(), f.type)) for f in files]).json()
            st.session_state.document_set_id = result["id"]
            st.session_state.document_info = f"{len(files)} {'file' if len(files) == 1 else 'files'} · {result['pages']} {'page' if result['pages'] == 1 else 'pages'}"
            st.session_state.ui_notice = f"Documents saved · {result['characters']:,} characters."
            st.rerun()
        if st.session_state.get("document_set_id"):
            st.badge("Documents saved", icon=":material/check:", color="green")
    with right, st.container(border=True, key="question_card"):
        st.header("2. Add reference answers")
        st.caption("1–20 questions with an expected answer for each")
        tab1, tab2 = st.tabs(["Upload file", "Write questions"])
        with tab1:
            upload = st.file_uploader("Question set", type=["csv", "json"], help="CSV columns: question, reference. JSON: a list of question/reference objects.")
            if st.button("Save test file", disabled=upload is None, width="stretch"):
                with st.spinner("Saving questions…"):
                    result = api("POST", "/api/test-sets/upload", files={"file": (upload.name, upload.getvalue())}).json()
                st.session_state.update(test_set_id=result["id"], question_count=len(result["questions"]), ui_notice="Reference questions saved.")
                st.rerun()
        with tab2:
            grid = st.data_editor(pd.DataFrame([{"question": "", "reference": ""}]), num_rows="dynamic", key="questions_grid", hide_index=True,
                column_config={"question": st.column_config.TextColumn("Question"), "reference": st.column_config.TextColumn("Reference answer")}, width="stretch")
            if st.button("Save entered questions", width="stretch"):
                rows = [r for r in grid.fillna("").to_dict("records") if r["question"].strip() or r["reference"].strip()]
                if not rows or any(len(r["question"].strip()) < 3 or len(r["reference"].strip()) < 3 for r in rows):
                    st.error("Add a question and reference answer of at least three characters in each row.")
                else:
                    result = api("POST", "/api/test-sets", json={"questions": rows}).json()
                    st.session_state.update(test_set_id=result["id"], question_count=len(result["questions"]), ui_notice="Reference questions saved.")
                    st.rerun()
        if st.session_state.get("test_set_id"):
            st.badge("Reference answers saved", icon=":material/check:", color="green")
    st.header("3. Choose what to compare")
    st.caption("Pick 2–4 configurations. These starting points use the same chunk size; change one setting at a time to isolate its effect.")
    st.session_state.setdefault("configs", [])
    left, right = st.columns(2)
    names = {c["name"].casefold() for c in st.session_state.configs}
    left.button("Add MiniLM baseline", on_click=add_preset, args=(MODELS[0],), disabled=len(names) >= 4 or "minilm baseline" in names, width="stretch")
    right.button("Add BGE + reranking", on_click=add_preset, args=(MODELS[1],), disabled=len(names) >= 4 or "bge + reranking" in names, width="stretch")
    configuration_cards()
    configuration_builder()
    st.divider()
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.header("Ready to compare?" if setup_ready() else "Finish the three steps above.")
        st.caption("Your documents and questions will be used for every selected configuration.")
    right.button("Continue to evaluation", type="primary", disabled=not setup_ready(), on_click=navigate, args=("Run",), width="stretch")


def run_controls(run):
    run_id = run.get("id", st.session_state.get("run_id"))
    if run["status"] in {"queued", "running"}:
        if run.get("cancel_requested"):
            st.info("Cancellation requested. The current operation will finish and save its result first.")
        st.button("Cancel run", key=f"cancel_{run_id}", disabled=run.get("cancel_requested", False), on_click=run_action, args=(run_id, "cancel"), type="tertiary")
    elif run["status"] in {"partial", "failed", "cancelled"}:
        st.caption("Keep saved answers and scores. Retry missing work after quota recovers; the original model and judge settings must match.")
        st.button("Retry missing work", key=f"retry_{run_id}", on_click=run_action, args=(run_id, "retry"), type="primary")


def score_counts(run):
    rows = run["rows"]
    valid = sum(v is not None and math.isfinite(v) for r in rows for v in r["scores"].values())
    scored = sum(bool(r["answer"]) and not r["errors"] and
                 all(r["scores"].get(m) is not None and math.isfinite(r["scores"][m]) for m in METRICS) for r in rows)
    return scored, valid


def run_page():
    header("WORKSPACE / EVALUATION", "Run your experiment.", "Every configuration gets the same documents and reference questions. Follow progress here as results arrive.")
    with st.container(border=True, key="run_setup_card"):
        setup_progress()
        left, right = st.columns([1, 2], vertical_alignment="center")
        left.button("Run evaluation", type="primary", disabled=not setup_ready(), on_click=launch_run, width="stretch")
        if not setup_ready():
            right.button("Complete setup", on_click=navigate, args=("Upload / Setup",), type="tertiary")
        else:
            right.caption("Generation and scoring use Groq quota. First use also downloads local models.")
    if st.session_state.get("run_id"):
        poll_run()
    else:
        st.info("No active benchmark selected. Start one above or open a saved run from History.")
        st.button("Browse history", on_click=navigate, args=("History",))
    with st.expander("Open a run by ID"):
        existing = st.text_input("Run ID", key="open_run_id", placeholder="Paste a saved run ID")
        if st.button("Load run", disabled=not existing):
            api("GET", f"/api/runs/{existing}")
            st.session_state.run_id = existing
            st.rerun()


@st.fragment(run_every="5s")
def poll_run():
    try:
        run = api("GET", f"/api/runs/{st.session_state.run_id}").json()
        with st.container(border=True, key="current_run_card"):
            st.header("Current benchmark")
            status_badge(run["status"])
            st.progress(run["completed"] / run["total"], text=run["stage"])
            scored, valid = score_counts(run)
            cols = st.columns(3)
            cols[0].metric("Processed answers", f"{run['completed']} / {run['total']}")
            cols[1].metric("Fully scored answers", f"{scored} / {run['total']}")
            cols[2].metric("Valid scores", f"{valid} / {run['total'] * len(METRICS)}")
            if run["status"] == "completed":
                st.success("Your comparison is ready. Explore scores, answers and supporting passages.")
                if st.button("View results", type="primary"):
                    st.session_state.update(run_id=run["id"], results_run_id=run["id"], next_page="Results")
                    st.rerun(scope="app")
            elif run["status"] in {"partial", "failed", "cancelled"}:
                st.warning(run.get("error", "Some scores are unavailable. Inspect results or retry the missing work."))
            run_controls(run)
            with st.expander("Run details"):
                st.caption(f"Created {format_date(run['created_at'])} · {len(run['configurations'])} configurations")
                st.code(run["id"], language=None)
    except (requests.RequestException, RuntimeError) as exc:
        st.error(str(exc))


def comparison_chart(frame, kind):
    if kind == "Bars":
        tidy = frame.reset_index().melt(id_vars="name", var_name="metric", value_name="score")
        tidy["metric"] = tidy["metric"].map(LABELS)
        fig = px.bar(tidy, x="metric", y="score", color="name", barmode="group", color_discrete_sequence=COLORS,
            labels={"name": "Configuration", "metric": "", "score": "Mean score"})
        minimum = tidy["score"].min()
        fig.update_yaxes(range=[min(-0.05, float(minimum) - 0.05) if pd.notna(minimum) else -0.05, 1.05], gridcolor="#eceef5")
        fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>Mean score: %{y:.3f}<extra>%{fullData.name}</extra>")
    else:
        fig = go.Figure()
        for i, (name, row) in enumerate(frame.iterrows()):
            values = [None if pd.isna(row[m]) else row[m] for m in METRICS]
            fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=list(LABELS.values()) + [LABELS[METRICS[0]]],
                name=name, fill="toself", line_color=COLORS[i], connectgaps=False))
        fig.update_layout(polar={"bgcolor": "white", "radialaxis": {"visible": True, "range": [0, 1], "gridcolor": "#e8eaf3"}})
        st.caption("Radar shows 0–1. Use Bars or the score table for any negative relevancy scores.")
    fig.update_layout(height=420, font={"family": "sans-serif", "color": "#626a83", "size": 12},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="",
        legend={"orientation": "h", "y": 1.16, "x": 0}, margin={"t": 55, "l": 10, "r": 10, "b": 10})
    return fig


def answer_explorer(run):
    st.header("Follow an answer back to its evidence.")
    index = st.selectbox("Question", range(len(run["questions"])), format_func=lambda i: run["questions"][i]["question"])
    with st.container(border=True, key="reference_card"):
        st.caption("REFERENCE ANSWER")
        st.write(run["questions"][index]["reference"])
    config_map = {c["id"]: c for c in run["configurations"]}
    selected = st.multiselect("Configurations to inspect", list(config_map), default=list(config_map)[:2],
        format_func=lambda cid: config_map[cid]["name"], help="Choose up to four; cards stay in two columns for readable answers.")
    if not selected:
        st.info("Select a configuration to inspect its answer and passages.")
    for start in range(0, len(selected), 2):
        columns = st.columns(2)
        for col, config_id in zip(columns, selected[start:start + 2]):
            config = config_map[config_id]
            with col, st.container(border=True, key=f"answer_card_{config_id}"):
                st.subheader(config["name"])
                row = next((r for r in run["rows"] if r["configuration_id"] == config_id and r["question_index"] == index), None)
                if not row:
                    st.caption("This answer has not been evaluated yet.")
                    continue
                st.write(row["answer"] or ("Generation pending" if run["status"] in {"queued", "running"} else "Generation unavailable"))
                st.html('<div class="score-strip">' + ''.join(f'<div><small>{LABELS[m]}</small><strong>{format_score(row["scores"].get(m))}</strong></div>' for m in METRICS) + '</div>')
                st.caption(f"Generation and scoring · {row['latency_seconds']:.1f}s")
                if row["errors"]:
                    with st.expander("Scoring errors", expanded=True):
                        for metric, error in row["errors"].items():
                            st.write(f"{LABELS.get(metric, metric.title())}: {error}")
                st.markdown("**Supporting passages**")
                for rank, chunk in enumerate(row["contexts"], 1):
                    with st.expander(f"Passage {rank} · {chunk['source']} · page {chunk['page']}"):
                        st.text(chunk["text"])
                        st.caption(f"Cosine distance {chunk['distance']:.4f} · tokens {chunk['token_start']}–{chunk['token_end']}")
                        if "rerank_score" in chunk:
                            st.caption(f"Reranking score {chunk['rerank_score']:.3f}")


def results():
    header("WORKSPACE / RESULTS", "See what worked.", "Compare the scores, then inspect the answers and passages behind them.")
    with st.expander("Open a run by ID", expanded=not st.session_state.get("run_id")):
        run_id = st.text_input("Run ID", value=st.session_state.get("run_id", ""), key="results_run_id", placeholder="Paste a saved run ID")
        st.button("Refresh results")
    if not run_id:
        st.info("Run a benchmark first, or choose a saved comparison from History.")
        st.button("Browse history", on_click=navigate, args=("History",))
        return
    run = api("GET", f"/api/runs/{run_id}").json()
    scored, valid = score_counts(run)
    status_badge(run["status"])
    st.caption(f"{format_date(run['created_at'])} · {scored}/{run['total']} fully scored answers · {valid}/{run['total'] * len(METRICS)} valid scores")
    run_controls(run)
    if not run["rows"]:
        st.info(run.get("error", run["stage"]))
        return
    summaries = run["summary"]
    all_complete = run["status"] == "completed" and bool(summaries) and all(s["complete"] for s in summaries)
    if all_complete:
        best = max(s["overall"] for s in summaries)
        winners = [s["name"] for s in summaries if abs(s["overall"] - best) < 1e-9]
        st.success(f"Highest equal-weight mean: {', '.join(winners)} · {best:.3f}")
    else:
        st.warning("Results are incomplete. Averages exclude missing scores; no overall winner is declared.")
    overview, answers = st.tabs(["Score overview", "Inspect answers"])
    with overview:
        for start in range(0, len(summaries), 2):
            columns = st.columns(2)
            for col, summary in zip(columns, summaries[start:start + 2]):
                col.metric(summary["name"], format_score(summary["overall"]), help="Equal-weight mean of the four metrics. Available only with complete scores.")
        with st.container(border=True, key="comparison_card"):
            st.header("Metric comparison")
            chart_kind = st.radio("Comparison view", ["Bars", "Radar"], horizontal=True, label_visibility="collapsed")
            frame = pd.DataFrame(summaries).set_index("name")[METRICS]
            st.plotly_chart(comparison_chart(frame, chart_kind), width="stretch")
            st.caption("Higher is generally better. Judge scores are estimates; review the underlying evidence before choosing a configuration.")
        with st.expander("Score table and valid counts", expanded=True):
            readable = frame.rename(columns=LABELS).rename_axis("Configuration")
            st.dataframe(readable.style.format("{:.3f}", na_rep="—").highlight_max(axis=0, color="#eeecff"), width="stretch")
            counts = pd.DataFrame([{"Configuration": s["name"], **{LABELS[m]: f"{s['valid_counts'][m]}/{s['expected_questions']}" for m in METRICS}} for s in summaries])
            st.caption("Valid scores / reference questions")
            st.dataframe(counts, hide_index=True, width="stretch")
        with st.expander("How this run was measured"):
            st.caption("The same Groq model generates and judges answers. Keep reference quality and judge bias in mind when comparing results.")
            st.json(run["provenance"])
            st.json(run["configurations"])
    with answers:
        answer_explorer(run)
    st.divider()
    csv = api("GET", f"/api/runs/{run_id}/export").content
    st.download_button("Download full results CSV", csv, file_name=f"rag-bench-{run_id}.csv", mime="text/csv", icon=":material/download:")


def history():
    header("WORKSPACE / HISTORY", "Your benchmark library.", "Reopen a comparison, inspect its evidence, or resume the work that is still missing.")
    st.button("Refresh history", icon=":material/refresh:")
    offset = st.session_state.get("history_offset", 0)
    response = api("GET", "/api/runs", params={"limit": 20, "offset": offset}).json()
    runs = response["runs"]
    with st.container(border=True, key="history_card"):
        left, right = st.columns([2, 1])
        query = left.text_input("Search this page", placeholder="Configuration name or run ID").strip().casefold()
        statuses = right.multiselect("Status", list(STATUS_LABELS), format_func=lambda s: STATUS_LABELS[s], placeholder="All statuses")
        filtered = [r for r in runs if (not statuses or r["status"] in statuses)
                    and (not query or query in r["id"].casefold() or any(query in c["name"].casefold() for c in r["configurations"]))]
        st.caption(f"Page {offset // 20 + 1} · {len(filtered)} of {len(runs)} runs shown. Search and filters apply to this page.")
        if filtered:
            st.dataframe(pd.DataFrame([{"Created": format_date(r["created_at"]), "Status": STATUS_LABELS.get(r["status"], r["status"]),
                "Processed": f"{r['completed']}/{r['total']}", "Fully scored": r.get("scored") if r.get("scored") is not None else "—",
                "Configurations": ", ".join(c["name"] for c in r["configurations"])} for r in filtered]), hide_index=True, width="stretch")
            choices = {r["id"]: r for r in filtered}
            selected = st.selectbox("Saved run", list(choices), format_func=lambda rid:
                f"{format_date(choices[rid]['created_at'])} · {', '.join(c['name'] for c in choices[rid]['configurations'])} · {rid[:8]}")
            left, right = st.columns(2)
            left.button("Open results", on_click=open_saved_run, args=(selected, "Results"), type="primary", width="stretch")
            right.button("Open run", on_click=open_saved_run, args=(selected, "Run"), width="stretch")
        elif runs:
            st.info("No runs match these filters. Clear the search or choose another status.")
        else:
            st.info("No saved runs on this page. Start a benchmark in Setup.")
            st.button("Build a benchmark", on_click=navigate, args=("Upload / Setup",))
    left, middle, right = st.columns([1, 2, 1])
    if left.button("Newer runs", disabled=offset == 0, width="stretch"):
        st.session_state.history_offset = max(0, offset - 20)
        st.rerun()
    middle.caption("Runs are stored in this shared workspace. Export results to keep a copy.")
    if right.button("Older runs", disabled=response["next_offset"] is None, width="stretch"):
        st.session_state.history_offset = response["next_offset"]
        st.rerun()


def sidebar():
    with st.sidebar:
        st.html('<div class="brand"><span class="brand-mark" aria-hidden="true">◈</span><div><strong>RAG Bench</strong><small>Retrieval, measured.</small></div></div>')
        labels = {"Upload / Setup": "01  Setup", "Run": "02  Evaluation", "Results": "03  Results", "History": "04  History"}
        page = st.radio("Workspace", list(labels), format_func=labels.get, key="page", label_visibility="collapsed")
        st.divider()
        st.caption("YOUR WORKSPACE")
        st.write(f"{len(st.session_state.get('configs', []))} configurations selected")
        if st.session_state.get("run_id"):
            st.button("Open latest results", on_click=open_saved_run, args=(st.session_state.run_id, "Results"), width="stretch")
        with st.expander("About this workspace"):
            st.caption("Local retrieval, Groq answers and Ragas scores. Real evaluations use Groq quota and can take several minutes.")
            st.caption("Everyone with dashboard access shares the workspace and can view runs. Uploaded data is sent to Groq for generation and evaluation.")
        return page


if destination := st.session_state.pop("next_page", None):
    navigate(destination)
apply_styles()
page = sidebar()
if st.session_state.get("ui_error"):
    st.error(st.session_state.pop("ui_error"))
if st.session_state.get("ui_notice"):
    st.success(st.session_state.pop("ui_notice"))
try:
    {"Upload / Setup": setup, "Run": run_page, "Results": results, "History": history}[page]()
except (requests.RequestException, RuntimeError) as exc:
    st.error(str(exc))
    st.caption("Check the backend connection and credentials. A free hosted backend may need time to wake up.")
