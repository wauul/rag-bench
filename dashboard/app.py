"""Presentation only: ingestion, configuration validation and evaluations live in FastAPI."""
import os
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

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
MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
LABELS = {m: m.replace("_", " ").title() for m in METRICS}
COLORS = ["#007f73", "#de7c43", "#6778bf", "#bb5d87"]


def api(method, path, **kwargs):
    if not BASE:
        raise RuntimeError("Backend deployment is pending. Set BACKEND_URL in Streamlit secrets once the API is ready.")
    response = requests.request(method, BASE + path, timeout=(10, 120),
        headers={"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}, **kwargs)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text[:300]
        raise RuntimeError(f"Backend {response.status_code}: {detail}")
    return response


def launch_demo():
    try:
        demo = api("POST", "/api/demo").json()
        st.session_state.update(document_set_id=demo["document_set_id"], test_set_id=demo["test_set_id"],
                                configs=demo["configurations"])
        run = api("POST", "/api/runs", json={k: demo[k] for k in
            ["document_set_id", "test_set_id", "configuration_ids"]}).json()
        st.session_state.run_id = run["id"]
        st.session_state.page = "Run"
    except (requests.RequestException, RuntimeError) as exc:
        st.session_state.demo_error = str(exc)


st.markdown("""<style>
.stApp {background:#f7f8f6} h1,h2,h3 {letter-spacing:-.035em}
[data-testid="stMetric"] {background:white;padding:18px;border:1px solid #dee7e1;border-radius:12px}
[data-testid="stSidebar"] {background:#eaf0ec}
.eyebrow {font-size:12px;letter-spacing:.18em;color:#007f73;font-weight:700;margin-top:10px}
</style>""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## ◈ RAG Bench")
    st.caption("EVIDENCE OVER INTUITION")
    page = st.radio("Workspace", ["Upload / Setup", "Run", "Results", "History"], key="page")
    st.divider()
    st.caption("Local embeddings · ChromaDB\n\nGroq generation · Ragas evaluation")
    st.info("Real evaluations use Groq quota and may take several minutes. First use also downloads local models.")
    st.caption("Shared workspace: everyone with this dashboard's access can run evaluations and view results by ID.")


def setup():
    st.markdown('<p class="eyebrow">01 / BUILD YOUR BENCHMARK</p>', unsafe_allow_html=True)
    st.title("Better retrieval starts with evidence.")
    st.write("Compare configurations on the same questions. Inspect every answer and the passages behind it.")
    if not BASE:
        st.warning("Backend connection pending. This dashboard is deployed; evaluations become available once the API and Groq key are configured.")
    if st.session_state.get("demo_error"):
        st.error(st.session_state.pop("demo_error"))
    with st.container(border=True):
        st.subheader("Start with a ready-to-run benchmark")
        st.write("A community lab handbook, ten reference answers, and two configurations. Real retrieval. Real judge scores.")
        st.button("Try it now →", type="primary", on_click=launch_demo)
        st.caption("The demo varies multiple settings to showcase the tool. Change one setting at a time to isolate its effect.")
    left, right = st.columns(2)
    with left:
        st.subheader("1. Knowledge base")
        files = st.file_uploader("PDF or UTF-8 text", type=["pdf", "txt", "md"], accept_multiple_files=True)
        if st.button("Save documents", disabled=not files):
            result = api("POST", "/api/documents", files=[("files", (f.name, f.getvalue(), f.type)) for f in files]).json()
            st.session_state.document_set_id = result["id"]
            st.success(f"Saved {result['pages']} pages / {result['characters']:,} characters")
        if st.session_state.get("document_set_id"):
            st.caption("✓ Document set saved")
    with right:
        st.subheader("2. Reference questions")
        tab1, tab2 = st.tabs(["Upload", "Enter questions"])
        with tab1:
            upload = st.file_uploader("CSV / JSON: question, reference", type=["csv", "json"])
            if st.button("Save test file", disabled=upload is None):
                result = api("POST", "/api/test-sets/upload", files={"file": (upload.name, upload.getvalue())}).json()
                st.session_state.test_set_id = result["id"]
                st.success(f"Saved {len(result['questions'])} questions")
        with tab2:
            grid = st.data_editor(pd.DataFrame([{"question": "", "reference": ""}]), num_rows="dynamic", key="questions_grid", hide_index=True)
            if st.button("Save entered questions"):
                rows = grid.fillna("").to_dict("records")
                rows = [r for r in rows if r["question"].strip() or r["reference"].strip()]
                result = api("POST", "/api/test-sets", json={"questions": rows}).json()
                st.session_state.test_set_id = result["id"]
                st.success(f"Saved {len(rows)} questions")
        if st.session_state.get("test_set_id"):
            st.caption("✓ Test set saved")
    st.subheader("3. Configurations")
    st.caption("Define 2–4 configurations. Chunk sizes use a shared WordPiece tokenizer (32–240 tokens).")
    st.session_state.setdefault("configs", [])
    rerank = st.checkbox("Cross-encoder reranking", key="configuration_rerank")
    with st.form("configuration"):
        cols = st.columns([2, 2, 1, 1])
        name = cols[0].text_input("Name", value=f"Configuration {len(st.session_state.configs) + 1}")
        model = cols[1].selectbox("Embedding model", MODELS)
        size = cols[2].number_input("Chunk tokens", 32, 240, 192)
        overlap = cols[3].number_input("Overlap", 0, 239, 32)
        passage_cols = st.columns(2)
        context_k = passage_cols[0].number_input("Final passages", 1, 8, 3,
            help="Passages supplied to answer generation and all judge metrics.")
        candidate_k = passage_cols[1].number_input("Retrieval candidates", 1, 40, 20,
            disabled=not rerank, help="Retrieve this pool, rerank it, then keep the final passages.")
        if st.form_submit_button("Add configuration", disabled=len(st.session_state.configs) >= 4):
            result = api("POST", "/api/configurations", json={"name": name, "embedding_model": model,
                "chunk_size": size, "overlap": overlap, "context_k": context_k,
                "candidate_k": candidate_k if rerank else context_k, "rerank": rerank}).json()
            st.session_state.configs.append(result)
    if st.session_state.configs:
        st.dataframe(pd.DataFrame(st.session_state.configs).drop(columns="id"), hide_index=True, width="stretch")
        if st.button("Clear configuration selection"):
            st.session_state.configs = []
            st.rerun()


def run_page():
    st.markdown('<p class="eyebrow">02 / RUN THE EXPERIMENT</p>', unsafe_allow_html=True)
    st.title("One test set. Every configuration.")
    ready = all(st.session_state.get(k) for k in ["document_set_id", "test_set_id"]) and 2 <= len(st.session_state.get("configs", [])) <= 4
    if not ready:
        st.info("Start with Try it now, or save documents, reference questions and 2–4 configurations in Setup.")
    if st.button("Run evaluation", type="primary", disabled=not ready):
        result = api("POST", "/api/runs", json={"document_set_id": st.session_state.document_set_id,
            "test_set_id": st.session_state.test_set_id, "configuration_ids": [c["id"] for c in st.session_state.configs]}).json()
        st.session_state.run_id = result["id"]
    st.caption("Find saved benchmarks in History, or reopen one by ID below.")
    with st.expander("Open an existing run"):
        existing = st.text_input("Run ID")
        if st.button("Load run", disabled=not existing):
            api("GET", f"/api/runs/{existing}")
            st.session_state.run_id = existing
    if st.session_state.get("run_id"):
        poll_run()


def run_controls(run):
    if run["status"] in {"queued", "running"}:
        if run.get("cancel_requested"):
            st.info("Cancellation requested. The current operation will finish and save its result first.")
        if st.button("Cancel run", key=f"cancel_{run['id']}", disabled=run.get("cancel_requested", False)):
            api("POST", f"/api/runs/{run['id']}/cancel")
            st.rerun()
    elif run["status"] in {"partial", "failed", "cancelled"}:
        st.caption("Resume saved passages and answers, retrying only missing scores. The original model and judge settings must match.")
        if st.button("Retry missing work", key=f"retry_{run['id']}"):
            api("POST", f"/api/runs/{run['id']}/retry")
            st.session_state.run_id = run["id"]
            st.rerun()


def score_counts(run):
    rows = run["rows"]
    valid = sum(v is not None and math.isfinite(v) for r in rows for v in r["scores"].values())
    scored = sum(bool(r["answer"]) and not r["errors"] and
                 all(r["scores"].get(m) is not None and math.isfinite(r["scores"][m]) for m in METRICS) for r in rows)
    return scored, valid


@st.fragment(run_every="5s")
def poll_run():
    try:
        run = api("GET", f"/api/runs/{st.session_state.run_id}").json()
        st.code(run["id"], language=None)
        st.progress(run["completed"] / run["total"], text=run["stage"])
        scored, valid = score_counts(run)
        cols = st.columns(4)
        cols[0].metric("Status", run["status"].title())
        cols[1].metric("Processed answers", f"{run['completed']} / {run['total']}")
        cols[2].metric("Fully scored answers", f"{scored} / {run['total']}")
        cols[3].metric("Valid scores", f"{valid} / {run['total'] * len(METRICS)}")
        if run["status"] == "completed":
            st.success("Evaluation complete. Open Results to compare configurations.")
        elif run["status"] in {"partial", "failed", "cancelled"}:
            st.warning(run.get("error", "Some scores are unavailable. Inspect errors in Results."))
        run_controls(run)
    except (requests.RequestException, RuntimeError) as exc:
        st.error(str(exc))


def results():
    st.markdown('<p class="eyebrow">03 / FOLLOW THE EVIDENCE</p>', unsafe_allow_html=True)
    st.title("See what actually worked.")
    run_id = st.text_input("Run ID", value=st.session_state.get("run_id", ""), key="results_run_id")
    if not run_id:
        st.info("Run a benchmark first. Its summary and every retrieved passage will appear here.")
        return
    run = api("GET", f"/api/runs/{run_id}").json()
    scored, valid = score_counts(run)
    st.caption(f"{run['status'].title()} · {run['completed']}/{run['total']} processed · {scored} fully scored · {valid} valid scores · {run['created_at']}")
    run_controls(run)
    if not run["rows"]:
        st.info(run.get("error", run["stage"]))
        return
    summaries = run["summary"]
    all_complete = run["status"] == "completed" and all(s["complete"] for s in summaries)
    if all_complete:
        best = max(s["overall"] for s in summaries)
        winners = [s["name"] for s in summaries if abs(s["overall"] - best) < 1e-9]
        st.success(f"Highest equal-weight mean: {', '.join(winners)} · {best:.3f}")
    else:
        st.warning("Results are incomplete. Averages exclude missing scores; no overall winner is declared.")
    st.caption("Judge scores are estimates, not ground truth. The same Groq model generates and judges answers; compare with human review.")
    frame = pd.DataFrame(summaries).set_index("name")[METRICS]
    st.dataframe(frame.style.format("{:.3f}", na_rep="—").highlight_max(axis=0, color="#d0eee2"), width="stretch")
    counts = pd.DataFrame([{ "name": s["name"], **{m: f"{s['valid_counts'][m]}/{s['expected_questions']}" for m in METRICS}} for s in summaries])
    with st.expander("Valid score counts and run provenance"):
        st.dataframe(counts, hide_index=True)
        st.json(run["provenance"])
        st.json(run["configurations"])
    chart_kind = st.radio("Comparison view", ["Bars", "Radar"], horizontal=True)
    if chart_kind == "Bars":
        tidy = frame.reset_index().melt(id_vars="name", var_name="metric", value_name="score")
        fig = px.bar(tidy, x="metric", y="score", color="name", barmode="group", color_discrete_sequence=COLORS,
                     labels={"name": "Configuration", "metric": "", "score": "Mean Ragas score"})
        fig.update_yaxes(range=[-0.05, 1.05])
    else:
        fig = go.Figure()
        for i, (name, row) in enumerate(frame.iterrows()):
            values = [None if pd.isna(row[m]) else row[m] for m in METRICS]
            fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=list(LABELS.values()) + [LABELS[METRICS[0]]],
                name=name, fill="toself", line_color=COLORS[i], connectgaps=False))
        fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}})
    fig.update_layout(height=410, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Inspect a question")
    index = st.selectbox("Question", range(len(run["questions"])), format_func=lambda i: run["questions"][i]["question"])
    st.info("Reference: " + run["questions"][index]["reference"])
    columns = st.columns(len(run["configurations"]))
    for col, config in zip(columns, run["configurations"]):
        with col:
            st.markdown("### " + config["name"])
            row = next((r for r in run["rows"] if r["configuration_id"] == config["id"] and r["question_index"] == index), None)
            if not row:
                st.caption("Not evaluated yet")
                continue
            st.write(row["answer"] or ("Generation pending" if run["status"] in {"queued", "running"} else "Generation unavailable"))
            st.dataframe(pd.DataFrame([{"Metric": LABELS[m], "Score": row["scores"][m]} for m in METRICS]), hide_index=True)
            st.caption(f"Generation + evaluation: {row['latency_seconds']:.1f}s")
            if row["errors"]:
                st.error(row["errors"])
            for rank, chunk in enumerate(row["contexts"], 1):
                with st.expander(f"{rank}. {chunk['source']} · p.{chunk['page']}"):
                    st.text(chunk["text"])
                    st.caption(f"Cosine distance: {chunk['distance']:.4f} · tokens {chunk['token_start']}–{chunk['token_end']}")
                    if "rerank_score" in chunk:
                        st.caption(f"Cross-encoder score: {chunk['rerank_score']:.3f}")
    csv = api("GET", f"/api/runs/{run_id}/export").content
    st.download_button("Download full results CSV", csv, file_name=f"rag-bench-{run_id}.csv", mime="text/csv")


def open_saved_run(run_id, destination):
    st.session_state.run_id = run_id
    st.session_state.results_run_id = run_id
    st.session_state.page = destination


def history():
    st.markdown('<p class="eyebrow">04 / YOUR BENCHMARK HISTORY</p>', unsafe_allow_html=True)
    st.title("Pick up where you left off.")
    st.caption("Saved runs in this shared workspace. Open results or resume missing work from the Run page.")
    st.button("Refresh history")
    offset = st.session_state.get("history_offset", 0)
    response = api("GET", "/api/runs", params={"limit": 20, "offset": offset}).json()
    runs = response["runs"]
    if runs:
        st.dataframe(pd.DataFrame([{"Created": r["created_at"], "Status": r["status"],
            "Processed": f"{r['completed']}/{r['total']}",
            "Fully scored": r.get("scored") if r.get("scored") is not None else "—",
            "Configurations": ", ".join(c["name"] for c in r["configurations"]),
            "Run ID": r["id"]} for r in runs]), hide_index=True, width="stretch")
        choices = {r["id"]: r for r in runs}
        selected = st.selectbox("Saved run", list(choices),
            format_func=lambda rid: f"{choices[rid]['created_at']} · {choices[rid]['status']} · {rid[:8]}")
        left, right = st.columns(2)
        left.button("Open results", on_click=open_saved_run, args=(selected, "Results"))
        right.button("Open run", on_click=open_saved_run, args=(selected, "Run"))
    else:
        st.info("No saved runs on this page. Start a benchmark in Setup.")
    left, right = st.columns(2)
    if left.button("Newer runs", disabled=offset == 0):
        st.session_state.history_offset = max(0, offset - 20)
        st.rerun()
    if right.button("Older runs", disabled=response["next_offset"] is None):
        st.session_state.history_offset = response["next_offset"]
        st.rerun()


try:
    {"Upload / Setup": setup, "Run": run_page, "Results": results, "History": history}[page]()
except (requests.RequestException, RuntimeError) as exc:
    st.error(str(exc))
    st.caption("Check backend availability, API_TOKEN, and GROQ_API_KEY configuration. Free hosting may need time to wake up.")
