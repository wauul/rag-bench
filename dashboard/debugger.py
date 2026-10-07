"""Evidence-first Streamlit workspace; every action is explicit."""

import json
import re

import streamlit as st

from dashboard.observability import trace_link
from dashboard.ui import header


def show_trace(record):
    st.caption(
        f"{record['evidence_kind'].title()} evidence · {record['status']} · stage: {record.get('stage', 'unavailable')}"
    )
    if record.get("error"):
        st.error(record["error"])
    candidates, context, answer, technical = st.tabs(
        ["Candidates & reranking", "Final context", "Answer", "Prompt & provenance"]
    )
    with candidates:
        if record.get("candidates") is None:
            st.info(
                "Candidate retrieval was not recorded. Replay creates new evidence."
                if record["evidence_kind"] == "historical"
                else "Candidate retrieval has not completed."
            )
        else:
            ordered = record.get("ordered_candidates") or []
            ranks = {c["chunk_id"]: i for i, c in enumerate(ordered, 1)}
            scores = {c["chunk_id"]: c.get("rerank_score") for c in ordered}
            included = {c.get("chunk_id") for c in record.get("selected") or []}
            st.caption(
                "Cosine distance: lower is better. Reranker relevance score: higher is better. Scores across models are not calibrated or directly comparable."
            )
            for i, c in enumerate(record["candidates"], 1):
                cid = c.get("chunk_id")
                st.write(
                    f"Retrieval #{i} · final order #{ranks.get(cid, 'unavailable')} · {c.get('source', 'unavailable')} · page {c.get('page', 'unavailable')}"
                )
                st.caption(
                    f"Cosine distance: {c.get('distance', 'unavailable')} · Rerank score: {scores.get(cid) if scores.get(cid) is not None else 'unavailable'} · {'Included' if cid in included else 'Excluded / selection unavailable'}"
                )
                with st.expander(f"Full chunk {i} · {cid or 'identity unavailable'}"):
                    st.text(c["text"])
                    st.json({k: v for k, v in c.items() if k != "text"})
    with context:
        st.caption(record.get("context_policy", "Original context-budget decisions unavailable"))
        for i, c in enumerate(record.get("selected") or [], 1):
            st.markdown(f"<a id='debug-context-{i}'></a>", unsafe_allow_html=True)
            with st.expander(
                f"Passage [{i}] · {c.get('source', 'unavailable')} · page {c.get('page', 'unavailable')}",
                expanded=True,
            ):
                st.text(c["text"])
    with answer:
        st.text(record.get("answer") or "No answer generated.")
        for marker in dict.fromkeys(re.findall(r"\[(\d+)\]", record.get("answer") or "")):
            if len(marker) <= 6 and 1 <= int(marker) <= len(record.get("selected") or []):
                st.markdown(f"[Open supplied passage [{marker}]](#debug-passage-{marker})")
                st.markdown(f"<a id='debug-passage-{int(marker)}'></a>", unsafe_allow_html=True)
                with st.expander(f"Source [{marker}]"):
                    st.text(record["selected"][int(marker) - 1]["text"])
            else:
                st.warning(f"Invalid source marker [{marker}]: no supplied passage.")
        st.caption(
            "A valid source marker identifies a supplied passage; it does not prove the claim is supported."
        )
    with technical:
        with st.expander("Exact messages supplied to generation", expanded=True):
            if record.get("prompt_messages") is None:
                st.info(
                    "Prompt messages unavailable: generation has not run or historical messages were not recorded."
                )
            else:
                for message in record["prompt_messages"]:
                    st.caption(message["role"])
                    st.text(message["content"])
        st.json(
            {
                k: record.get(k, "unavailable")
                for k in (
                    "configuration",
                    "document_fingerprint",
                    "context_fingerprint",
                    "index",
                    "provenance",
                    "timings",
                    "usage",
                    "profiling",
                )
            }
        )


def debugger_page(api):
    header(
        "WORKSPACE / DEBUGGER",
        "Retrieval debugger",
        "Inspect retrieval, save the context, then choose whether to generate an answer.",
    )
    st.caption(
        "Debug runs use separate history and usage. They never add benchmark results or call Ragas."
    )
    pending = st.session_state.pop("debug_origin", None)
    if pending:
        st.session_state.debug_evidence = api(
            "GET", f"/api/debugger/historical/{pending[0]}/{pending[1]}/{pending[2]}"
        ).json()
        st.session_state.pop("debug_id", None)
    inputs = api("GET", "/api/debugger/inputs").json()
    history = api("GET", "/api/debugger").json()["runs"]
    new, saved = st.tabs(["Question & settings", "Saved debug runs"])
    record = st.session_state.get("debug_evidence")
    if debug_id := st.session_state.get("debug_id"):
        record = api("GET", f"/api/debugger/{debug_id}").json()
    with new:
        docs = inputs["document_sets"]
        if not docs:
            st.info("Upload documents in Setup to start a debug run.")
        else:
            doc_id = st.selectbox(
                "Document set",
                [d["id"] for d in docs],
                format_func=lambda value: (
                    ", ".join(next(d["sources"] for d in docs if d["id"] == value))
                    + " · "
                    + value[:8]
                ),
            )
            configs = inputs["configurations"]
            base = (
                st.selectbox("Saved configuration", configs, format_func=lambda c: c["name"])
                if configs
                else None
            )
            base = base or {
                "name": "Debugger",
                "chunk_size": 192,
                "overlap": 32,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "candidate_k": 8,
                "context_k": 3,
                "rerank": False,
                "engine": "langgraph",
            }
            question = st.text_area(
                "Question", value=record["question"] if record else "", max_chars=2000
            )
            with st.expander("Retrieval settings", expanded=True):
                left, right = st.columns(2)
                chunk_size = left.number_input(
                    "Chunk size (WordPiece tokens)", 32, 240, int(base["chunk_size"])
                )
                overlap = right.number_input(
                    "Overlap (WordPiece tokens)", 0, 239, int(base["overlap"])
                )
                candidate_k = left.number_input(
                    "Candidate passages", 1, 40, int(base.get("candidate_k") or base["context_k"])
                )
                context_k = right.number_input(
                    "Final context passages", 1, 8, int(base["context_k"])
                )
                model = st.selectbox(
                    "Embedding model",
                    ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
                    index=0 if base["embedding_model"].startswith("sentence") else 1,
                )
                rerank = st.checkbox("Rerank candidates", value=base["rerank"])
            config = {
                "name": base["name"],
                "engine": base.get("engine", "existing"),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "candidate_k": candidate_k,
                "context_k": context_k,
                "embedding_model": model,
                "rerank": rerank,
            }
            invalid = overlap >= chunk_size or candidate_k < context_k or len(question.strip()) < 3
            active = record and record["status"] in {"queued", "running"}
            stale = record and (
                record["question"] != question.strip()
                or record["document_set_id"] != doc_id
                or any(record["configuration"].get(k) != v for k, v in config.items())
            )
            if stale:
                st.warning(
                    "Displayed evidence belongs to prior settings. Rerun to inspect these settings; generation below uses the displayed saved context."
                )
            a, b = st.columns(2)
            for col, label, generate in (
                (a, "Run retrieval", False),
                (b, "Run retrieval and generate", True),
            ):
                if col.button(label, disabled=bool(invalid or active), width="stretch"):
                    result = api(
                        "POST",
                        "/api/debugger",
                        json={
                            "question": question,
                            "document_set_id": doc_id,
                            "configuration": config,
                            "generate": generate,
                        },
                    ).json()
                    st.session_state.debug_id = result["id"]
                    st.session_state.pop("debug_evidence", None)
                    st.rerun()
    with saved:
        if history:
            chosen = st.selectbox(
                "Saved run",
                history,
                format_func=lambda r: f"{r['question'][:70]} · {r['status']} · {r['id'][:8]}",
            )
            if st.button("Open saved trace"):
                st.session_state.debug_id = chosen["id"]
                st.session_state.pop("debug_evidence", None)
                st.rerun()
        else:
            st.info("Your first retrieval run will appear here.")
    if not record:
        return
    if record["evidence_kind"] == "historical":
        st.info("Original recorded benchmark evidence. Missing metadata remains unavailable.")
        if st.button("Replay with these settings"):
            o = record["origin"]
            result = api(
                "POST",
                f"/api/debugger/replay/{o['run_id']}/{o['configuration_id']}/{o['question_index']}",
            ).json()
            st.session_state.debug_id = result["id"]
            st.session_state.pop("debug_evidence", None)
            st.rerun()
    else:
        debug_id = record["id"]
        if record["status"] in {"queued", "running"}:
            st.info(f"Executing: {record['stage']}")
            if st.button("Cancel debug run"):
                api("POST", f"/api/debugger/{debug_id}/cancel")
            st.button("Refresh stage")
        elif record["status"] in {"paused", "failed"}:
            if record.get("selected") and not record.get("answer"):
                if st.button("Generate answer from saved context", type="primary"):
                    api(
                        "POST",
                        f"/api/debugger/{debug_id}/resume",
                        json={
                            "mode": "generate",
                            "context_fingerprint": record["context_fingerprint"],
                        },
                    )
                    st.rerun()
            if record["status"] == "failed" and st.button("Resume saved stages"):
                api("POST", f"/api/debugger/{debug_id}/resume", json={"mode": "retrieve"})
                st.rerun()
        st.download_button(
            "Export trace JSON",
            json.dumps(record, indent=2),
            file_name=f"debug-{debug_id}.json",
            mime="application/json",
        )
        others = [r for r in history if r["id"] != debug_id]
        with st.expander("Compare two saved runs"):
            if others:
                other = st.selectbox(
                    "Compare with",
                    others,
                    format_func=lambda r: r["question"][:50] + " · " + r["id"][:8],
                )
                if st.button("Compare saved evidence"):
                    comparison = api(
                        "GET", f"/api/debugger/{debug_id}/compare/{other['id']}"
                    ).json()
                    st.json({k: v for k, v in comparison.items() if k != "answers"})
                    left, right = st.columns(2)
                    left.text(comparison["answers"][0] or "No answer")
                    right.text(comparison["answers"][1] or "No answer")
            else:
                st.caption("Save a second run for the same question and documents to compare.")
        with st.expander("Delete this debug run"):
            st.caption("Removes this trace and graph checkpoints. Shared indexes are retained.")
            if st.button(
                "Delete saved debug run", disabled=record["status"] in {"queued", "running"}
            ):
                api("DELETE", f"/api/debugger/{debug_id}")
                st.session_state.pop("debug_id", None)
                st.rerun()
    trace_link(api, "debugger", record["id"])
    show_trace(record)
