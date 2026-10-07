"""Answer-level investigation UI; drafts never submit benchmark jobs."""

from uuid import uuid4

import streamlit as st

from dashboard.observability import trace_link


def open_configuration_draft(api, investigation_id, index, edit_configuration, navigate):
    """Callbacks run before the sidebar navigation widget is instantiated."""
    try:
        draft = api(
            "POST", f"/api/investigations/{investigation_id}/experiments/{index}/draft"
        ).json()["configuration"]
        edit_configuration(draft)
        navigate("Upload / Setup")
    except RuntimeError as exc:
        st.session_state["investigation_error"] = str(exc)


def investigation_panel(api, run_id, row, edit_configuration, navigate):
    key = f"inv-{run_id}-{row['configuration_id']}-{row['question_index']}"
    with st.expander("Failure Investigator"):
        if error := st.session_state.pop("investigation_error", None):
            st.error(error)
        st.caption(
            "Get diagnostic hypotheses and experiments. Uses up to two additional model requests, "
            "including repair, with at most 3,000 output tokens each. Benchmark scores stay unchanged."
        )
        search = st.checkbox(
            "Search the original source snapshot (one query, up to 12 chunks)", key=key + "search"
        )
        items = (
            api(
                "GET",
                f"/api/runs/{run_id}/investigations",
                params={
                    "configuration_id": row["configuration_id"],
                    "question_index": row["question_index"],
                },
            )
            .json()
            .get("investigations", [])
        )
        if st.button("Investigate result", key=key + "start", disabled=bool(items)):
            api(
                "POST",
                f"/api/runs/{run_id}/investigations",
                json={
                    "configuration_id": row["configuration_id"],
                    "question_index": row["question_index"],
                    "search_sources": search,
                    "request_key": st.session_state.setdefault(key + "request", uuid4().hex),
                },
            )
            st.rerun()
        if not items:
            return
        selected = st.selectbox(
            "Saved investigation",
            [i["id"] for i in items],
            format_func=lambda ident: next(
                f"Version {i['version']} · {i['status']}" for i in items if i["id"] == ident
            ),
            key=key + "selected",
        )
        report = api("GET", f"/api/investigations/{selected}").json()
        trace_link(api, "investigation", selected)
        st.caption(f"{report['status'].title()} · {report['stage'].replace('_', ' ')}")
        if report["status"] in {"queued", "running"}:
            if st.button("Cancel investigation", key=key + "cancel"):
                api("POST", f"/api/investigations/{selected}/cancel")
                st.rerun()
            st.button("Refresh investigation", key=key + "refresh")
        elif report["status"] == "failed":
            st.error(report.get("error", "Investigation failed"))
            if st.button("Resume saved investigation", key=key + "resume"):
                api("POST", f"/api/investigations/{selected}/resume")
                st.rerun()
        if report["status"] not in {"queued", "running"}:
            if st.button("Run again", key=key + "again"):
                api(
                    "POST",
                    f"/api/runs/{run_id}/investigations",
                    json={
                        "configuration_id": row["configuration_id"],
                        "question_index": row["question_index"],
                        "search_sources": search,
                        "request_key": uuid4().hex,
                        "run_again": True,
                    },
                )
                st.rerun()
        diagnosis = report.get("diagnosis")
        if diagnosis and report["status"] == "completed":
            st.write(diagnosis["summary"])
            for h in diagnosis["hypotheses"]:
                st.markdown(
                    f"**{h['category'].replace('_', ' ').title()} · {h['strength']} evidence**"
                )
                st.write(h["rationale"])
                with st.expander("Supporting and contradicting evidence · " + h["category"]):
                    for label in ("supporting", "contradicting"):
                        for ref in h[label]:
                            evidence = report["evidence"][ref["evidence_id"]]
                            st.caption(
                                f"{label.title()} · {evidence['origin']} · {evidence.get('source', '')} "
                                f"page {evidence.get('page', '')} · chunk {evidence.get('chunk_id', '')}"
                            )
                            st.text(ref["quote"])
            for index, experiment in enumerate(report["experiments"]):
                st.markdown(f"**{experiment['title']}**")
                st.write(experiment["condition"])
                st.button(
                    "Create configuration draft",
                    key=key + f"draft-{index}",
                    on_click=open_configuration_draft,
                    args=(api, selected, index, edit_configuration, navigate),
                )
        with st.expander("Missing evidence and limitations"):
            for limitation in report.get("limitations", []) + (diagnosis or {}).get(
                "limitations", []
            ):
                st.write(limitation)
        with st.expander("Technical provenance and model usage"):
            st.json(
                {
                    k: report.get(k)
                    for k in (
                        "fingerprint",
                        "source_snapshot_id",
                        "provenance",
                        "budget",
                        "attempts",
                    )
                }
            )
        for format, mime in (("json", "application/json"), ("markdown", "text/markdown")):
            data = api(
                "GET", f"/api/investigations/{selected}/export", params={"format": format}
            ).content
            st.download_button(
                f"Download investigation {format.upper()}",
                data,
                file_name=f"investigation-{selected}.{'md' if format == 'markdown' else 'json'}",
                mime=mime,
                key=key + format,
            )
