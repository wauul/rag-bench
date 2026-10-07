"""Render the actual panel and exercise its explicit, non-launching draft action."""

import json
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from backend.investigator import execute
from tests.test_investigator import fake, fixture


def test_panel_report_export_and_draft(tmp_path):
    store, run, record = fixture.__wrapped__(tmp_path)
    record["version"] = 1
    store.save("investigation", record, record["id"])
    report = execute(store, record["id"], model_call=fake)
    from backend.investigation_api import public
    from backend.investigator import configuration_draft

    script = """
import json
import streamlit as st
from types import SimpleNamespace
from dashboard.investigation import investigation_panel
report = json.loads(REPORT)
draft = json.loads(DRAFT)
def api(method, path, **kwargs):
    if path.endswith('/investigations'):
        result = {'investigations': [report]}
    elif path.endswith('/draft'):
        st.session_state['draft_requested'] = True
        result = {'configuration': draft, 'launched': False}
    else:
        result = report
    return SimpleNamespace(json=lambda: result, content=b'export fixture')
def edit(value):
    st.session_state['edited'] = value
def navigate(value):
    st.session_state['page'] = value
st.radio('Workspace', ['Results', 'Upload / Setup'], key='page')
if not st.session_state.get('draft_requested'):
    investigation_panel(api, 'run', {'configuration_id': 'cfg', 'question_index': 0}, edit, navigate)
else:
    st.write('Editable draft ready')
""".replace("REPORT", repr(json.dumps(public(report)))).replace(
        "DRAFT", repr(json.dumps(configuration_draft(report, 0)))
    )
    app = AppTest.from_string(script, default_timeout=15).run()
    assert not app.exception
    assert any("The answer may" in m.value for m in app.markdown)
    assert any("seven days" in t.value for t in app.text)
    assert any("incomplete" in m.value for m in app.markdown)
    assert len(app.get("download_button")) == 2
    next(b for b in app.button if b.label == "Create configuration draft").click().run()
    assert not app.exception
    assert app.session_state["edited"]["context_k"] == 2
    assert app.session_state["page"] == "Upload / Setup"
    assert len(store.list("run")) == 1


def test_full_dashboard_draft_navigation_preserves_original_settings(tmp_path, monkeypatch):
    """Includes the real sidebar and setup form, including a partial result with no summary."""
    from backend.investigation_api import public
    from backend.investigator import configuration_draft

    store, run, record = fixture.__wrapped__(tmp_path)
    record["version"] = 1
    store.save("investigation", record, record["id"])
    report = execute(store, record["id"], model_call=fake)
    run.update(total=1, completed=1, summary=[], created_at="2026-10-07T12:00:00Z", stage="Partial")
    run["rows"][0]["latency_seconds"] = 0.1
    run["rows"][0]["contexts"][0].update(distance=0.1, token_start=0, token_end=5)
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    sent = []

    def request(method, url, **kwargs):
        from types import SimpleNamespace

        if method == "POST":
            sent.append((url, kwargs.get("json")))
        if url.endswith("/investigations"):
            result = {"investigations": [public(report)]}
        elif url.endswith("/draft"):
            result = {"configuration": configuration_draft(report, 0), "launched": False}
        elif url.endswith("/configurations"):
            result = {"id": "new-config", **kwargs["json"]}
        elif "/investigations/" in url:
            result = public(report)
        else:
            result = run
        return SimpleNamespace(ok=True, json=lambda: result, content=b"fixture export")

    with patch("requests.request", side_effect=request):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), default_timeout=15
        )
        app.session_state["run_id"] = run["id"]
        app.session_state["page"] = "Results"
        app.run()
        assert not app.exception
        next(b for b in app.button if b.label == "Create configuration draft").click().run()
        assert not app.exception
        assert app.session_state["page"] == "Upload / Setup"
        assert next(n for n in app.number_input if n.label == "Final passages").value == 2
        assert next(n for n in app.number_input if n.label == "Retrieval candidates").value == 4
        next(b for b in app.button if b.label == "Add configuration").click().run()
        assert not app.exception
    assert sent[-1][1]["candidate_k"] == 4
    assert sent[-1][1]["context_k"] == 2
    assert all(not url.endswith("/api/runs") for url, _ in sent)
