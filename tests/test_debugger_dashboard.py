"""Actual dashboard flow against the API with deterministic provider boundaries."""

from pathlib import Path
from threading import Lock

from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from tests.test_debugger import setup  # noqa: F401, F811
from tests.test_runs_api import DeferredExecutor


def test_full_debugger_flow(setup, monkeypatch):  # noqa: F811
    import backend.main as main

    store, run, record, calls, _ = setup
    run["status"] = "completed"
    record["status"] = "failed"
    store.save("run", run, run["id"])
    store.save("debug_run", record, record["id"])
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "run_lock", Lock())
    monkeypatch.setattr(main, "cancel_events", {})
    executor = DeferredExecutor()
    monkeypatch.setattr(main, "executor", executor)
    monkeypatch.setenv("API_TOKEN", "debug-ui")
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setenv("BACKEND_URL", "http://test-debugger")
    monkeypatch.setenv("APP_ENV", "development")
    client = TestClient(main.app)

    def request(method, url, **kwargs):
        response = client.request(
            method,
            url.replace("http://test-debugger", ""),
            headers=kwargs.get("headers"),
            json=kwargs.get("json"),
            params=kwargs.get("params"),
        )
        if executor.jobs:
            executor.finish()
        response.ok = response.is_success
        return response

    monkeypatch.setattr("requests.request", request)
    app = AppTest.from_file(str(Path("dashboard/app.py").resolve()), default_timeout=25)
    app.session_state["page"] = "Retrieval debugger"
    app.run()
    assert not app.exception
    app.text_area[0].set_value("Question 0?").run()
    next(b for b in app.button if b.label == "Run retrieval").click().run()
    assert not app.exception
    assert calls["generation"] == 0
    assert any("Cosine distance" in c.value for c in app.caption)
    next(b for b in app.button if b.label == "Generate answer from saved context").click().run()
    assert not app.exception and calls["generation"] == 1
    assert any("A saved answer" in t.value for t in app.text)
    next(n for n in app.number_input if n.label == "Final context passages").set_value(1).run()
    assert any("prior settings" in w.value for w in app.warning)
    next(b for b in app.button if b.label == "Run retrieval").click().run()
    assert not app.exception and calls["generation"] == 1
    next(b for b in app.button if b.label == "Compare saved evidence").click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "Open saved trace").click().run()
    assert not app.exception
    assert app.get("download_button")


def test_source_markers_render_safely():
    import json

    record = {
        "evidence_kind": "historical",
        "status": "recorded",
        "stage": "generated",
        "candidates": None,
        "selected": [{"text": "<script>untrusted</script>", "source": "test.txt", "page": 1}],
        "answer": "Evidence [1]; invalid [9].",
        "prompt_messages": None,
    }
    script = (
        "from dashboard.debugger import show_trace\nimport json\nshow_trace(json.loads("
        + repr(json.dumps(record))
        + "))"
    )
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert any("Invalid source marker [9]" in w.value for w in app.warning)
    assert any("<script>untrusted</script>" == t.value for t in app.text)
    assert any("#debug-passage-1" in m.value for m in app.markdown)
