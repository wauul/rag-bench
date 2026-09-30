"""Exercise UI rendering with controlled API fixtures; these are not benchmark results."""
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard/app.py")


def test_setup_and_navigation_without_backend(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "")
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert "Backend connection pending" in app.warning[0].value
    app.button[0].click().run()
    assert not app.exception
    assert "Backend deployment is pending" in app.error[0].value
    app.radio[0].set_value("Run").run()
    assert not app.exception
    assert app.button[0].disabled
    app.radio[0].set_value("Results").run()
    assert not app.exception


def test_results_table_chart_and_drilldown(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    configs = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]
    summary = [{"configuration_id": c["id"], "name": c["name"], **{m: 0.5 for m in metrics},
                "overall": 0.5, "complete": True, "expected_questions": 1,
                "valid_counts": {m: 1 for m in metrics}} for c in configs]
    rows = [{"configuration_id": c["id"], "question_index": 0, "answer": "A test answer", "errors": {},
             "latency_seconds": 1.0, "scores": {m: 0.5 for m in metrics},
             "contexts": [{"source": "test.txt", "page": 1, "text": "A test passage", "distance": 0.2,
                           "token_start": 0, "token_end": 4}]} for c in configs]
    run = {"status": "completed", "completed": 2, "total": 2, "created_at": "test fixture", "rows": rows,
           "configurations": configs, "questions": [{"question": "A test question?", "reference": "A test answer"}],
           "summary": summary, "provenance": {"source": "UI test fixture"}}

    class Response:
        ok = True
        content = b"question,answer\nTest,Test\n"

        def json(self):
            return run

    with patch("requests.request", return_value=Response()):
        app = AppTest.from_file(APP)
        app.session_state["run_id"] = "fixture"
        app.session_state["page"] = "Results"
        app.run()
        assert not app.exception
        assert "A, B" in app.success[0].value
        assert len(app.get("plotly_chart")) == 1
        assert len(app.dataframe) >= 3
        next(r for r in app.radio if r.label == "Comparison view").set_value("Radar").run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == 1


def test_candidate_controls_send_separate_limits(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    sent = []

    class Response:
        ok = True

        def json(self):
            return {"id": "config", **sent[-1]}

    def request(method, url, **kwargs):
        assert method == "POST" and url.endswith("/api/configurations")
        sent.append(kwargs["json"])
        return Response()

    with patch("requests.request", side_effect=request):
        app = AppTest.from_file(APP).run()
        assert next(n for n in app.number_input if n.label == "Retrieval candidates").disabled
        app.checkbox[0].set_value(True).run()
        next(n for n in app.number_input if n.label == "Retrieval candidates").set_value(12)
        next(n for n in app.number_input if n.label == "Final passages").set_value(2)
        next(b for b in app.button if b.label == "Add configuration").click().run()
        assert not app.exception
        assert sent[0]["candidate_k"] == 12 and sent[0]["context_k"] == 2
        assert sent[0]["rerank"] is True


def test_history_opens_results_and_supports_retry_cancel_and_progress(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    config = {"id": "cfg", "name": "Test configuration"}
    run = {"id": "saved", "status": "partial", "stage": "Finished with errors",
        "completed": 1, "total": 1, "created_at": "2026-09-30", "configurations": [config],
        "questions": [{"question": "A test question?", "reference": "A test answer"}],
        "rows": [{"configuration_id": "cfg", "question_index": 0, "answer": "A test answer",
            "scores": {m: None for m in metrics}, "errors": {"faithfulness": "Quota failure"},
            "latency_seconds": 1, "contexts": []}],
        "summary": [{"configuration_id": "cfg", "name": config["name"], **{m: None for m in metrics},
            "overall": None, "complete": False, "expected_questions": 1,
            "valid_counts": {m: 0 for m in metrics}}], "provenance": {}}
    calls = []

    class Response:
        ok = True
        content = b"question,answer\nTest,Test\n"

        def __init__(self, value):
            self.value = value

        def json(self):
            return self.value

    def request(method, url, **kwargs):
        path = url.removeprefix("http://test-backend")
        calls.append((method, path))
        if path == "/api/runs":
            return Response({"runs": [run], "next_offset": None})
        if method == "POST" and path.endswith("/retry"):
            run.update(status="queued", stage="Queued to resume missing work")
        if method == "POST" and path.endswith("/cancel"):
            run["cancel_requested"] = True
        return Response(run)

    with patch("requests.request", side_effect=request):
        app = AppTest.from_file(APP)
        app.session_state["page"] = "History"
        app.session_state["results_run_id"] = "previous selection"
        app.run()
        assert not app.exception and len(app.dataframe) == 1
        next(b for b in app.button if b.label == "Open results").click().run()
        assert not app.exception
        assert next(r for r in app.radio if r.label == "Workspace").value == "Results"
        assert app.text_input[0].value == "saved"
        assert any("no overall winner" in w.value for w in app.warning)
        next(b for b in app.button if b.label == "Retry missing work").click().run()
        assert not app.exception and ("POST", "/api/runs/saved/retry") in calls
        next(r for r in app.radio if r.label == "Workspace").set_value("Run").run()
        values = {m.label: m.value for m in app.metric}
        assert values["Processed answers"] == "1 / 1"
        assert values["Fully scored answers"] == "0 / 1" and values["Valid scores"] == "0 / 4"
        next(b for b in app.button if b.label == "Cancel run").click().run()
        assert not app.exception and ("POST", "/api/runs/saved/cancel") in calls
        assert next(b for b in app.button if b.label == "Cancel run").disabled
        assert any("Cancellation requested" in i.value for i in app.info)
