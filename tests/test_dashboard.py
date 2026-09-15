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
