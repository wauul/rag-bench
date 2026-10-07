"""Exercise UI rendering with controlled API fixtures; these are not benchmark results."""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard/app.py")


def dashboard_test():
    # Allow cold Plotly/Streamlit imports on busy Windows and CI hosts.
    return AppTest.from_file(APP, default_timeout=15)


def test_setup_and_navigation_without_backend(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "")
    app = dashboard_test().run()
    assert not app.exception
    assert "Backend connection pending" in app.warning[0].value
    app.button[0].click().run()
    assert not app.exception
    assert "Backend deployment is pending" in app.error[0].value
    app.radio(key="page").set_value("Run").run()
    assert not app.exception
    assert app.button[0].disabled
    app.radio(key="page").set_value("Results").run()
    assert not app.exception


def test_results_table_chart_and_drilldown(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    configs = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]
    summary = [
        {
            "configuration_id": c["id"],
            "name": c["name"],
            **{m: 0.5 for m in metrics},
            "overall": 0.5,
            "complete": True,
            "expected_questions": 1,
            "valid_counts": {m: 1 for m in metrics},
        }
        for c in configs
    ]
    rows = [
        {
            "configuration_id": c["id"],
            "question_index": 0,
            "answer": "A test answer",
            "errors": {},
            "latency_seconds": 1.0,
            "scores": {m: 0.5 for m in metrics},
            "contexts": [
                {
                    "source": "test.txt",
                    "page": 1,
                    "text": "A test passage",
                    "distance": 0.2,
                    "token_start": 0,
                    "token_end": 4,
                }
            ],
        }
        for c in configs
    ]
    run = {
        "status": "completed",
        "completed": 2,
        "total": 2,
        "created_at": "test fixture",
        "rows": rows,
        "configurations": configs,
        "questions": [{"question": "A test question?", "reference": "A test answer"}],
        "summary": summary,
        "provenance": {"source": "UI test fixture"},
    }

    class Response:
        ok = True
        content = b"question,answer\nTest,Test\n"

        def json(self):
            return run

    with patch("requests.request", return_value=Response()):
        app = dashboard_test()
        app.session_state["run_id"] = "fixture"
        app.session_state["page"] = "Results"
        app.run()
        assert not app.exception
        assert "A, B" in app.success[0].value
        assert len(app.get("plotly_chart")) == 1
        assert len(app.dataframe) == 2
        assert any(t.value == "A test passage" for t in app.text)
        assert [tab.label for tab in app.tabs] == [
            "Score overview",
            "Inspect answers",
            "Performance",
        ]
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
        app = dashboard_test().run()
        assert next(n for n in app.number_input if n.label == "Retrieval candidates").disabled
        app.checkbox[0].set_value(True).run()
        next(r for r in app.radio if r.label == "Execution engine").set_value("langgraph")
        next(n for n in app.number_input if n.label == "Retrieval candidates").set_value(12)
        next(n for n in app.number_input if n.label == "Final passages").set_value(2)
        next(b for b in app.button if b.label == "Add configuration").click().run()
        assert not app.exception
        assert sent[0]["candidate_k"] == 12 and sent[0]["context_k"] == 2
        assert sent[0]["rerank"] is True
        assert sent[0]["engine"] == "langgraph"


def test_history_opens_results_and_supports_retry_cancel_and_progress(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    config = {"id": "cfg", "name": "Test configuration"}
    run = {
        "id": "saved",
        "status": "partial",
        "stage": "Finished with errors",
        "completed": 1,
        "total": 1,
        "created_at": "2026-09-30",
        "configurations": [config],
        "questions": [{"question": "A test question?", "reference": "A test answer"}],
        "rows": [
            {
                "configuration_id": "cfg",
                "question_index": 0,
                "answer": "A test answer",
                "scores": {m: None for m in metrics},
                "errors": {"faithfulness": "Quota failure"},
                "latency_seconds": 1,
                "contexts": [],
            }
        ],
        "summary": [
            {
                "configuration_id": "cfg",
                "name": config["name"],
                **{m: None for m in metrics},
                "overall": None,
                "complete": False,
                "expected_questions": 1,
                "valid_counts": {m: 0 for m in metrics},
            }
        ],
        "provenance": {},
    }
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
        app = dashboard_test()
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
        run.update(status="completed", stage="Complete")
        app.run()
        next(b for b in app.button if b.label == "View results").click().run()
        assert not app.exception
        assert next(r for r in app.radio if r.label == "Workspace").value == "Results"


def test_presets_edit_remove_and_guided_navigation(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    sent = []

    class Response:
        ok = True

        def json(self):
            return {"id": f"cfg{len(sent)}", **sent[-1]}

    def request(method, url, **kwargs):
        assert method == "POST" and url.endswith("/api/configurations")
        sent.append(kwargs["json"])
        return Response()

    with patch("requests.request", side_effect=request):
        app = dashboard_test()
        app.session_state["document_set_id"] = "docs"
        app.session_state["test_set_id"] = "questions"
        app.run()
        assert next(b for b in app.button if b.label == "Continue to evaluation").disabled
        next(b for b in app.button if b.label == "Add MiniLM baseline").click().run()
        next(b for b in app.button if b.label == "Add BGE + reranking").click().run()
        assert not app.exception and len(app.session_state["configs"]) == 2
        assert sent[0]["context_k"] == sent[0]["candidate_k"] == 3
        assert sent[1]["candidate_k"] == 20 and sent[1]["rerank"]
        assert not next(b for b in app.button if b.label == "Continue to evaluation").disabled
        next(b for b in app.button if b.key == "edit_cfg1").click().run()
        next(t for t in app.text_input if t.label == "Name").set_value("MiniLM tuned")
        next(n for n in app.number_input if n.label == "Chunk tokens").set_value(96)
        next(b for b in app.button if b.label == "Save changes").click().run()
        assert not app.exception
        assert len(app.session_state["configs"]) == 2
        assert app.session_state["configs"][0]["name"] == "MiniLM tuned"
        assert app.session_state["configs"][0]["chunk_size"] == 96
        next(b for b in app.button if b.label == "Continue to evaluation").click().run()
        assert next(r for r in app.radio if r.label == "Workspace").value == "Run"
        assert not next(b for b in app.button if b.label == "Run evaluation").disabled
        next(r for r in app.radio if r.label == "Workspace").set_value("Upload / Setup").run()
        next(b for b in app.button if b.key == "remove_cfg2").click().run()
        assert len(app.session_state["configs"]) == 1
        assert next(b for b in app.button if b.label == "Continue to evaluation").disabled


def test_invalid_configuration_and_blank_reference_rows_do_not_call_api(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    with patch("requests.request") as request:
        app = dashboard_test().run()
        next(b for b in app.button if b.label == "Save entered questions").click().run()
        assert not app.exception
        assert any("at least three characters" in e.value for e in app.error)
        next(n for n in app.number_input if n.label == "Chunk tokens").set_value(32)
        next(n for n in app.number_input if n.label == "Overlap").set_value(32)
        next(b for b in app.button if b.label == "Add configuration").click().run()
        assert not app.exception and any("Overlap must be smaller" in e.value for e in app.error)
        request.assert_not_called()


def test_history_search_filters_loaded_runs(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    runs = [
        {
            "id": "one",
            "created_at": "2026-09-30T12:00:00Z",
            "status": "completed",
            "completed": 2,
            "total": 2,
            "scored": 2,
            "configurations": [{"name": "MiniLM baseline"}],
        },
        {
            "id": "two",
            "created_at": "2026-09-29T12:00:00Z",
            "status": "partial",
            "completed": 2,
            "total": 2,
            "scored": 1,
            "configurations": [{"name": "BGE + reranking"}],
        },
    ]

    class Response:
        ok = True

        def json(self):
            return {"runs": runs, "next_offset": None}

    with patch("requests.request", return_value=Response()):
        app = dashboard_test()
        app.session_state["page"] = "History"
        app.run()
        next(t for t in app.text_input if t.label == "Search this page").set_value("bge").run()
        assert not app.exception and len(app.dataframe[0].value) == 1
        assert app.selectbox[0].value == "two"
        next(m for m in app.multiselect if m.label == "Status").set_value(["completed"]).run()
        assert not app.exception and not app.dataframe
        assert any("No runs match" in i.value for i in app.info)


def test_performance_tab_shows_measured_fields_and_missing_legacy_data(monkeypatch):
    from backend.profiling_report import empty_usage, summarize_profile

    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    usage = {
        **empty_usage(),
        "http_requests": 2,
        "http_successes": 1,
        "http_failures": 1,
        "rate_limit_responses": 1,
        "usage_reports": 1,
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "prompt_token_reports": 1,
        "completion_token_reports": 1,
        "total_token_reports": 1,
    }
    entries = [
        {
            "stage": stage,
            "configuration_id": "a",
            "question_index": 0 if stage != "indexing" else None,
            "metric": "faithfulness" if stage == "scoring" else None,
            "seconds": seconds,
            "status": "completed",
            "rss_peak_mb": 123,
            "groq": usage if stage == "scoring" else empty_usage(),
        }
        for stage, seconds in (("indexing", 2), ("generation", 3), ("scoring", 4))
    ]
    profile = {
        "version": 1,
        "coverage": "full",
        "attempts": [
            {"number": 1, "status": "completed", "seconds": 9, "rss_peak_mb": 123, "spans": entries}
        ],
    }
    profile["summary"] = summarize_profile(profile)
    run = {
        "id": "fixture",
        "status": "completed",
        "completed": 1,
        "total": 1,
        "created_at": "test fixture",
        "configurations": [{"id": "a", "name": "UI fixture"}],
        "questions": [{"question": "Fixture?", "reference": "Fixture"}],
        "rows": [
            {
                "configuration_id": "a",
                "question_index": 0,
                "answer": "Fixture",
                "errors": {},
                "contexts": [],
                "latency_seconds": 7,
                "scores": dict.fromkeys(metrics, 0.5),
            }
        ],
        "summary": [
            {
                "configuration_id": "a",
                "name": "UI fixture",
                **dict.fromkeys(metrics, 0.5),
                "overall": 0.5,
                "complete": True,
                "expected_questions": 1,
                "valid_counts": dict.fromkeys(metrics, 1),
            }
        ],
        "provenance": {"source": "UI fixture"},
        "profiling": profile,
    }

    class Response:
        ok = True
        content = b"question,answer\nFixture,Fixture\n"

        def json(self):
            return run

    with patch("requests.request", return_value=Response()):
        app = dashboard_test()
        app.session_state["run_id"] = "fixture"
        app.session_state["page"] = "Results"
        app.run()
        assert not app.exception
        values = {m.label: m.value for m in app.metric}
        assert values["Indexing time"] == "2.0s" and values["Scoring time"] == "4.0s"
        assert values["Peak backend RAM"] == "123.0 MiB" and values["Groq HTTP requests"] == "2"
        assert values["Reported total tokens"] == "14"
        assert any("1 rate limits" in c.value for c in app.caption)
        run["profiling"]["coverage"] = "since_enabled"
        run["profiling"]["attempts"][0]["status"] = "interrupted"
        app.run()
        assert not app.exception and any("lower bounds" in w.value for w in app.warning)
        assert any("before profiling was enabled" in w.value for w in app.warning)
        run.pop("profiling")
        app.run()
        assert not app.exception and any("not recorded" in i.value for i in app.info)
