"""Dashboard review/start/export/save and empty-state flows against controlled API."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def record(status="planned"):
    configuration = {
        "id": "baseline",
        "name": "Baseline",
        "engine": "langgraph",
        "chunk_size": 192,
        "overlap": 32,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "candidate_k": 3,
        "context_k": 1,
        "rerank": False,
    }
    result = {
        "run_id": "trial",
        "configuration_id": "baseline",
        "status": "completed",
        "eligible": True,
        "quality": 0.8,
        "latency_seconds": 1.2,
        "timings": {"indexing": 1, "generation": 1.1, "scoring": 3},
        "usage": {"http_requests": 10, "total_tokens": 1000, "total_token_reports": 10},
        "failures": 0,
        "index_cache": "fresh indexing",
    }
    return {
        "id": "experiment",
        "status": status,
        "stage": "Review plan",
        "plan_fingerprint": "frozen",
        "plan": {
            "baseline": configuration,
            "candidates": [{**configuration, "id": "candidate-1", "chunk_size": 128}],
            "split": {"tuning": [0, 1, 2, 3], "heldout": [4, 5]},
            "selection_policy": "Equal mean",
            "provenance": {},
            "dataset_fingerprint": "dataset",
            "request": {"objective": "quality", "max_trials": 4, "quality_threshold": 0.7},
        },
        "usage": {
            "requests_reserved": 20 if status == "completed" else 0,
            "total_tokens": 2000,
            "token_reports": 20,
        },
        "trials": [],
        "max_requests": 50,
        "estimated_requests": 36,
        "estimate_note": "Estimate only",
        "limitations": ["Observed dataset only"],
        "amendments": [],
        "rejected_combinations": 0,
        "tuning_results": [result] if status == "completed" else [],
        "heldout_results": [],
        **(
            {"selection": {"selected": "baseline", "outcome": "No improvement found"}}
            if status == "completed"
            else {}
        ),
    }


def test_review_start_completed_comparison_export_save(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    current = record()
    calls = []

    class Response:
        ok = True

        def __init__(self, value):
            self.value = deepcopy(value)

        def json(self):
            return self.value

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/api/optimizations"):
            return Response({"experiments": [current]})
        if url.endswith("/configuration"):
            return Response({"id": "saved"})
        if url.endswith("/start"):
            current.update(record("completed"))
        return Response(current)

    with patch("requests.request", request):
        app = AppTest.from_file(str(Path("dashboard/app.py").resolve())).run()
        app.radio(key="page").set_value("Optimize configuration")
        app.session_state.optimization_choice = "experiment"
        app.run()
        assert not app.exception
        assert len(app.dataframe) == 1
        assert not any(method == "POST" for method, _, _ in calls)
        next(b for b in app.button if b.label == "Approve plan and start experiments").click().run()
        assert not app.exception
        assert "No improvement found" in [h.value for h in app.subheader]
        assert len(app.dataframe) == 2
        assert any(
            url.endswith("/start") and kwargs["json"]["plan_fingerprint"] == "frozen"
            for _, url, kwargs in calls
        )
        next(b for b in app.button if b.label == "Save as configuration").click().run()
        assert not app.exception and app.success
        assert any(url.endswith("/configuration") for _, url, _ in calls)


def test_plan_form_submits_without_launching(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://test-backend")
    current = record()
    created = False
    calls = []

    class Response:
        ok = True

        def __init__(self, value):
            self.value = value

        def json(self):
            return self.value

    def request(method, url, **kwargs):
        nonlocal created
        calls.append((method, url))
        if url.endswith("/inputs"):
            return Response(
                {
                    "documents": [{"id": "docs", "name": "Knowledge", "pages": 2}],
                    "test_sets": [{"id": "test", "name": "Reference", "count": 6}],
                    "configurations": [current["plan"]["baseline"]],
                }
            )
        if method == "POST":
            created = True
            assert kwargs["json"]["space"]["candidate_k"] == [3, 8]
            return Response(current)
        if url.endswith("/api/optimizations"):
            return Response({"experiments": [current] if created else []})
        return Response(current)

    with patch("requests.request", request):
        app = AppTest.from_file(str(Path("dashboard/app.py").resolve())).run()
        app.radio(key="page").set_value("Optimize configuration").run()
        assert not app.exception
        next(b for b in app.button if b.label == "Create plan for review").click().run()
        assert not app.exception
        assert "Review plan" in [h.value for h in app.subheader]
        assert not any(url.endswith("/start") for _, url in calls)
