"""Exercise worker admission, auth, history, cancellation and restart recovery through HTTP."""

from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.models import METRICS
from backend.storage import Store


class DeferredExecutor:
    def __init__(self):
        self.jobs = []

    def submit(self, function, *args):
        self.jobs.append((function, args))

    def finish(self):
        function, args = self.jobs.pop(0)
        function(*args)


@pytest.fixture
def api(tmp_path, monkeypatch):
    import backend.main as main
    import backend.pipeline as pipeline

    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setattr(main, "run_lock", Lock())
    monkeypatch.setattr(main, "cancel_events", {})
    executor = DeferredExecutor()
    monkeypatch.setattr(main, "executor", executor)
    monkeypatch.setenv("API_TOKEN", "test-only")
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    for key in ("GROQ_MODEL", "INFERENCE_BACKEND", "GROQ_REQUEST_INTERVAL", "JUDGE_EXAMPLES"):
        monkeypatch.delenv(key, raising=False)
    calls = []

    def retrieve(store, run_id, config, questions, indexes, checkpoint, stage):
        calls.append(("retrieval", config["id"]))
        return {
            i: [
                {
                    "text": "Saved evidence",
                    "source": "test.txt",
                    "page": 1,
                    "chunk_id": "0",
                    "distance": 0.1,
                    "token_start": 0,
                    "token_end": 2,
                }
            ]
            for i in indexes
        }

    class Metric:
        async def single_turn_ascore(self, sample, **kwargs):
            return 0.75

    monkeypatch.setattr(pipeline, "retrieve_questions", retrieve)
    monkeypatch.setattr(
        pipeline,
        "make_llm",
        lambda: SimpleNamespace(invoke=lambda _: SimpleNamespace(content="A useful answer")),
    )
    monkeypatch.setattr(pipeline, "load_embedder", lambda _: object())
    monkeypatch.setattr(pipeline, "make_metrics", lambda *_: {m: Metric() for m in METRICS})
    with TestClient(main.app) as client:
        client.headers["Authorization"] = "Bearer test-only"
        yield client, main, executor, calls
    # Do not leave a deferred worker holding the global run lock at fixture teardown.
    while executor.jobs:
        for event in main.cancel_events.values():
            event.set()
        executor.finish()


def start(client):
    demo = client.post("/api/demo").json()
    body = {k: demo[k] for k in ("document_set_id", "test_set_id", "configuration_ids")}
    result = client.post("/api/runs", json=body)
    assert result.status_code == 202
    return result.json()["id"], body


def test_history_cancel_retry_and_completed_results(api):
    client, main, executor, calls = api
    run_id, body = start(client)
    assert client.post("/api/runs", json=body).status_code == 409
    assert client.post(f"/api/runs/{run_id}/retry").status_code == 409
    client.headers.clear()
    for method, path in (
        ("GET", "/api/runs"),
        ("POST", f"/api/runs/{run_id}/cancel"),
        ("POST", f"/api/runs/{run_id}/retry"),
        ("GET", f"/api/runs/{run_id}/profile"),
        ("GET", f"/api/runs/{run_id}/profile/export"),
    ):
        assert client.request(method, path).status_code == 401
    client.headers["Authorization"] = "Bearer test-only"
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 202
    assert client.get(f"/api/runs/{run_id}").json()["cancel_requested"] is True
    executor.finish()
    assert not main.run_lock.locked()
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "cancelled"
    assert not calls
    assert client.post(f"/api/runs/{run_id}/retry").status_code == 202
    executor.finish()
    finished = client.get(f"/api/runs/{run_id}").json()
    assert finished["status"] == "completed" and finished["attempt"] == 2
    assert finished["completed"] == finished["scored"] == 20
    assert finished["valid_scores"] == 80 and len(finished["rows"]) == 20
    assert client.get(f"/api/runs/{run_id}/export").status_code == 200
    assert client.post(f"/api/runs/{run_id}/retry").status_code == 409
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409
    history = client.get("/api/runs").json()["runs"]
    assert history[0]["id"] == run_id and history[0]["scored"] == 20
    assert not {"rows", "questions", "retry_history"}.intersection(history[0])
    assert "profiling" not in history[0]
    profile = client.get(f"/api/runs/{run_id}/profile").json()["profiling"]
    assert [a["status"] for a in profile["attempts"]] == ["cancelled", "completed"]
    assert profile["summary"]["timings"]["scoring"] > 0
    phase_export = client.get(f"/api/runs/{run_id}/profile/export")
    assert phase_export.status_code == 200 and "rss_peak_mb" in phase_export.text
    assert "groq_total_tokens" in phase_export.text
    csv = client.get(f"/api/runs/{run_id}/export").text
    assert (
        "generation_seconds" in csv
        and "faithfulness_seconds" in csv
        and "groq_http_requests" in csv
    )


def test_legacy_profile_is_explicitly_missing_and_crashed_spans_are_interrupted(api):
    client, main, _, _ = api
    legacy = main.store.save("run", {"status": "completed", "rows": [], "configurations": []})
    assert client.get(f"/api/runs/{legacy['id']}/profile").json()["profiling"] is None
    assert client.get(f"/api/runs/{legacy['id']}/profile/export").status_code == 409
    interrupted = main.store.save(
        "run",
        {
            "status": "running",
            "rows": [],
            "profiling": {
                "version": 1,
                "attempts": [
                    {
                        "number": 1,
                        "status": "running",
                        "seconds": 3.5,
                        "spans": [{"stage": "scoring", "status": "running", "seconds": 2.0}],
                    }
                ],
            },
        },
    )
    client.__exit__(None, None, None)
    with TestClient(main.app) as restarted:
        restarted.headers["Authorization"] = "Bearer test-only"
        saved = restarted.get(f"/api/runs/{interrupted['id']}").json()
    attempt = saved["profiling"]["attempts"][0]
    assert (
        saved["status"] == "failed"
        and attempt["status"] == attempt["spans"][0]["status"] == "interrupted"
    )
    assert attempt["seconds"] == 3.5 and attempt["spans"][0]["seconds"] == 2.0


def test_retry_preserves_successes_and_rejects_changed_provenance(api, monkeypatch):
    client, main, executor, calls = api
    run_id, _ = start(client)
    executor.finish()
    run = main.store.get("run", run_id)
    run["status"] = "partial"
    run["rows"][0]["scores"]["faithfulness"] = None
    run["rows"][0]["errors"]["faithfulness"] = "RateLimitError"
    preserved = deepcopy(run["rows"][1:])
    main.store.save("run", run, run_id)
    monkeypatch.setenv("GROQ_MODEL", "changed-model")
    assert client.post(f"/api/runs/{run_id}/retry").status_code == 409
    assert not main.run_lock.locked()
    monkeypatch.delenv("GROQ_MODEL")
    assert client.post(f"/api/runs/{run_id}/retry").status_code == 202
    executor.finish()
    final = client.get(f"/api/runs/{run_id}").json()
    assert final["rows"][1:] == preserved and final["status"] == "completed"
    assert len(final["retry_history"]) == 1 and len(calls) == 2


def test_history_pagination_stays_ordered_when_old_run_is_updated(api):
    client, main, _, _ = api
    for index in range(3):
        main.store.save(
            "run",
            {
                "status": "completed",
                "stage": "Finished",
                "rows": [],
                "created_at": f"2026-09-{20 + index}T00:00:00Z",
                "completed": 2,
                "total": 2,
                "configurations": [],
            },
            str(index),
        )
    old = main.store.get("run", "0")
    main.store.save("run", old, "0")
    first = client.get("/api/runs?limit=2").json()
    assert [r["id"] for r in first["runs"]] == ["2", "1"] and first["next_offset"] == 2
    last = client.get("/api/runs?limit=2&offset=2").json()
    assert [r["id"] for r in last["runs"]] == ["0"] and last["next_offset"] is None
    assert client.get("/api/runs?limit=101").status_code == 422
    assert client.get("/api/runs?offset=-1").status_code == 422


def test_submission_failure_is_persisted_and_releases_lock(api, monkeypatch):
    client, main, executor, _ = api
    demo = client.post("/api/demo").json()
    body = {k: demo[k] for k in ("document_set_id", "test_set_id", "configuration_ids")}

    def fail(*args):
        raise RuntimeError("Executor closed")

    monkeypatch.setattr(executor, "submit", fail)
    with pytest.raises(RuntimeError, match="Executor closed"):
        client.post("/api/runs", json=body)
    assert not main.run_lock.locked() and not main.cancel_events
    assert main.store.list("run")[0]["status"] == "failed"


def test_startup_interruption_can_be_retried(api):
    client, main, executor, _ = api
    run_id, _ = start(client)
    # Simulate the interrupted process disappearing, then a fresh application lifespan.
    executor.jobs.clear()
    main.cancel_events.clear()
    main.run_lock.release()
    client.__exit__(None, None, None)
    with TestClient(main.app) as restarted:
        restarted.headers["Authorization"] = "Bearer test-only"
        assert restarted.get(f"/api/runs/{run_id}").json()["status"] == "failed"
        assert restarted.post(f"/api/runs/{run_id}/retry").status_code == 202
        executor.finish()
        assert restarted.get(f"/api/runs/{run_id}").json()["status"] == "completed"


def test_cancel_during_active_metric_preserves_result_and_releases_worker(api, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import backend.pipeline as pipeline

    client, main, _, _ = api
    entered, release = Event(), Event()

    class SlowMetric:
        async def single_turn_ascore(self, sample, **kwargs):
            entered.set()
            assert release.wait(timeout=5)
            return 0.6

    monkeypatch.setattr(pipeline, "make_metrics", lambda *_: {m: SlowMetric() for m in METRICS})
    with ThreadPoolExecutor(max_workers=1) as real_executor:
        futures = []

        def submit(function, *args):
            futures.append(real_executor.submit(function, *args))

        monkeypatch.setattr(main, "executor", SimpleNamespace(submit=submit))
        run_id, body = start(client)
        try:
            assert entered.wait(timeout=5)
            assert client.post(f"/api/runs/{run_id}/cancel").status_code == 202
            assert client.post("/api/runs", json=body).status_code == 409
        finally:
            release.set()
        futures[0].result(timeout=5)
        saved = client.get(f"/api/runs/{run_id}").json()
        assert saved["status"] == "cancelled" and not saved["cancel_requested"]
        assert saved["rows"][0]["scores"]["faithfulness"] == 0.6
        assert saved["rows"][0]["scores"]["answer_relevancy"] is None
        assert not main.run_lock.locked()


def test_mixed_engine_api_history_export_and_deletion(api, monkeypatch):
    import csv
    import io
    import json

    from graph_fakes import install

    client, main, executor, _ = api
    install(monkeypatch)
    docs = client.post(
        "/api/documents", files={"files": ("fixture.txt", b"Public fixture text", "text/plain")}
    ).json()
    questions = client.post(
        "/api/test-sets",
        json={
            "questions": [{"question": "Question 0?", "reference": "SECRET_REFERENCE_EVAL_ONLY"}]
        },
    ).json()
    configs = [
        client.post(
            "/api/configurations",
            json={
                "name": engine,
                "engine": engine,
                "candidate_k": 3,
                "context_k": 1,
                "rerank": True,
            },
        ).json()
        for engine in ("existing", "langgraph")
    ]
    response = client.post(
        "/api/runs",
        json={
            "document_set_id": docs["id"],
            "test_set_id": questions["id"],
            "configuration_ids": [c["id"] for c in configs],
        },
    )
    assert response.status_code == 202
    run_id = response.json()["id"]
    assert client.delete(f"/api/runs/{run_id}").status_code == 409
    executor.finish()
    result = client.get(f"/api/runs/{run_id}").json()
    assert result["status"] == "completed" and result["pending_metrics"] == 0
    history = client.get("/api/runs").json()["runs"][0]
    assert [c["engine"] for c in history["configurations"]] == ["existing", "langgraph"]
    rows = list(csv.DictReader(io.StringIO(client.get(f"/api/runs/{run_id}/export").text)))
    assert [r["engine"] for r in rows] == ["existing", "langgraph"]
    assert json.loads(rows[1]["run_provenance"])["graph_version"] == "1"
    assert "orchestration_seconds" in rows[1]
    assert client.delete(f"/api/runs/{run_id}").status_code == 204
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.get("/api/runs").json()["runs"] == []
