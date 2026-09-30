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
        return {i: [{"text": "Saved evidence", "source": "test.txt", "page": 1,
                     "chunk_id": "0", "distance": 0.1, "token_start": 0, "token_end": 2}] for i in indexes}

    class Metric:
        async def single_turn_ascore(self, sample, **kwargs):
            return 0.75

    monkeypatch.setattr(pipeline, "retrieve_questions", retrieve)
    monkeypatch.setattr(pipeline, "make_llm", lambda: SimpleNamespace(invoke=lambda _: SimpleNamespace(content="A useful answer")))
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
    for method, path in (("GET", "/api/runs"), ("POST", f"/api/runs/{run_id}/cancel"),
                         ("POST", f"/api/runs/{run_id}/retry")):
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
        main.store.save("run", {"status": "completed", "stage": "Finished", "rows": [],
            "created_at": f"2026-09-{20 + index}T00:00:00Z", "completed": 2,
            "total": 2, "configurations": []}, str(index))
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
