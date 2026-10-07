"""Real graph/checkpoint/index execution with deterministic model boundaries."""

import json
import os
import subprocess
import sys
from threading import Event, Lock

import pytest
from fastapi.testclient import TestClient

from backend import debugger
from backend.models import METRICS
from backend.storage import Store
from tests.graph_fakes import create_run, install


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for key in ("GROQ_MODEL", "INFERENCE_BACKEND", "GROQ_REQUEST_INTERVAL", "JUDGE_EXAMPLES"):
        monkeypatch.delenv(key, raising=False)
    calls, prompts = install(monkeypatch)
    store = Store(tmp_path)
    run = create_run(store)
    record = debugger.create(store, "Question 0?", run["document_set_id"], run["configurations"][0])
    return store, run, record, calls, prompts


def test_retrieval_pause_resume_exact_context(setup):
    store, run, record, calls, prompts = setup
    result = debugger.execute(store, record["id"], Event())
    assert result["status"] == "paused", result.get("error")
    assert not calls["generation"] and all(not calls[m] for m in METRICS)
    assert [c["chunk_id"] for c in result["candidates"]] == ["0", "1", "2"]
    assert result["selected"][0]["chunk_id"] == "2"
    assert result["ordered_candidates"][0]["rerank_score"] == 10
    assert "lower is better" in result["score_semantics"]["distance"]
    expected = result["context_fingerprint"]
    # Same persisted store and graph reconstructed; no model/index held across calls.
    result = debugger.execute(Store(store.root), record["id"], Event(), True, expected)
    assert result["status"] == "completed", result.get("error")
    assert calls["generation"] == 1 and calls["rerank"] == 1
    assert result["prompt_messages"] == debugger.exact_prompt(
        result["question"], result["selected"]
    )
    assert prompts[0] == [(m["role"], m["content"]) for m in result["prompt_messages"]]
    assert "Best evidence" in prompts[0][1][1] and "Nearest text" not in prompts[0][1][1]
    assert not store.get("run", run["id"])["rows"]


def test_fresh_process_resume(setup):
    store, _, record, _, _ = setup
    paused = debugger.execute(store, record["id"], Event())
    script = """
import sys
from threading import Event
from pytest import MonkeyPatch
from tests.graph_fakes import install
from backend.storage import Store
from backend.debugger import execute
m = MonkeyPatch()
calls, prompts = install(m)
r = execute(Store(sys.argv[1]), sys.argv[2], Event(), True, sys.argv[3])
assert r['status'] == 'completed', r
assert calls['generation'] == 1 and calls['rerank'] == 0
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(store.root),
            record["id"],
            paused["context_fingerprint"],
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr


def test_application_answer_reconciles_lost_checkpoint(setup):
    from backend.graph_pipeline import checkpointer

    store, _, record, calls, _ = setup
    completed = debugger.execute(store, record["id"], Event(), True)
    assert completed["status"] == "completed"
    completed["status"] = "failed"
    store.save("debug_run", completed, completed["id"])
    with checkpointer(store) as saver:
        saver.delete_thread("debug-" + completed["id"])
    recovered = debugger.execute(Store(store.root), completed["id"], Event(), False)
    assert recovered["status"] == "completed" and calls["generation"] == 1


def test_benchmark_component_parity(setup):
    from backend.pipeline import execute_run

    store, run, record, _, _ = setup
    execute_run(store, run["id"], Event())
    benchmark = store.get("run", run["id"])
    result = debugger.execute(store, record["id"], Event(), True)
    assert result["status"] == "completed"
    for row in benchmark["rows"]:
        assert row["contexts"] == result["selected"]
        assert row["prompt_messages"] == result["prompt_messages"]


def test_cancel_and_tamper(setup):
    store, _, record, calls, _ = setup
    event = Event()
    event.set()
    result = debugger.execute(store, record["id"], event)
    assert result["status"] == "cancelled" and not calls["generation"]
    record = debugger.create(
        store, "Question 0?", record["document_set_id"], record["configuration"]
    )
    result = debugger.execute(store, record["id"], Event())
    result["selected"][0]["text"] = "changed"
    store.save("debug_run", result, result["id"])
    result = debugger.execute(store, record["id"], Event(), True, result["context_fingerprint"])
    assert result["status"] == "failed" and not calls["generation"]


def test_retrieval_resume_never_generates_after_provider_failure(setup, monkeypatch):
    from backend import pipeline

    store, _, record, calls, _ = setup
    paused = debugger.execute(store, record["id"], Event())
    original = pipeline.make_llm

    def broken():
        raise TimeoutError("secret provider error")

    monkeypatch.setattr(pipeline, "make_llm", broken)
    failed = debugger.execute(store, record["id"], Event(), True, paused["context_fingerprint"])
    assert failed["status"] == "failed"
    monkeypatch.setattr(pipeline, "make_llm", original)
    resumed = debugger.execute(store, record["id"], Event(), False)
    assert resumed["status"] == "paused" and not calls["generation"]
    completed = debugger.execute(store, record["id"], Event(), True, paused["context_fingerprint"])
    assert completed["status"] == "completed" and calls["generation"] == 1


def test_historical_replay_and_comparison(setup):
    store, run, record, _, _ = setup
    config = run["configurations"][0]
    run["rows"] = [
        {
            "configuration_id": config["id"],
            "question_index": 0,
            "question": "Question 0?",
            "contexts": [{"text": "Old recorded text", "source": "test.txt", "page": 2}],
            "answer": "Old answer",
        }
    ]
    evidence = debugger.historical(store, run, config["id"], 0)
    assert evidence["candidates"] is None and evidence["prompt_messages"] is None
    store.save("run", run, run["id"])
    replay = debugger.replay(store, evidence)
    assert replay["evidence_kind"] == "replay" and replay["answer"] is None
    a = debugger.execute(store, record["id"], Event())
    b = debugger.execute(store, replay["id"], Event())
    comparison = debugger.compare(a, b)
    assert comparison["chunking_compatible"] and len(comparison["ranks"]) == 3
    b["configuration"]["chunk_size"] = 128
    assert debugger.compare(a, b)["ranks"] is None
    b["document_fingerprint"] = "another corpus"
    with pytest.raises(ValueError):
        debugger.compare(a, b)
    assert debugger.is_stale(
        a, a["question"], a["document_set_id"], {**a["configuration"], "context_k": 2}
    )
    assert not debugger.is_stale(a, a["question"], a["document_set_id"], a["configuration"])
    assert debugger.markers("Evidence [1], unsupported marker [9]", a["selected"]) == [
        {"marker": 1, "valid": True, "passage": 1},
        {"marker": 9, "valid": False, "passage": None},
    ]


def test_document_snapshot_index_isolation(setup, monkeypatch):
    import numpy as np

    from backend import retrieval

    store, _, record, _, _ = setup
    monkeypatch.setattr(
        retrieval,
        "chunk_documents",
        lambda documents, config: [{**d, "token_start": 0, "token_end": 10} for d in documents],
    )

    class Embedder:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(retrieval, "load_embedder", lambda _: Embedder())
    a = store.save(
        "documents", {"documents": [{"source": "a.txt", "page": 1, "text": "Only document A"}]}
    )
    b = store.save(
        "documents", {"documents": [{"source": "b.txt", "page": 1, "text": "Only document B"}]}
    )
    config = {**record["configuration"], "rerank": False}
    ra = debugger.create(store, "Question 0?", a["id"], config)
    rb = debugger.create(store, "Question 0?", b["id"], config)
    # Later upload mutations cannot rewrite the saved corpus snapshot.
    store.save("documents", b, a["id"])
    ra = debugger.execute(store, ra["id"], Event())
    rb = debugger.execute(store, rb["id"], Event())
    assert ra["status"] == rb["status"] == "paused"
    assert ra["selected"][0]["text"] == "Only document A"
    assert rb["selected"][0]["text"] == "Only document B"
    assert ra["index"]["collection"] != rb["index"]["collection"]


def test_api_auth_export_isolation_delete(setup, monkeypatch):
    import backend.main as main
    from tests.test_runs_api import DeferredExecutor

    store, run, record, calls, _ = setup
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "run_lock", Lock())
    monkeypatch.setattr(main, "cancel_events", {})
    executor = DeferredExecutor()
    monkeypatch.setattr(main, "executor", executor)
    monkeypatch.setenv("API_TOKEN", "debug-test")
    client = TestClient(main.app)
    assert client.get("/api/debugger").status_code == 401
    client.headers["Authorization"] = "Bearer debug-test"
    assert client.get("/api/debugger/missing").status_code == 404
    body = {
        "question": "Question 0?",
        "document_set_id": "unknown",
        "configuration": record["configuration"],
    }
    assert client.post("/api/debugger", json=body).status_code == 404
    body["document_set_id"] = run["document_set_id"]
    # No generation key required in retrieval-only mode.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    run["status"] = "completed"
    store.save("run", run, run["id"])
    record["status"] = "failed"
    store.save("debug_run", record, record["id"])
    response = client.post("/api/debugger", json=body)
    assert response.status_code == 202, response.text
    debug_id = response.json()["id"]
    executor.finish()
    result = client.get(f"/api/debugger/{debug_id}/export").json()
    assert result["status"] == "paused" and "inputs" not in result
    assert json.loads(json.dumps(result))["selected"]
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    assert (
        client.post(
            f"/api/debugger/{debug_id}/resume",
            json={"mode": "generate", "context_fingerprint": "wrong"},
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/debugger/{debug_id}/resume",
            json={"mode": "generate", "context_fingerprint": result["context_fingerprint"]},
        ).status_code
        == 202
    )
    executor.finish()
    assert calls["generation"] == 1
    assert client.delete(f"/api/debugger/{debug_id}").status_code == 204
    assert client.get(f"/api/debugger/{debug_id}").status_code == 404
    from backend.graph_pipeline import checkpointer

    with checkpointer(store) as saver:
        assert saver.get_tuple({"configurable": {"thread_id": "debug-" + debug_id}}) is None
