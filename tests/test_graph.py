"""Real StateGraph + SQLite + Chroma, fake external providers and small local vectors."""

import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from threading import Event

import pytest
from graph_fakes import create_run, install

from backend.models import METRICS, Configuration
from backend.pipeline import execute_run
from backend.provenance import validate_snapshot
from backend.storage import Store


@pytest.fixture(autouse=True)
def stable_settings(monkeypatch):
    for key in ("GROQ_MODEL", "INFERENCE_BACKEND", "GROQ_REQUEST_INTERVAL", "JUDGE_EXAMPLES"):
        monkeypatch.delenv(key, raising=False)


def test_engines_retrieval_prompt_parity_and_provenance(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run = create_run(store)
    calls, prompts = install(monkeypatch)
    result = execute_run(store, run["id"])
    assert result["status"] == "completed", result.get("error")
    a, b = result["rows"]
    assert a["contexts"] == b["contexts"]
    assert a["contexts"][0]["text"] == "Best evidence"
    from backend.retrieval_trace import load_trace, trace_summary

    for config, row in zip(result["configurations"], result["rows"]):
        trace = load_trace(store, result, config, 0, row["contexts"])
        assert trace["rerank_applied"]
        assert trace["candidates"][0]["text"] != "Best evidence"
        assert trace["ordered_candidates"][0]["text"] == "Best evidence"
        assert all("rerank_score" not in c for c in trace["candidates"])
        assert sum(p["supplied_to_answer"] for p in trace_summary(trace)["passages"]) == 1
    assert prompts[0] == prompts[1]
    assert calls == {"generation": 2, "rerank": 2, **dict.fromkeys(METRICS, 2)}
    assert result["answered"] == result["scored"] == 2
    assert result["pending_answers"] == result["pending_metrics"] == 0
    assert b["engine"] == "langgraph" and b["graph_thread_id"] != run["id"]
    from importlib.metadata import version

    assert result["provenance"]["dependencies"]["langgraph"] == version("langgraph")
    from backend.graph_pipeline import checkpointer

    with checkpointer(store) as saver:
        saved = list(saver.list({"configurable": {"thread_id": b["graph_thread_id"]}}))
        stages = {s.checkpoint["channel_values"].get("stage") for s in saved}
        assert {
            "prepare_index",
            "retrieve",
            "rerank",
            "generate",
            "finalize",
            *["evaluate_" + m for m in METRICS],
        } <= stages
        assert "SECRET_REFERENCE" not in str(saved) and "A saved answer" not in str(saved)


@pytest.mark.parametrize("rerank", [True, False])
def test_cancel_and_retry_keep_individual_commits_and_index(tmp_path, monkeypatch, rerank):
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",), rerank=rerank, questions=2)
    event = Event()
    calls, _ = install(monkeypatch, event=event)
    cancelled = execute_run(store, run["id"], cancel_event=event)
    assert cancelled["status"] == "cancelled"
    assert cancelled["valid_scores"] == 1 and cancelled["answered"] == 1
    row = deepcopy(cancelled["rows"][0])
    event.clear()
    from scripts.retry_failed import recover

    recover(store, run["id"])
    result = store.get("run", run["id"])
    assert result["status"] == "completed"
    assert calls["generation"] == 2 and all(calls[m] == 2 for m in METRICS)
    assert result["rows"][0]["contexts"] == row["contexts"]
    assert result["rows"][0]["answer"] == row["answer"]
    from backend.profiling_report import spans

    phases = list(spans(result["profiling"]))
    assert len([p for p in phases if p["stage"] == "indexing"]) == 1
    assert len([p for p in phases if p["stage"] == "index_reuse"]) == 1


def test_partial_scoring_only_missing_metric_retried(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    install(monkeypatch, fail_metric="context_precision")
    partial = execute_run(store, run["id"])
    assert partial["status"] == "partial" and partial["pending_metrics"] == 1
    assert partial["rows"][0]["failure_kinds"]["context_precision"] == "retryable_provider"
    calls, _ = install(monkeypatch)
    from scripts.retry_failed import recover

    recover(store, run["id"])
    assert calls == {"context_precision": 1}
    assert store.get("run", run["id"])["status"] == "completed"


@pytest.mark.parametrize("mutation", ["question", "config", "version", "snapshot", "model"])
def test_stale_inputs_rejected_before_provider(tmp_path, monkeypatch, mutation):
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    calls, _ = install(monkeypatch)
    if mutation == "question":
        run["questions"][0]["reference"] = "changed"
    elif mutation == "config":
        run["configurations"][0]["context_k"] = 2
    elif mutation == "version":
        run["provenance"]["graph_version"] = "future"
    elif mutation == "model":
        monkeypatch.setenv("GROQ_MODEL", "different")
    else:
        snap = store.get("snapshot", run["id"] + "-snapshot")
        snap["inputs"]["documents"][0]["text"] = "changed"
        store.save("snapshot", snap, snap["id"])
    store.save("run", run, run["id"])
    result = execute_run(store, run["id"])
    assert result["status"] == "failed" and not calls


def test_source_edits_cannot_change_snapshotted_documents(tmp_path):
    store = Store(tmp_path)
    run = create_run(store)
    store.save("documents", {"documents": []}, run["document_set_id"])
    validate_snapshot(store, run)
    from backend.provenance import run_documents

    assert run_documents(store, run)[0]["text"]


def test_duplicate_execution_rejected_in_fresh_process(tmp_path):
    from backend.execution_lock import execution_lock

    store = Store(tmp_path)
    with execution_lock(store):
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "from backend.storage import Store; from backend.execution_lock import execution_lock; "
                f"\nwith execution_lock(Store({str(tmp_path)!r})): print('acquired')",
            ],
            capture_output=True,
            text=True,
        )
    assert process.returncode != 0 and "ExecutionBusy" in process.stderr


@pytest.mark.parametrize(
    "crash,expected_generations",
    [
        ("before_call", 1),
        ("before_answer_save", 2),
        ("after_answer_save", 1),
        ("between_metrics", 1),
        ("retrieval", 1),
        ("indexing", 1),
        ("initial_checkpoint", 1),
    ],
)
def test_process_death_and_recovery(tmp_path, crash, expected_generations):
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    log = tmp_path / "calls.txt"
    script = (
        "import sys; sys.path.insert(0, 'tests'); from pytest import MonkeyPatch; "
        "from graph_fakes import install; from backend.storage import Store; from backend.pipeline import execute_run; "
        f"install(MonkeyPatch(), log_path={str(log)!r}, crash=CRASH); "
        f"execute_run(Store({str(tmp_path)!r}), {run['id']!r}, retry=True)"
    )
    first = subprocess.run(
        [sys.executable, "-c", script.replace("CRASH", repr(crash))],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert first.returncode == 73, first.stderr
    assert (tmp_path / "langgraph.sqlite3").exists()
    # A fresh interpreter recreates all clients, graph, sqlite connections and model fakes.
    second = subprocess.run(
        [sys.executable, "-c", script.replace("CRASH", "None")],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert second.returncode == 0, second.stderr
    result = store.get("run", run["id"])
    assert result["status"] == "completed", result.get("error")
    calls = Counter(log.read_text().splitlines())
    assert calls["generation"] == expected_generations
    assert all(calls[m] == 1 for m in METRICS)


def test_checkpoint_ahead_of_app_reconciles_missing_results(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    install(monkeypatch)
    result = execute_run(store, run["id"])
    result["rows"][0]["scores"]["context_recall"] = None
    result["status"] = "partial"
    store.save("run", result, run["id"])
    calls, _ = install(monkeypatch)
    from scripts.retry_failed import recover

    recover(store, run["id"])
    assert calls == {"context_recall": 1}


def test_delete_cleans_graph_and_chroma_without_other_runs(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    install(monkeypatch)
    result = execute_run(store, run["id"])
    other = create_run(store, ("langgraph",))
    from backend.execution_lock import execution_lock

    with execution_lock(store):
        store.delete_run(run["id"])
    assert store.get("run", other["id"])
    assert not store.list("graph_artifact")
    from backend.graph_pipeline import checkpointer

    with checkpointer(store) as saver:
        assert (
            saver.get_tuple({"configurable": {"thread_id": result["rows"][0]["graph_thread_id"]}})
            is None
        )
    with pytest.raises(KeyError):
        store.get("run", run["id"])


def test_legacy_and_new_configuration_defaults():
    assert Configuration(name="Legacy", top_k=2).engine == "existing"
    assert Configuration(name="Graph", engine="langgraph").engine == "langgraph"
    with pytest.raises(ValueError):
        Configuration(name="Unknown", engine="agent")


def test_stale_graph_checkpoint_is_rejected(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    install(monkeypatch, fail_metric="context_recall")
    execute_run(store, run["id"])
    import backend.graph_pipeline as graph

    monkeypatch.setattr(graph, "GRAPH_VERSION", "incompatible")
    calls, _ = install(monkeypatch)
    result = execute_run(store, run["id"], retry=True)
    assert result["status"] == "failed" and not calls


@pytest.mark.parametrize("answer", ["", "   ", ["non-text"]])
def test_empty_answers_are_not_scored(tmp_path, monkeypatch, answer):
    from types import SimpleNamespace

    from backend import pipeline

    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    calls, _ = install(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "make_llm",
        lambda: SimpleNamespace(invoke=lambda _: SimpleNamespace(content=answer)),
    )
    result = execute_run(store, run["id"])
    assert result["status"] == "partial" and result["answered"] == 0
    assert result["pending_metrics"] == 4 and not any(calls[m] for m in METRICS)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.1, True])
def test_invalid_scores_are_missing(value):
    import asyncio

    from backend.pipeline import score_row

    class Metric:
        async def single_turn_ascore(self, *args, **kwargs):
            return value

    row = {"question": "Question?", "answer": "Answer", "reference": "Reference", "contexts": []}
    scores, errors = asyncio.run(score_row({"faithfulness": Metric()}, row))
    assert scores == {"faithfulness": None} and errors


def test_valid_values_override_stale_errors_without_provider_calls(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run = create_run(store)
    install(monkeypatch)
    result = execute_run(store, run["id"])
    result["status"] = "partial"
    for row in result["rows"]:
        row["errors"] = {"generation": "stale", "context_recall": "stale"}
    store.save("run", result, run["id"])
    calls, _ = install(monkeypatch)
    result = execute_run(store, run["id"], retry=True)
    assert result["status"] == "completed" and not calls


def test_missing_answer_invalidates_orphaned_scores(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    install(monkeypatch)
    result = execute_run(store, run["id"])
    result["status"] = "partial"
    result["rows"][0]["answer"] = "   "
    store.save("run", result, run["id"])
    calls, _ = install(monkeypatch)
    result = execute_run(store, run["id"], retry=True)
    assert result["status"] == "completed" and calls == {
        "generation": 1,
        **dict.fromkeys(METRICS, 1),
    }


def test_actual_chatgroq_and_ragas_with_controlled_http(tmp_path, monkeypatch):
    """Real adapters, prompts, parsers, algorithms, SDK, graph and profiling; no network."""
    import httpx

    from backend import pipeline
    from backend.profiling import ProfiledAsyncClient, ProfiledClient

    make_llm, make_metrics = pipeline.make_llm, pipeline.make_metrics
    install(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    monkeypatch.setenv("GROQ_REQUEST_INTERVAL", "0.1")
    requests = []

    def provider(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body.get("n", 1) == 1
        prompt = body["messages"][0]["content"]
        if prompt.startswith("Answer only"):
            assert "SECRET_REFERENCE" not in str(body)
            content = "A saved answer"
        elif "Break down each sentence" in prompt:
            content = json.dumps({"statements": ["A saved answer"]})
        elif "noncommittal" in prompt:
            content = json.dumps({"question": "Question 0?", "noncommittal": 0})
        elif "classifications" in prompt:
            content = json.dumps(
                {
                    "classifications": [
                        {"statement": "A saved answer", "reason": "fixture", "attributed": 1}
                    ]
                }
            )
        elif "statements" in prompt:
            content = json.dumps(
                {"statements": [{"statement": "A saved answer", "reason": "fixture", "verdict": 1}]}
            )
        else:
            content = json.dumps({"reason": "fixture", "verdict": 1})
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    class Client(ProfiledClient):
        def __init__(self):
            super().__init__(transport=httpx.MockTransport(provider))

    class AsyncClient(ProfiledAsyncClient):
        def __init__(self):
            super().__init__(transport=httpx.MockTransport(provider))

    monkeypatch.setattr(pipeline, "ProfiledClient", Client)
    monkeypatch.setattr(pipeline, "ProfiledAsyncClient", AsyncClient)
    monkeypatch.setattr(pipeline, "make_llm", make_llm)
    monkeypatch.setattr(pipeline, "make_metrics", make_metrics)
    store = Store(tmp_path)
    run = create_run(store)
    result = execute_run(store, run["id"])
    assert result["status"] == "completed", [r["errors"] for r in result["rows"]]
    assert (
        len(requests) == 16
    )  # Each engine: generation + 2 faithfulness + 3 relevancy + precision + recall.
    assert result["profiling"]["summary"]["groq"]["http_requests"] == 16
    assert result["rows"][0]["scores"] == result["rows"][1]["scores"]


@pytest.mark.parametrize("failure", [429, 500, "timeout"])
def test_provider_retry_budget_is_two_total_attempts(tmp_path, monkeypatch, failure):
    import httpx

    from backend import pipeline
    from backend.profiling import ProfiledClient

    make_llm = pipeline.make_llm
    install(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    monkeypatch.setenv("GROQ_REQUEST_INTERVAL", "0.1")
    requests = []

    def provider(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("injected", request=request)
        return httpx.Response(failure, json={"error": {"message": "injected", "type": "fixture"}})

    class Client(ProfiledClient):
        def __init__(self):
            super().__init__(transport=httpx.MockTransport(provider))

    monkeypatch.setattr(pipeline, "ProfiledClient", Client)
    monkeypatch.setattr(pipeline, "make_llm", make_llm)
    store = Store(tmp_path)
    run = create_run(store, ("langgraph",))
    result = execute_run(store, run["id"])
    assert result["status"] == "partial" and result["answered"] == 0
    assert len(requests) == result["profiling"]["summary"]["groq"]["http_requests"] == 2
    assert result["rows"][0]["failure_kinds"]["generation"] == "retryable_provider"
