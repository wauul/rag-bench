"""Real LangGraph/Chroma/storage with deterministic HTTP provider boundaries."""

import asyncio
import subprocess
import sys
from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from graph_fakes import install

from backend.models import METRICS, Configuration
from backend.optimization import (
    PlanRequest,
    SearchSpace,
    candidates,
    create_plan,
    execute_experiment,
    prepare_trial,
    select,
    split_questions,
    validate_plan,
)
from backend.optimization_budget import ExperimentBudget, budget_context
from backend.profiling import ProfiledAsyncClient, ProfiledClient
from backend.run_state import RunCancelled
from backend.storage import Store


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for key in ("GROQ_MODEL", "INFERENCE_BACKEND", "GROQ_REQUEST_INTERVAL", "JUDGE_EXAMPLES"):
        monkeypatch.delenv(key, raising=False)
    store = Store(tmp_path)
    docs = store.save(
        "documents",
        {
            "documents": [
                {"text": "Nearest text Second text Best evidence", "source": "test.txt", "page": 2}
            ]
        },
    )
    questions = [
        {"question": f"Question {i}?", "reference": "SECRET_REFERENCE_EVAL_ONLY"} for i in range(6)
    ]
    tests = store.save("test_set", {"name": "Test", "questions": questions})
    baseline = store.save(
        "configuration",
        Configuration(name="Original", engine="langgraph", context_k=1, candidate_k=3).model_dump(),
    )
    request = PlanRequest(
        document_set_id=docs["id"],
        test_set_id=tests["id"],
        baseline_configuration_id=baseline["id"],
        max_trials=4,
        space=SearchSpace(
            chunk_size=[128],
            overlap=[0],
            embedding_model=[baseline["embedding_model"]],
            candidate_k=[3],
            context_k=[1],
            rerank=[False],
        ),
    )
    return store, request


def approve(store, record):
    record.update(approved_at="2026-10-07", status="queued", attempt=record["attempt"] + 1)
    return store.save("optimization", record, record["id"])


def providers(monkeypatch, store, identity, cancel_at=None, fail_metric=None):
    from backend import pipeline

    calls, prompts = install(monkeypatch)
    sent = []

    def transport(request):
        sent.append(request)
        if cancel_at == len(sent):
            record = store.get("optimization", identity)
            store.save("optimization_cancel", {"attempt": record["attempt"]}, identity + "-cancel")
        return httpx.Response(
            200, json={"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
        )

    class LLM:
        def __init__(self):
            self.http_client = ProfiledClient(transport=httpx.MockTransport(transport))
            self.http_async_client = ProfiledAsyncClient(transport=httpx.MockTransport(transport))

        def invoke(self, value):
            assert "SECRET_REFERENCE" not in str(value)
            prompts.append(str(value))
            self.http_client.post("https://provider.test/completions", json={"max_tokens": 2048})
            return SimpleNamespace(content="A saved answer")

    class Metric:
        def __init__(self, llm, name):
            self.llm, self.name = llm, name

        async def single_turn_ascore(self, sample, **kwargs):
            assert sample.reference == "SECRET_REFERENCE_EVAL_ONLY"
            await self.llm.http_async_client.post(
                "https://provider.test/completions", json={"max_tokens": 2048}
            )
            if self.name == fail_metric:
                raise TimeoutError()
            return 0.75

    monkeypatch.setattr(pipeline, "make_llm", LLM)
    monkeypatch.setattr(
        pipeline, "make_metrics", lambda llm, _: {m: Metric(llm, m) for m in METRICS}
    )
    return sent, prompts


def test_candidates_and_split(setup):
    store, request = setup
    baseline = Configuration(engine="langgraph", name="Test").model_dump()
    a = candidates(baseline, request)
    assert a == candidates(baseline, request)
    assert len(a[0]) == 1
    request.space = SearchSpace(chunk_size=[32], overlap=[32], context_k=[8], candidate_k=[1])
    with pytest.raises(ValueError, match="no valid alternative"):
        candidates(baseline, request)
    questions = store.get("test_set", request.test_set_id)["questions"]
    split = split_questions(questions, 42, False)
    assert split == split_questions(questions, 42, False)
    assert set(split["tuning"]).isdisjoint(split["heldout"])
    assert len(split["tuning"]) == 4 and len(split["heldout"]) == 2
    with pytest.raises(ValueError, match="acknowledge"):
        split_questions(questions[:3], 42, False)
    assert split_questions(questions[:3], 42, True)["heldout"] == []
    with pytest.raises(ValueError, match="Duplicate"):
        split_questions([questions[0], questions[0]], 42, True)


@pytest.mark.parametrize(
    "space",
    [
        {"chunk_size": [31]},
        {"overlap": [-1]},
        {"candidate_k": [41]},
        {"context_k": [9]},
        {"embedding_model": ["fake"]},
        {"rerank": [True, True]},
        {"chunk_size": [128, 128]},
    ],
)
def test_invalid_space(space):
    with pytest.raises(ValueError):
        SearchSpace(**space)


def test_plan_fingerprints_and_compatible_reuse(setup, monkeypatch):
    store, request = setup
    record = approve(store, create_plan(store, request))
    validate_plan(record)
    trial = prepare_trial(store, record, "tuning", record["plan"]["baseline"])
    assert prepare_trial(store, record, "tuning", record["plan"]["baseline"])["id"] == trial["id"]
    assert len(record["trials"]) == 1
    trial["questions"][0]["question"] = "Changed question?"
    store.save("run", trial, trial["id"])
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_trial(store, record, "tuning", record["plan"]["baseline"])
    monkeypatch.setenv("JUDGE_EXAMPLES", "1")
    with pytest.raises(ValueError, match="Restore"):
        validate_plan(record)
    record["plan"]["split"]["seed"] += 1
    with pytest.raises(ValueError, match="plan changed"):
        validate_plan(record)


def test_full_graph_no_improvement_and_heldout_selection(setup, monkeypatch):
    store, request = setup
    record = approve(store, create_plan(store, request))
    sent, prompts = providers(monkeypatch, store, record["id"])
    result = execute_experiment(store, record["id"])
    assert result["status"] == "completed", result.get("error")
    assert result["selection"] == {"selected": "baseline", "outcome": "No improvement found"}
    assert len(result["trials"]) == 3  # tie: baseline heldout is evaluated once
    assert len(sent) == 50 and result["usage"]["requests_reserved"] == 50
    assert result["usage"]["total_tokens"] == 400
    assert len(prompts) == 10 and "SECRET_REFERENCE" not in str(prompts)
    tuning = deepcopy(result["tuning_results"])
    result["heldout_results"][0]["quality"] = 0.01
    assert select(tuning, "quality", 0) == result["selection"]
    assert all(r["eligible"] for r in tuning)
    from backend.graph_pipeline import checkpointer

    with checkpointer(store) as saver:
        checkpoint = saver.get_tuple(
            {"configurable": {"thread_id": "optimization-" + record["id"]}}
        )
        assert "SECRET_REFERENCE" not in str(checkpoint)


@pytest.mark.parametrize("stop", ["budget", "cancel"])
def test_stop_resume_keeps_answers_and_budget(setup, monkeypatch, stop):
    store, request = setup
    request.max_requests = 2 if stop == "budget" else 100
    record = approve(store, create_plan(store, request))
    sent, _ = providers(monkeypatch, store, record["id"], cancel_at=2 if stop == "cancel" else None)
    stopped = execute_experiment(store, record["id"])
    assert stopped["status"] == ("budget_exhausted" if stop == "budget" else "cancelled")
    assert len(sent) == stopped["usage"]["requests_reserved"] == 2
    assert stopped["usage"]["total_tokens"] == 16  # in-flight cancellation response counted
    trial = store.get("run", stopped["trials"][0]["run_id"])
    assert trial["rows"][0]["answer"] == "A saved answer"
    assert trial["rows"][0]["scores"][METRICS[0]] == 0.75
    stopped["max_requests"] = 100
    approve(store, stopped)
    providers(monkeypatch, store, record["id"])
    resumed = execute_experiment(store, record["id"])
    assert resumed["status"] == "completed"
    assert resumed["usage"]["requests_reserved"] == 50
    assert resumed["usage"]["total_tokens"] == 400
    assert len({t["run_id"] for t in resumed["trials"]}) == len(resumed["trials"]) == 3
    assert resumed["plan"]["split"] == record["plan"]["split"]


def test_incomplete_metrics_no_eligible_winner(setup, monkeypatch):
    store, request = setup
    record = approve(store, create_plan(store, request))
    providers(monkeypatch, store, record["id"], fail_metric=METRICS[0])
    result = execute_experiment(store, record["id"])
    assert result["selection"]["selected"] is None
    assert not result["heldout_results"]
    assert all(
        not t["eligible"] and t["usage"]["http_requests"] > 0 for t in result["tuning_results"]
    )


def test_latency_threshold_and_selection():
    baseline = {
        "configuration_id": "baseline",
        "eligible": True,
        "quality": 0.7,
        "latency_seconds": 2,
    }
    candidate = {
        "configuration_id": "candidate-1",
        "eligible": True,
        "quality": 0.8,
        "latency_seconds": 1,
    }
    assert select([baseline, candidate], "quality", 0)["selected"] == "candidate-1"
    assert select([baseline, candidate], "latency", 0.75)["selected"] == "candidate-1"
    assert select([baseline, candidate], "latency", 0.9)["selected"] is None
    candidate["eligible"] = False
    assert select([baseline, candidate], "quality", 0)["outcome"] == "No improvement found"


def test_graph_retains_disappointing_holdout_without_reselection(setup, monkeypatch):
    from backend import pipeline

    store, request = setup
    record = approve(store, create_plan(store, request))
    providers(monkeypatch, store, record["id"])
    factory, make_metrics = pipeline.make_llm, pipeline.make_metrics
    created = []

    def llm():
        model = factory()
        model.ordinal = len(created)
        created.append(model)
        return model

    class QualityMetric:
        def __init__(self, original, value):
            self.original, self.value = original, value

        async def single_turn_ascore(self, sample, **kwargs):
            await self.original.single_turn_ascore(sample, **kwargs)
            return self.value

    def metrics(model, embedding):
        value = [0.6, 0.8, 0.9, 0.1][model.ordinal]
        return {k: QualityMetric(m, value) for k, m in make_metrics(model, embedding).items()}

    monkeypatch.setattr(pipeline, "make_llm", llm)
    monkeypatch.setattr(pipeline, "make_metrics", metrics)
    result = execute_experiment(store, record["id"])
    assert result["status"] == "completed"
    assert result["selection"]["selected"] == "candidate-1"
    assert len(result["trials"]) == request.max_trials == 4
    assert result["heldout_results"][0]["quality"] == pytest.approx(0.9)
    assert result["heldout_results"][1]["quality"] == pytest.approx(0.1)
    assert result["usage"]["requests_reserved"] == 60


@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_http_retry_attempts_and_crash_reservations(setup, asynchronous):
    store, request = setup
    request.max_requests = 2
    record = approve(store, create_plan(store, request))
    budget = ExperimentBudget(store, record)
    sent = []

    def transport(request):
        sent.append(request)
        raise httpx.ConnectError("Injected transport failure")

    with budget_context(budget):
        if asynchronous:

            async def run():
                async with ProfiledAsyncClient(transport=httpx.MockTransport(transport)) as client:
                    for _ in range(2):
                        with pytest.raises(httpx.ConnectError):
                            await client.get("https://provider.test")
                    with pytest.raises(RunCancelled):
                        await client.get("https://provider.test")

            asyncio.run(run())
        else:
            with ProfiledClient(transport=httpx.MockTransport(transport)) as client:
                for _ in range(2):
                    with pytest.raises(httpx.ConnectError):
                        client.get("https://provider.test")
                with pytest.raises(RunCancelled):
                    client.get("https://provider.test")
    restored = store.get("optimization", record["id"])
    assert restored["usage"]["requests_reserved"] == len(sent) == 2
    with pytest.raises(RunCancelled):
        ExperimentBudget(store, restored).reserve()


def test_real_sdk_retry_cannot_exceed_budget(setup, monkeypatch):
    from groq import APIConnectionError

    from backend.pipeline import make_llm

    store, request = setup
    request.max_requests = 1
    record = approve(store, create_plan(store, request))
    sent = []

    def transport(request):
        sent.append(request)
        return httpx.Response(
            429,
            json={"error": {"message": "fixture quota", "type": "rate_limit"}},
            headers={"retry-after": "0"},
        )

    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    monkeypatch.setenv("GROQ_REQUEST_INTERVAL", "0.1")
    llm = make_llm()
    llm.http_client.close()
    client = ProfiledClient(transport=httpx.MockTransport(transport))
    llm.client._client._client = client
    try:
        # Groq wraps a denied retry as connection failure; the coordinator's next
        # checkpoint sees the exhausted gate and stops. No second HTTP send occurs.
        with budget_context(ExperimentBudget(store, record)), pytest.raises(APIConnectionError):
            llm.invoke("Test the enforced request cap")
    finally:
        client.close()
        asyncio.run(llm.http_async_client.aclose())
    assert len(sent) == store.get("optimization", record["id"])["usage"]["requests_reserved"] == 1


def test_fresh_process_recovery_preserves_reserved_attempt(setup, monkeypatch):
    store, request = setup
    record = approve(store, create_plan(store, request))
    code = """
import os, sys
sys.path.insert(0, 'tests')
import httpx
from pytest import MonkeyPatch
from test_optimization import providers
from backend.storage import Store
from backend.optimization import execute_experiment
store = Store(sys.argv[1])
providers(MonkeyPatch(), store, sys.argv[2])
def die(*args, **kwargs):
    os._exit(73)
# send() has already durably reserved the HTTP attempt before entering transport.
httpx.MockTransport.handle_request = die
execute_experiment(store, sys.argv[2])
"""
    killed = subprocess.run([sys.executable, "-c", code, str(store.root), record["id"]], timeout=90)
    assert killed.returncode == 73
    recovered = Store(store.root)
    saved = recovered.get("optimization", record["id"])
    assert saved["usage"]["requests_reserved"] == 1
    original_trial = saved["trials"][0]["run_id"]
    approve(recovered, saved)
    providers(monkeypatch, recovered, record["id"])
    result = execute_experiment(recovered, record["id"])
    assert result["status"] == "completed"
    assert result["usage"]["requests_reserved"] == 51  # response lost; reservation retained
    assert result["trials"][0]["run_id"] == original_trial
    assert len(result["trials"]) == 3


def test_api_review_auth_admission_amend_and_save(setup, monkeypatch):
    from test_runs_api import DeferredExecutor

    import backend.main as main

    store, request = setup
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "run_lock", Lock())
    executor = DeferredExecutor()
    monkeypatch.setattr(main, "executor", executor)
    monkeypatch.setenv("API_TOKEN", "test-only")
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    with TestClient(main.app) as client:
        assert client.get("/api/optimizations").status_code == 401
        client.headers["Authorization"] = "Bearer test-only"
        request.max_requests = 2
        created = client.post("/api/optimizations", json=request.model_dump())
        assert created.status_code == 201
        record = created.json()
        assert not executor.jobs and "documents" not in record["plan"]
        prefix = "/api/optimizations/" + record["id"]
        assert client.post(prefix + "/start", json={"plan_fingerprint": "wrong"}).status_code == 409
        approval = {"plan_fingerprint": record["plan_fingerprint"]}
        providers(monkeypatch, store, record["id"])
        assert client.post(prefix + "/start", json=approval).status_code == 202
        assert client.post(prefix + "/start", json=approval).status_code == 409
        executor.finish()
        assert client.post(prefix + "/resume", json=approval).status_code == 409
        assert (
            client.post(
                prefix + "/budget",
                json={"max_requests": 100, "reason": "Finish the held-out evaluation"},
            ).status_code
            == 200
        )
        assert client.post(prefix + "/resume", json=approval).status_code == 202
        executor.finish()
        completed = client.get(prefix).json()
        assert completed["status"] == "completed"
        assert completed["usage"]["requests_reserved"] == 50
        assert len(completed["amendments"]) == 1
        assert client.get(prefix + "/export").json()["selection"]["selected"] == "baseline"
        saved = client.post(
            prefix + "/configuration", json={"configuration_id": "baseline", "name": "Observed"}
        )
        assert saved.status_code == 201 and saved.json()["id"] != request.baseline_configuration_id
        assert store.get("configuration", request.baseline_configuration_id)["name"] == "Original"
        run_id = completed["trials"][0]["run_id"]
        assert client.post(f"/api/runs/{run_id}/retry").status_code == 409
        assert client.delete(f"/api/runs/{run_id}").status_code == 409
        assert client.get("/api/optimizations/inputs").status_code == 200
