"""Exercise durable worker checkpoints with deterministic retrieval/LLM boundaries."""
from collections import Counter
from copy import deepcopy
from threading import Event
from types import SimpleNamespace
import pytest
from backend.models import Configuration, METRICS
from backend.settings import load_settings
from backend.storage import Store
from backend.pipeline import execute_run


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    for key in ("GROQ_MODEL", "INFERENCE_BACKEND", "GROQ_REQUEST_INTERVAL", "JUDGE_EXAMPLES"):
        monkeypatch.delenv(key, raising=False)
    store = Store(tmp_path)
    configurations = [{"id": cid, **Configuration(name=cid).model_dump()} for cid in ("a", "b")]
    questions = [{"question": "Question one?", "reference": "Answer one"},
                 {"question": "Question two?", "reference": "Answer two"}]
    run = store.save("run", {"status": "queued", "stage": "Queued", "created_at": "2026-09-30T00:00:00Z",
        "document_set_id": "docs", "questions": questions, "configurations": configurations,
        "total": 4, "completed": 0, "rows": [], "summary": [], "attempt": 1,
        "provenance": load_settings().provenance()})
    return store, run


def fake_dependencies(monkeypatch, store, run_id, event=None):
    import backend.pipeline as pipeline
    calls = Counter()

    def retrieve(store, run_id, config, questions, indexes, checkpoint, stage):
        calls["retrieval:" + config["id"]] += 1
        return {i: [{"text": "Saved evidence", "source": "test.txt", "page": 1,
                     "chunk_id": "0", "distance": 0.1, "token_start": 0, "token_end": 2}] for i in indexes}

    class LLM:
        def invoke(self, prompt):
            if isinstance(prompt, str):
                return SimpleNamespace(content="OK")
            calls["generation"] += 1
            saved = store.get("run", run_id)
            assert saved["rows"] and all(r["contexts"] for r in saved["rows"])
            return SimpleNamespace(content="Generated answer")

    class Metric:
        def __init__(self, name):
            self.name = name

        async def single_turn_ascore(self, sample, **kwargs):
            calls[self.name] += 1
            saved = store.get("run", run_id)
            row = next(r for r in saved["rows"] if r["question"] == sample.user_input
                       and r.get("stage") == "generated") if self.name == METRICS[0] else None
            if row:
                assert row["answer"] == "Generated answer"
            if event and self.name == METRICS[0] and calls[self.name] == 1:
                event.set()
            return 0.8

    monkeypatch.setattr(pipeline, "retrieve_questions", retrieve)
    monkeypatch.setattr(pipeline, "make_llm", LLM)
    monkeypatch.setattr(pipeline, "load_embedder", lambda _: object())
    monkeypatch.setattr(pipeline, "make_metrics", lambda *_: {m: Metric(m) for m in METRICS})
    return calls


def test_cancel_checkpoint_and_resume_without_repeating_answers_or_scores(benchmark, monkeypatch):
    store, run = benchmark
    event = Event()
    calls = fake_dependencies(monkeypatch, store, run["id"], event)
    execute_run(store, run["id"], cancel_event=event)
    cancelled = store.get("run", run["id"])
    assert cancelled["status"] == "cancelled"
    assert len(cancelled["rows"]) == 2  # Retrieval saved before generation, including unanswered rows.
    first = cancelled["rows"][0]
    assert first["answer"] == "Generated answer"
    assert first["scores"]["faithfulness"] == 0.8
    assert first["scores"]["answer_relevancy"] is None
    assert cancelled["completed"] == 0 and cancelled["valid_scores"] == 1
    preserved = deepcopy(first)

    # Retry after restart/cancellation uses stored evidence, fills the missing configuration.
    from scripts.retry_failed import recover
    recover(store, run["id"])
    finished = store.get("run", run["id"])
    assert finished["status"] == "completed"
    assert calls["retrieval:a"] == calls["retrieval:b"] == 1
    assert calls["generation"] == 4
    assert all(calls[m] == 4 for m in METRICS)
    assert finished["rows"][0]["contexts"] == preserved["contexts"]
    assert finished["retry_history"][0]["previous_row"] == preserved
    assert finished["completed"] == finished["scored"] == 4 and finished["valid_scores"] == 16


def test_cancel_queued_run_makes_no_model_or_provider_calls(benchmark, monkeypatch):
    store, run = benchmark
    import backend.pipeline as pipeline
    monkeypatch.setattr(pipeline, "make_llm", lambda: pytest.fail("Queued cancellation called provider"))
    event = Event()
    event.set()
    execute_run(store, run["id"], cancel_event=event)
    assert store.get("run", run["id"])["status"] == "cancelled"


def test_generation_failure_retries_only_that_answer(benchmark, monkeypatch):
    store, run = benchmark
    calls = fake_dependencies(monkeypatch, store, run["id"])
    import backend.pipeline as pipeline
    generate = pipeline.generate_answer
    failed = False

    def once(llm, row):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("Quota exhausted")
        return generate(llm, row)

    monkeypatch.setattr(pipeline, "generate_answer", once)
    execute_run(store, run["id"])
    partial = store.get("run", run["id"])
    assert partial["status"] == "partial" and partial["completed"] == 4 and partial["scored"] == 3
    good = deepcopy(partial["rows"][1:])
    from scripts.retry_failed import recover
    recover(store, run["id"])
    finished = store.get("run", run["id"])
    assert finished["status"] == "completed" and finished["rows"][1:] == good
    assert calls["generation"] == 4 and all(calls[m] == 4 for m in METRICS)


def test_restart_after_last_metric_finishes_from_saved_scores(benchmark, monkeypatch):
    store, run = benchmark
    fake_dependencies(monkeypatch, store, run["id"])
    execute_run(store, run["id"])
    saved = store.get("run", run["id"])
    saved.update(status="failed", error="Interrupted by server restart")
    saved["rows"][-1].update(stage="scoring", processed=False)
    original_rows = deepcopy(saved["rows"])
    store.save("run", saved, run["id"])
    import backend.pipeline as pipeline
    monkeypatch.setattr(pipeline, "make_llm", lambda: pytest.fail("No provider work was missing"))
    from scripts.retry_failed import recover
    recover(store, run["id"])
    finished = store.get("run", run["id"])
    assert finished["status"] == "completed" and finished["completed"] == 4
    assert finished["rows"] == original_rows and "error" not in finished
