"""Verify accounting against controlled HTTP responses; no real provider calls or scores."""
import asyncio
import json
from contextvars import copy_context
from copy import deepcopy
import httpx
import pytest
from backend.profiling import ProfiledClient, ProfiledAsyncClient, RunProfiler, measure
from backend.profiling_report import interrupt_profile, profile_csv_rows, row_profile


def run_record():
    return {"rows": [], "attempt": 1, "provenance": {"inference_backend": "test fixture"}}


def completion(usage=True):
    payload = {"id": "test", "model": "fixture", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "A controlled response"}, "finish_reason": "stop"}]}
    if usage:
        payload["usage"] = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            "completion_tokens_details": {"reasoning_tokens": 2}, "prompt_tokens_details": {"cached_tokens": 3},
            "total_time": 0.25}
    return payload


def test_sampled_peak_timings_and_model_delta_are_preserved():
    clock, rss = [0.0], [100.0]
    run = run_record()
    with RunProfiler(run, clock=lambda: clock[0], rss_reader=lambda: rss[0]) as profiler:
        with measure("embedding_model_load", configuration_id="a", model="fixture"):
            clock[0] = 2.0
            rss[0] = 600
            profiler.sample_memory()
            rss[0] = 120
            clock[0] = 3.0
        profiler.status = "completed"
    profile = run["profiling"]
    entry = profile["attempts"][0]["spans"][0]
    assert entry["seconds"] == 3 and entry["rss_delta_mb"] == 20
    assert entry["rss_peak_mb"] == profile["summary"]["rss_peak_mb"] == 600
    assert profile["attempts"][0]["status"] == "completed"
    assert profile["summary"]["worker_seconds"] == 3
    assert not profiler._thread.is_alive()


def test_groq_sdk_retries_count_each_http_attempt_and_reported_usage(monkeypatch):
    from backend.pipeline import make_llm
    import backend.pipeline as pipeline
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    monkeypatch.setenv("GROQ_REQUEST_INTERVAL", "0.1")
    calls = []

    def provider(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "Fixture rate limit"}})
        return httpx.Response(200, json=completion())

    client = ProfiledClient(transport=httpx.MockTransport(provider))
    monkeypatch.setattr(pipeline, "ProfiledClient", lambda: client)
    llm = make_llm()
    run = run_record()
    try:
        with RunProfiler(run) as profiler:
            with measure("generation", configuration_id="a", question_index=0):
                assert llm.invoke("Private prompt must not be persisted").content == "A controlled response"
            profiler.status = "completed"
    finally:
        client.close()
        asyncio.run(llm.http_async_client.aclose())
    stats = run["profiling"]["summary"]["groq"]
    assert stats["http_requests"] == 2 and stats["http_successes"] == stats["http_failures"] == 1
    assert stats["rate_limit_responses"] == 1 and stats["usage_reports"] == 1
    assert stats["prompt_tokens"] == 10 and stats["completion_tokens"] == 4 and stats["total_tokens"] == 14
    assert stats["reasoning_tokens"] == 2 and stats["cached_tokens"] == 3
    serialized = json.dumps(run["profiling"], allow_nan=False)
    assert "Private prompt" not in serialized and "test-only" not in serialized and "controlled response" not in serialized


def test_async_judge_multiple_completions_and_row_context_are_not_double_counted(monkeypatch):
    from backend.pipeline import make_llm
    from backend.groq_judge import GroqRagasLLM
    from langchain_core.prompt_values import StringPromptValue
    import backend.pipeline as pipeline
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    monkeypatch.setenv("GROQ_REQUEST_INTERVAL", "0.1")
    async_client = ProfiledAsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion())))
    monkeypatch.setattr(pipeline, "ProfiledAsyncClient", lambda: async_client)
    llm = make_llm()
    judge = GroqRagasLLM(llm, bypass_n=True)
    run = run_record()

    async def score(index):
        with measure("scoring", configuration_id="a", question_index=index, metric="answer_relevancy"):
            return await judge.agenerate_text(StringPromptValue(text="Fixture question"), n=3)

    # Runner reuse previously captured the first row's context. Each run must use the current context.
    with asyncio.Runner() as runner, RunProfiler(run) as profiler:
        for index in (0, 1):
            assert len(runner.run(score(index), context=copy_context()).generations[0]) == 3
        runner.run(async_client.aclose())
        profiler.status = "completed"
    llm.http_client.close()
    profile = run["profiling"]
    assert profile["summary"]["groq"]["http_requests"] == 6
    for index in (0, 1):
        stats = row_profile(profile, "a", index)
        assert stats["groq"]["total_tokens"] == 42
        assert stats["metrics"]["answer_relevancy"]["groq"]["http_requests"] == 3


def test_missing_usage_transport_errors_and_zero_tokens_are_distinguished():
    responses = [httpx.Response(200, json=completion(False))]
    zero = completion()
    zero["usage"].update(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    responses.append(httpx.Response(200, json=zero))

    def provider(request):
        if responses:
            return responses.pop(0)
        raise httpx.ConnectError("Fixture transport failure", request=request)

    run = run_record()
    with ProfiledClient(transport=httpx.MockTransport(provider)) as client, RunProfiler(run) as profiler:
        for index in (0, 1, 2):
            try:
                with measure("generation", configuration_id="a", question_index=index):
                    client.get("https://fixture.invalid")
            except httpx.ConnectError:
                pass
        profiler.status = "partial"
    profile = run["profiling"]
    stats = profile["summary"]["groq"]
    assert stats["http_requests"] == 3 and stats["transport_errors"] == 1
    assert stats["usage_reports"] == stats["missing_usage_reports"] == 1
    rows = profile_csv_rows(profile)
    assert rows[0]["groq_total_tokens"] is None and rows[1]["groq_total_tokens"] == 0
    assert rows[2]["status"] == "failed" and rows[2]["groq_total_tokens"] is None


def test_retry_appends_measurements_and_marks_crash_data_incomplete():
    run = run_record()
    with RunProfiler(run) as profiler:
        with measure("indexing", configuration_id="a"):
            pass
        profiler.status = "cancelled"
    original = deepcopy(run["profiling"]["attempts"][0])
    run["attempt"] = 2
    with RunProfiler(run) as profiler:
        with measure("scoring", configuration_id="a", question_index=0, metric="faithfulness"):
            pass
        profiler.status = "completed"
    assert run["profiling"]["attempts"][0] == original
    assert [a["number"] for a in run["profiling"]["attempts"]] == [1, 2]
    # Simulate a process kill before a response/phase finish was checkpointed.
    attempt = run["profiling"]["attempts"][-1]
    attempt.update(status="running", finished_at=None)
    entry = attempt["spans"][-1]
    entry.update(status="running", finished_at=None)
    preserved_seconds = entry["seconds"]
    interrupt_profile(run)
    assert attempt["status"] == entry["status"] == "interrupted"
    assert entry["seconds"] == preserved_seconds and entry["finished_at"] is None


def test_unavailable_memory_and_legacy_runs_do_not_fabricate_measurements():
    run = run_record()
    run["rows"] = [{"answer": "Legacy answer"}]
    with RunProfiler(run, rss_reader=lambda: None) as profiler:
        with measure("scoring", configuration_id="a", question_index=0, metric="faithfulness"):
            pass
        profiler.status = "completed"
    assert run["profiling"]["coverage"] == "since_enabled"
    assert run["profiling"]["summary"]["rss_peak_mb"] is None
    assert row_profile({}, "a", 0) is None
