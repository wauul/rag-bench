"""Sensitive fixtures exercise actual SDK span payloads, not only the masking helper."""

import json
from types import SimpleNamespace

import httpx
import pytest

from backend import observability as obs
from backend.storage import Store


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("RAGBENCH_LANGFUSE", "true")
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", obs.PROJECT)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-synthetic-export-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-sensitive-fixture-secret")
    monkeypatch.setenv("RAGBENCH_LANGSMITH", "false")
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)


def test_disabled_and_debugger_separate_opt_in(monkeypatch, enabled):
    assert obs.enabled("benchmark") and not obs.enabled("debugger")
    monkeypatch.setenv("RAGBENCH_DEBUGGER_TRACING", "metadata")
    assert obs.enabled("debugger")
    monkeypatch.setenv("RAGBENCH_LANGSMITH", "true")
    assert not obs.enabled("benchmark")
    monkeypatch.setenv("RAGBENCH_LANGFUSE", "false")
    monkeypatch.setattr(obs, "get_client", lambda: pytest.fail("disabled tracing initialized SDK"))
    obs.emit("generation")


def test_sdk_payload_is_metadata_only_and_correlated(enabled, monkeypatch, tmp_path):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    payloads = []
    real_client = httpx.Client

    def transport(request):
        payloads.append(request.content)
        return httpx.Response(200)

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(transport), **kw)
    )
    monkeypatch.setattr(obs, "_client", None)
    store = Store(tmp_path)
    run = store.save("run", {"attempt": 1, "question": "PRIVATE QUESTION", "status": "paused"})

    @obs.workflow("benchmark", "run")
    def execute(store, identity):
        obs.emit(
            "retrieval",
            question="PRIVATE QUESTION",
            prompt="PRIVATE PROMPT",
            answer="PRIVATE ANSWER",
            model="sk-sensitive-fixture-secret",
            question_index=0,
        )
        obs.provider_response(
            httpx.Response(200, json={"usage": {"total_tokens": 12}, "answer": "PRIVATE ANSWER"})
        )
        return store.get("run", identity)

    execute(store, run["id"])
    client = obs.get_client()
    assert client._resources.tracer_provider.force_flush(timeout_millis=5000)
    assert payloads
    joined = b"".join(payloads)
    for private in (
        b"PRIVATE QUESTION",
        b"PRIVATE PROMPT",
        b"PRIVATE ANSWER",
        b"sk-sensitive-fixture-secret",
        b"pk-synthetic-export-test",
    ):
        assert private not in joined
    traces = []
    attributes = {}
    for payload in payloads:
        batch = ExportTraceServiceRequest.FromString(payload)
        for resource in batch.resource_spans:
            for scope in resource.scope_spans:
                for span in scope.spans:
                    attributes.update({a.key: a.value for a in span.attributes})
        traces.extend(
            span.trace_id.hex()
            for resource in batch.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        )
    assert set(traces) == {obs.identity("benchmark", run["id"])}
    assert attributes["langfuse.observation.metadata.workflow"].string_value == "benchmark"
    assert attributes["langfuse.observation.metadata.record_id"].string_value == run["id"]
    assert attributes["langfuse.observation.metadata.question_index"].int_value == 0
    assert attributes["langfuse.internal.as_root"].bool_value is True
    assert "langfuse.observation.metadata.question" not in attributes
    monkeypatch.setattr(obs, "_client", None)


def test_export_outage_does_not_change_result(enabled, monkeypatch, tmp_path):
    monkeypatch.setattr(
        obs, "get_client", lambda: (_ for _ in ()).throw(RuntimeError("private transport error"))
    )
    store = Store(tmp_path)
    run = store.save("run", {"attempt": 1, "status": "completed"})

    @obs.workflow("benchmark", "run")
    def execute(store, identity):
        with obs.stage("generation"):
            return 42

    assert execute(store, run["id"]) == 42
    assert obs._scope.get() is None


def test_score_association_missing_values_and_retry_identity(enabled, monkeypatch):
    payloads = []
    monkeypatch.setattr(obs, "enqueue_score", payloads.append)
    run = {
        "id": "trial-id",
        "rows": [
            {
                "configuration_id": "cfg",
                "question_index": 0,
                "scores": {"faithfulness": 0.7, "context_recall": None},
                "errors": {"context_recall": "timeout"},
            }
        ],
    }
    obs.export_scores(run)
    obs.export_scores(run)
    assert len(payloads) == 2 and payloads[0] == payloads[1]
    assert payloads[0]["trace_id"] == obs.identity("benchmark", "trial-id")
    assert payloads[0]["metadata"]["question_index"] == 0
    assert "timeout" not in json.dumps(payloads)


def test_mask_rejects_content_and_secrets(enabled):
    assert obs.mask(
        {"question": "private", "model": "sk-sensitive-fixture-secret", "attempt": 2}
    ) == {"attempt": 2}
    assert obs.mask("private") == "[redacted]"


def test_numeric_usage_missing_is_unknown(monkeypatch):
    calls = []
    monkeypatch.setattr(obs, "emit", lambda name, **values: calls.append((name, values)))
    obs.provider_response(httpx.Response(200, json={"answer": "secret"}))
    assert calls[0][1] == {"http_status": 200, "missing_usage": 1}
    obs.provider_response(httpx.Response(429, json={"usage": {"total_tokens": -1}}))
    assert "total_tokens" not in calls[-1][1]


def test_score_queue_is_bounded(monkeypatch):
    from queue import Queue

    monkeypatch.setattr(obs, "_scores", Queue(maxsize=1))
    monkeypatch.setattr(obs, "_score_worker", SimpleNamespace())
    obs.enqueue_score({"id": "one"})
    obs.enqueue_score({"id": "two"})
    assert obs._scores.qsize() == 1
