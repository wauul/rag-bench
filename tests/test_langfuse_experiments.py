"""Exercise the installed SDK's native experiment APIs without sending user content."""

import json

import httpx

from backend.observability import identity
from backend.storage import Store
from scripts import langfuse_export


def test_native_experiment_payload_is_sanitized_and_links_original_trial(monkeypatch, tmp_path):
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    payloads = []
    timestamp = "2026-10-07T00:00:00Z"

    def transport(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        response = {
            "id": "synthetic-id",
            "name": "synthetic-dataset",
            "projectId": "synthetic-project",
            "createdAt": timestamp,
            "updatedAt": timestamp,
            "datasetId": "synthetic-id",
            "datasetName": "synthetic-dataset",
            "datasetRunId": "synthetic-run",
            "datasetRunName": "synthetic-run",
            "datasetItemId": "synthetic-item",
            "status": "ACTIVE",
            "metadata": {},
            "input": {},
            "expectedOutput": None,
            "mediaReferences": [],
            "traceId": payload.get("traceId"),
        }
        return httpx.Response(200, json=response)

    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    client = Langfuse(
        public_key="pk-synthetic",
        secret_key="sk-synthetic",
        tracer_provider=provider,
        span_exporter=exporter,
        httpx_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    monkeypatch.setattr(langfuse_export, "get_client", lambda: client)
    monkeypatch.setattr(langfuse_export, "enabled", lambda *_: True)
    monkeypatch.setenv("RAGBENCH_LANGFUSE_EXPERIMENTS", "metadata")
    store = Store(tmp_path)
    run = store.save(
        "run",
        {
            "questions": [{"question": "PRIVATE QUESTION", "reference": "PRIVATE REFERENCE"}],
            "input_fingerprint": "a" * 64,
            "status": "completed",
        },
    )
    experiment = store.save(
        "optimization",
        {
            "plan": {"dataset_fingerprint": "b" * 64},
            "trials": [{"run_id": run["id"], "split": "tuning", "configuration_id": "baseline"}],
        },
    )
    result = langfuse_export.export(store, experiment["id"], max_items=1)
    assert result["items"] == 1 and len(payloads) == 3
    assert "PRIVATE" not in json.dumps(payloads)
    assert payloads[-1]["traceId"] == identity("benchmark", run["id"])
    assert payloads[-1]["metadata"]["optimization_id"] == experiment["id"]
    assert payloads[-1]["metadata"]["question_index"] == 0
    assert payloads[-1]["metadata"]["configuration_id"] == "baseline"
    assert payloads[-1]["observationId"]
    assert provider.force_flush(timeout_millis=5000)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = spans[0].attributes
    assert attributes["langfuse.experiment.id"] == "synthetic-run"
    assert (
        attributes["langfuse.experiment.item.root_observation_id"] == payloads[-1]["observationId"]
    )
    assert spans[0].context.trace_id == int(identity("benchmark", run["id"]), 16)
    assert "PRIVATE" not in str(attributes)
