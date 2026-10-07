"""Optional, metadata-only Langfuse destination. Durable records remain authoritative.

No callbacks, automatic instrumentation, content, exception messages or remote prompts.
All application-side operations fail open; exporting uses a bounded background queue.
"""

import functools
import hashlib
import math
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from queue import Full, Queue
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langfuse import Langfuse

_scope: ContextVar[dict[str, Any] | None] = ContextVar("ragbench_observation", default=None)
_client: "Langfuse | None" = None
_lock = Lock()
_scores: Queue[dict[str, Any]] = Queue(maxsize=128)
_score_worker = None
_provider_counter: ContextVar[dict[str, int] | None] = ContextVar(
    "ragbench_provider_counter", default=None
)
PROJECT = "cmuy5n8jc000mad0kjp3qkz0g"
HOSTS = {"https://cloud.langfuse.com", "https://us.cloud.langfuse.com"}
LABELS = {
    "workflow",
    "record_id",
    "attempt",
    "fingerprint",
    "revision",
    "configuration_id",
    "question_index",
    "metric",
    "stage",
    "status",
    "seconds",
    "model",
    "cache",
    "chunks",
    "origin_run_id",
    "optimization_id",
    "split",
    "http_status",
    "provider_attempt",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "missing_usage",
    "reused",
    "expected_questions",
    "valid_count",
    "evaluator",
    "error_count",
}


def enabled(workflow=None):
    return (
        os.getenv("RAGBENCH_LANGFUSE", "false").lower() == "true"
        and os.getenv("LANGFUSE_PROJECT_ID") == PROJECT
        and os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com") in HOSTS
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
        and os.getenv("RAGBENCH_LANGSMITH", "false").lower() != "true"
        and (workflow != "debugger" or os.getenv("RAGBENCH_DEBUGGER_TRACING") == "metadata")
    )


def identity(workflow, record_id):
    return hashlib.sha256(f"ragbench:{workflow}:{record_id}".encode()).hexdigest()[:32]


def metadata(value):
    """Allow structured numbers/identifiers only; never arbitrary text even in labels."""
    result: dict[str, int | float | str] = {}
    for key, item in value.items():
        if key not in LABELS:
            continue
        if isinstance(item, (int, float)) and math.isfinite(item):
            result[key] = item
        elif isinstance(item, str) and re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,160}", item):
            if not any(
                secret and secret in item
                for name, secret in os.environ.items()
                if any(tag in name for tag in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
            ):
                result[key] = item
    return result


def mask(data, **kwargs):
    # The SDK applies this to inputs/outputs/metadata. Content is never supported here.
    return metadata(data) if isinstance(data, dict) else "[redacted]"


def get_client():
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            import httpx
            from langfuse import Langfuse
            from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import SpanLimits, TracerProvider
            from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

            class OneAttemptExporter(SpanExporter):
                def export(self, spans):
                    try:
                        batch = encode_spans(spans)
                        # Final transport boundary strips scope keys, events and IO, including
                        # anything accidentally added outside our custom spans.
                        for resource in batch.resource_spans:
                            for scope in resource.scope_spans:
                                del scope.scope.attributes[:]
                                for span in scope.spans:
                                    attributes = [
                                        a
                                        for a in span.attributes
                                        if a.key
                                        in {
                                            "langfuse.observation.metadata",
                                            "langfuse.observation.type",
                                            "langfuse.environment",
                                            "langfuse.release",
                                        }
                                    ]
                                    del span.attributes[:]
                                    span.attributes.extend(attributes)
                                    del span.events[:]
                                    del span.links[:]
                        payload = batch.SerializeToString()
                        # No environment-configurable endpoint, redirects, SDK or HTTP retries.
                        with httpx.Client(timeout=2, trust_env=False) as transport:
                            response = transport.post(
                                os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
                                + "/api/public/otel/v1/traces",
                                content=payload,
                                auth=(
                                    os.environ["LANGFUSE_PUBLIC_KEY"],
                                    os.environ["LANGFUSE_SECRET_KEY"],
                                ),
                                headers={
                                    "Content-Type": "application/x-protobuf",
                                    "x-langfuse-ingestion-version": "4",
                                },
                            )
                        return (
                            SpanExportResult.SUCCESS
                            if response.is_success
                            else SpanExportResult.FAILURE
                        )
                    except Exception:
                        return SpanExportResult.FAILURE

            provider = TracerProvider(
                resource=Resource({"service.name": "ragbench"}),
                span_limits=SpanLimits(max_attributes=40, max_attribute_length=2048),
                shutdown_on_exit=False,
            )
            _client = Langfuse(
                base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
                timeout=2,
                flush_at=32,
                flush_interval=1,
                tracer_provider=provider,
                span_exporter=OneAttemptExporter(),
                mask=mask,
                environment=os.getenv("APP_ENV", "development"),
                release=os.getenv("APP_REVISION", os.getenv("RENDER_GIT_COMMIT", "local")),
                should_export_span=lambda span: (
                    span.instrumentation_scope is not None
                    and span.instrumentation_scope.name == "langfuse-sdk"
                ),
            )
            import atexit

            # SDK shutdown drains its entire queue. Our bounded, lossy flush takes precedence.
            if _client._resources is not None:
                atexit.unregister(_client._resources.shutdown)
            atexit.register(flush)
    return _client


def start_span(stage, **labels):
    scope = _scope.get()
    if scope is None or not enabled(scope["workflow"]) or "stage" not in metadata({"stage": stage}):
        return None
    try:
        values = metadata({**scope, **labels, "stage": stage})
        client = get_client()
        return client.start_observation(
            name=stage,
            trace_context={"trace_id": scope["trace_id"]},
            metadata=values,
        )
    except Exception:
        return None  # SDK exceptions can contain keys or transport details.


def emit(stage, **labels):
    observation = start_span(stage, **labels)
    if observation is not None:
        try:
            observation.end()
        except Exception:
            pass


@contextmanager
def stage(name, **labels):
    started = time.monotonic()
    status = "completed"
    previous = _scope.get()
    token = _scope.set({**previous, **metadata(labels)}) if previous else None
    observation = start_span(name, **labels)
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        if observation is not None:
            try:
                observation.update(
                    metadata=metadata(
                        {
                            **(_scope.get() or {}),
                            **labels,
                            "stage": name,
                            "status": status,
                            "seconds": round(time.monotonic() - started, 6),
                        }
                    )
                )
                observation.end()
            except Exception:
                pass
        if token is not None:
            _scope.reset(token)


def workflow(kind, storage_kind):
    """Stable trace across attempts; each invocation closes before human review."""

    def decorate(function):
        @functools.wraps(function)
        def call(store, record_id, *args, **kwargs):
            from backend.prompts import _generation, validate

            record = store.get(storage_kind, record_id)
            provenance = record.get("provenance", record.get("plan", {}).get("provenance", {}))
            prompt = provenance.get("generation_prompt")
            if prompt:
                validate(prompt)
            prompt_token = _generation.set(prompt["text"] if prompt else _generation.get())
            try:
                return invoke(store, record_id, record, *args, **kwargs)
            finally:
                _generation.reset(prompt_token)

        def invoke(store, record_id, record, *args, **kwargs):
            if not enabled(kind):
                return function(store, record_id, *args, **kwargs)
            scope = {
                "workflow": kind,
                "record_id": record_id,
                "trace_id": identity(kind, record_id),
                "attempt": (
                    record.get("attempt", 0) + 1
                    if kind == "debugger"
                    else record.get("execution_attempt", 0) + 1
                    if kind == "investigation"
                    else record.get(
                        "attempt", len(record.get("profiling", {}).get("attempts", [])) + 1
                    )
                ),
                "fingerprint": record.get(
                    "input_fingerprint", record.get("fingerprint", record.get("plan_fingerprint"))
                ),
                "revision": os.getenv("APP_REVISION", os.getenv("RENDER_GIT_COMMIT", "local")),
                "origin_run_id": record.get("run_id", (record.get("origin") or {}).get("run_id")),
                "optimization_id": record.get("optimization_id"),
                "configuration_id": record.get("configuration_id"),
                "question_index": record.get("question_index"),
            }
            token = _scope.set(scope)
            counter_token = _provider_counter.set({"count": 0})
            try:
                for name in record.get("completed_stages", []):
                    emit(name, reused=1)
                for row in record.get("rows", []):
                    labels = {
                        "configuration_id": row.get("configuration_id"),
                        "question_index": row.get("question_index"),
                        "reused": 1,
                    }
                    if row.get("answer"):
                        emit("generation", **labels)
                    for metric, value in row.get("scores", {}).items():
                        if value is not None:
                            emit("judging", metric=metric, **labels)
                with stage("execution"):
                    result = function(store, record_id, *args, **kwargs)
                try:
                    final = store.get(storage_kind, record_id)
                    emit("durable_boundary", status=final.get("status"))
                    # Scores describe original benchmark rows only; diagnoses never score.
                    if storage_kind == "run":
                        export_scores(final)
                except Exception:
                    pass
                return result
            finally:
                _provider_counter.reset(counter_token)
                _scope.reset(token)

        return call

    return decorate


def export_scores(run):
    from backend.run_state import valid_score

    for row in run.get("rows", []):
        for metric, value in row.get("scores", {}).items():
            emit(
                "score_coverage",
                configuration_id=row.get("configuration_id"),
                question_index=row.get("question_index"),
                metric=metric,
                valid_count=int(valid_score(metric, value)),
                evaluator="ragas-0.3.9",
                error_count=int(metric in row.get("errors", {})),
            )
            if valid_score(metric, value):
                try:
                    enqueue_score(
                        {
                            "trace_id": identity("benchmark", run["id"]),
                            "name": metric,
                            "value": value,
                            "id": hashlib.sha256(
                                f"{run['id']}:{row['configuration_id']}:{row['question_index']}:{metric}".encode()
                            ).hexdigest(),
                            "metadata": metadata(
                                {
                                    "configuration_id": row["configuration_id"],
                                    "question_index": row["question_index"],
                                    "evaluator": "ragas-0.3.9",
                                }
                            ),
                        }
                    )
                except Exception:
                    pass


def enqueue_score(payload):
    global _score_worker
    from threading import Thread

    with _lock:
        if _score_worker is None:

            def consume():
                while True:
                    item = _scores.get()
                    try:
                        # Public synchronous SDK API, bypassing the SDK's 100k ingestion queue.
                        get_client().api.scores.create(
                            **item, request_options={"timeout_in_seconds": 2, "max_retries": 0}
                        )
                    except Exception:
                        pass
                    finally:
                        _scores.task_done()

            _score_worker = Thread(target=consume, name="ragbench-scores", daemon=True)
            _score_worker.start()
    try:
        _scores.put_nowait(payload)
    except Full:
        pass  # Lossy telemetry; reports can be re-exported from authoritative saved rows.


def begin_provider_attempt():
    counter = _provider_counter.get()
    if counter is None:
        return None
    counter["count"] += 1
    handle = (counter["count"], time.monotonic())
    emit("provider_attempt", provider_attempt=handle[0])
    return handle


def provider_response(response, handle=None):
    """Report numeric usage only; missing usage is unknown, never a synthetic zero."""
    labels = {"http_status": response.status_code}
    if handle:
        labels.update(provider_attempt=handle[0], seconds=time.monotonic() - handle[1])
    try:
        usage = response.json().get("usage", {})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                labels[key] = value
        labels["missing_usage"] = int("total_tokens" not in labels)
    except (ValueError, AttributeError):
        labels["missing_usage"] = 1
    emit("provider_response", **labels)


def trace_link(kind, record_id):
    if not enabled(kind):
        return None
    return (
        os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        + f"/project/{PROJECT}/traces/{identity(kind, record_id)}"
    )


def flush():
    if _client is not None and _client._resources is not None:
        from threading import Thread

        provider = _client._resources.tracer_provider

        def bounded_flush():
            try:
                if provider is not None:
                    provider.force_flush()
            except Exception:
                pass

        # Current OTel force_flush ignores timeout_millis. Bound the caller's wait explicitly.
        worker = Thread(target=bounded_flush, name="ragbench-trace-flush", daemon=True)
        worker.start()
        worker.join(timeout=0.25)
