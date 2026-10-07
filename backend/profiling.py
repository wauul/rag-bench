"""Checkpointed phase timings, sampled process RSS, and Groq HTTP-attempt accounting.

Only numeric provider usage is retained. Prompts, answers, headers and keys are never
copied into the profile. Context variables attribute async judge calls to their row.
"""

import math
import platform
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import Event, Lock, Thread

import httpx
import psutil

from backend.profiling_report import empty_usage, interrupt_profile, summarize_profile

_current = ContextVar("rag_bench_profiler", default=None)
_span = ContextVar("rag_bench_profile_span", default=None)
SAMPLE_INTERVAL = 0.1


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def process_rss_mb():
    try:
        return psutil.Process().memory_info().rss / (1024**2)
    except (psutil.Error, OSError):
        return None


class RunProfiler:
    def __init__(self, run, on_update=None, *, clock=time.monotonic, rss_reader=process_rss_mb):
        interrupt_profile(run)
        self.profile = run.setdefault(
            "profiling",
            {
                "version": 1,
                "coverage": "since_enabled" if run["rows"] or run.get("attempt", 1) > 1 else "full",
                "memory_kind": "backend_process_rss",
                "memory_unit": "MiB",
                "sample_interval_seconds": SAMPLE_INTERVAL,
                "attempts": [],
            },
        )
        self.clock, self.rss_reader, self.on_update = clock, rss_reader, on_update
        self.attempt = {
            "number": run.get("attempt", 1),
            "started_at": timestamp(),
            "finished_at": None,
            "status": "running",
            "seconds": 0,
            "rss_start_mb": None,
            "rss_end_mb": None,
            "rss_peak_mb": None,
            "memory_samples": 0,
            "spans": [],
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "logical_cpus": psutil.cpu_count(),
                "inference_backend": run.get("provenance", {}).get("inference_backend"),
            },
        }
        self.profile["attempts"].append(self.attempt)
        self._lock, self._stop = Lock(), Event()
        self._active, self._peak, self._span_peak = None, None, None
        self._samples = 0

    def __enter__(self):
        self.started = self.clock()
        self._token = _current.set(self)
        self.attempt["rss_start_mb"] = self.sample_memory()
        self._thread = Thread(target=self._sample_loop, name="rag-bench-memory", daemon=True)
        self._thread.start()
        try:
            self.emit()
        except BaseException:
            self._stop.set()
            self._thread.join(timeout=1)
            _current.reset(self._token)
            raise
        return self

    def _sample_loop(self):
        while not self._stop.wait(SAMPLE_INTERVAL):
            self.sample_memory()

    def sample_memory(self):
        value = self.rss_reader()
        if value is None or not math.isfinite(value):
            return None
        with self._lock:
            self._samples += 1
            self._peak = max(self._peak or value, value)
            if self._active is not None:
                self._span_peak = max(self._span_peak or value, value)
        return round(value, 3)

    def snapshot(self):
        self.attempt["seconds"] = round(self.clock() - self.started, 6)
        with self._lock:
            self.attempt.update(
                rss_peak_mb=round(self._peak, 3) if self._peak is not None else None,
                memory_samples=self._samples,
            )
            if self._active is not None:
                self._active["rss_peak_mb"] = (
                    round(self._span_peak, 3) if self._span_peak is not None else None
                )
                self._active["seconds"] = round(self.clock() - self._active_started, 6)
        self.profile["summary"] = summarize_profile(self.profile)

    def emit(self):
        self.snapshot()
        if self.on_update:
            self.on_update()

    @contextmanager
    def phase(self, stage, **labels):
        # Operations are serial; no nesting prevents double-counting time and usage.
        if _span.get() is not None:
            raise RuntimeError("Profiling phases must not overlap")
        entry = {
            "stage": stage,
            **labels,
            "started_at": timestamp(),
            "finished_at": None,
            "status": "running",
            "seconds": 0,
            "rss_start_mb": None,
            "rss_end_mb": None,
            "rss_peak_mb": None,
            "rss_delta_mb": None,
            "groq": empty_usage(),
        }
        started = self.clock()
        with self._lock:
            self._active, self._active_started, self._span_peak = entry, started, None
        entry["rss_start_mb"] = self.sample_memory()
        self.attempt["spans"].append(entry)
        token = _span.set(entry)
        try:
            self.emit()
            yield entry
        except BaseException as exc:
            entry["status"] = "cancelled" if type(exc).__name__ == "RunCancelled" else "failed"
            raise
        else:
            entry["status"] = "completed"
        finally:
            entry["rss_end_mb"] = self.sample_memory()
            with self._lock:
                entry["rss_peak_mb"] = (
                    round(self._span_peak, 3) if self._span_peak is not None else None
                )
                self._active = None
            if entry["rss_start_mb"] is not None and entry["rss_end_mb"] is not None:
                entry["rss_delta_mb"] = round(entry["rss_end_mb"] - entry["rss_start_mb"], 3)
            entry.update(seconds=round(self.clock() - started, 6), finished_at=timestamp())
            _span.reset(token)
            self.emit()

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=1)
        try:
            self.attempt["rss_end_mb"] = self.sample_memory()
            self.attempt.update(finished_at=timestamp(), status=self.final_status)
            self.emit()
        finally:
            _current.reset(self._token)

    @property
    def final_status(self):
        return getattr(self, "status", "failed")


@contextmanager
def measure(stage, **labels):
    from backend.observability import stage as observe_stage

    profiler = _current.get()
    with (
        observe_stage(stage, **labels),
        profiler.phase(stage, **labels) if profiler else nullcontext(),
    ):
        yield


def begin_request():
    profiler, entry = _current.get(), _span.get()
    if not profiler or entry is None:
        return None
    entry["groq"]["http_requests"] += 1
    profiler.emit()  # Persist an attempt even if the worker dies before the response.
    return profiler, entry, profiler.clock()


def number(value):
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def finish_request(handle, response=None):
    if handle is None:
        return
    profiler, entry, started = handle
    stats = entry["groq"]
    stats["http_seconds"] = round(stats["http_seconds"] + profiler.clock() - started, 6)
    if response is None:
        stats["transport_errors"] += 1
    else:
        success = 200 <= response.status_code < 300
        stats["http_successes" if success else "http_failures"] += 1
        stats["rate_limit_responses"] += int(response.status_code == 429)
        try:
            payload = response.json()
            usage = payload.get("usage") if isinstance(payload, dict) else None
            usage = usage if isinstance(usage, dict) else {}
        except (ValueError, httpx.ResponseNotRead):
            usage = {}
        prompt, completion, total = (
            number(usage.get(k)) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        # Sum reported components if the provider omitted the redundant total.
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        complete = all(v is not None for v in (prompt, completion, total))
        if success:
            stats["usage_reports" if complete else "missing_usage_reports"] += 1
        for key, value in (
            ("prompt_tokens", prompt),
            ("completion_tokens", completion),
            ("total_tokens", total),
        ):
            if value is not None:
                stats[key] += value
                stats[key.replace("_tokens", "_token_reports")] += 1
        for source, detail, target in (
            ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
            ("prompt_tokens_details", "cached_tokens", "cached_tokens"),
        ):
            details = usage.get(source)
            value = number(details.get(detail)) if isinstance(details, dict) else None
            if value is not None:
                stats[target] += value
                stats[target.replace("_tokens", "_token_reports")] += 1
        provider_time = number(usage.get("total_time"))
        if provider_time is not None:
            stats["provider_seconds"] = round(stats["provider_seconds"] + provider_time, 6)
    profiler.emit()


class ProfiledClient(httpx.Client):
    """Wrap send so Groq SDK retries and transport failures are counted individually."""

    def send(self, request, **kwargs):
        from backend.guardrails import reserve

        reserve(request)
        from backend.optimization_budget import current_budget

        budget = current_budget()
        if budget:
            budget.reserve()
        from backend.observability import begin_provider_attempt, provider_response

        observation_handle = begin_provider_attempt()
        handle = begin_request()
        try:
            response = super().send(request, **kwargs)
        except BaseException:
            finish_request(handle)
            raise
        provider_response(response, observation_handle)
        finish_request(handle, response)
        if budget:
            budget.finish(response)
        return response


class ProfiledAsyncClient(httpx.AsyncClient):
    async def send(self, request, **kwargs):
        from backend.guardrails import reserve

        reserve(request)
        from backend.optimization_budget import current_budget

        budget = current_budget()
        if budget:
            budget.reserve()
        from backend.observability import begin_provider_attempt, provider_response

        observation_handle = begin_provider_attempt()
        handle = begin_request()
        try:
            response = await super().send(request, **kwargs)
        except BaseException:
            finish_request(handle)
            raise
        provider_response(response, observation_handle)
        finish_request(handle, response)
        if budget:
            budget.finish(response)
        return response
