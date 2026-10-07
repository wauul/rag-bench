"""Durable reservation before every HTTP send, including SDK retry attempts."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

from backend.run_state import RunCancelled

_budget = ContextVar("optimization_budget", default=None)


class ExperimentBudget:
    def __init__(self, store, record):
        self.store, self.record = store, record
        self.lock = Lock()
        self.exhausted = False

    def check(self):
        from backend.guardrails import check

        check(self.store)
        try:
            signal = self.store.get("optimization_cancel", self.record["id"] + "-cancel")
            if signal["attempt"] == self.record["attempt"]:
                raise RunCancelled()
        except KeyError:
            pass
        if self.exhausted:
            raise RunCancelled()

    def reserve(self):
        with self.lock:
            self.check()
            usage = self.record["usage"]
            if usage["requests_reserved"] >= self.record["max_requests"]:
                self.exhausted = True
                raise RunCancelled()
            usage["requests_reserved"] += 1
            self.store.save("optimization", self.record, self.record["id"])

    def finish(self, response):
        if response is None:
            return
        try:
            usage = response.json().get("usage", {})
            total = usage.get("total_tokens")
            if total is None and all(
                isinstance(usage.get(k), int) for k in ("prompt_tokens", "completion_tokens")
            ):
                total = usage["prompt_tokens"] + usage["completion_tokens"]
            if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                with self.lock:
                    self.record["usage"]["total_tokens"] += total
                    self.record["usage"]["token_reports"] += 1
                    self.store.save("optimization", self.record, self.record["id"])
        except (ValueError, AttributeError):
            pass


@contextmanager
def budget_context(budget):
    token = _budget.set(budget)
    try:
        yield
    finally:
        _budget.reset(token)


def current_budget():
    return _budget.get()
