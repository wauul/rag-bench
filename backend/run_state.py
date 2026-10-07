"""Run progress and retry eligibility shared by the API and execution worker."""

import math

from backend.models import METRICS
from backend.settings import load_settings

ACTIVE_STATES = {"queued", "running"}
RETRY_STATES = {"partial", "failed", "cancelled"}


class RunCancelled(Exception):
    pass


def missing_metrics(row):
    return [m for m in METRICS if not valid_score(m, row["scores"].get(m))]


def valid_score(metric, value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (-1 if metric == "answer_relevancy" else 0) <= value <= 1
    )


def needs_work(row):
    return not has_answer(row) or bool(missing_metrics(row)) or bool(row["errors"])


def has_answer(row):
    return isinstance(row.get("answer"), str) and bool(row["answer"].strip())


def progress(run):
    rows = run["rows"]
    total = run.get("total", len(rows))
    answered = sum(has_answer(r) for r in rows)
    valid = sum(len(METRICS) - len(missing_metrics(r)) for r in rows)
    return {
        "completed": sum(r.get("processed", True) or not needs_work(r) for r in rows),
        "retrieved": len(rows),
        "scored": sum(not needs_work(r) for r in rows),
        "valid_scores": valid,
        "answered": answered,
        "pending_answers": total - answered,
        "pending_metrics": total * len(METRICS) - valid,
    }


def validate_retry(run):
    if run["status"] not in RETRY_STATES:
        raise ValueError("Retry requires a partial, failed or cancelled run")
    expected = load_settings().provenance()
    # Older runs did not record all settings; enforce their known measurement settings.
    required = {"generator", "judge", "inference_backend", "judge_prompt_examples"}
    recorded = run.get("provenance", {})
    if any(recorded.get(k) != v for k, v in expected.items() if k in required or k in recorded):
        raise ValueError(
            "Restore the original model, inference backend and prompt settings before retry"
        )
