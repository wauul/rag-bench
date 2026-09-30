"""Run progress and retry eligibility shared by the API and execution worker."""
import math
from backend.models import METRICS
from backend.settings import load_settings

ACTIVE_STATES = {"queued", "running"}
RETRY_STATES = {"partial", "failed", "cancelled"}


class RunCancelled(Exception):
    pass


def missing_metrics(row):
    return [m for m in METRICS if row["scores"].get(m) is None
            or not math.isfinite(row["scores"][m])]


def needs_work(row):
    return not row.get("answer") or bool(missing_metrics(row)) or bool(row["errors"])


def progress(run):
    rows = run["rows"]
    return {"completed": sum(r.get("processed", True) or not needs_work(r) for r in rows),
            "retrieved": len(rows), "scored": sum(not needs_work(r) for r in rows),
            "valid_scores": sum(len(METRICS) - len(missing_metrics(r)) for r in rows)}


def validate_retry(run):
    if run["status"] not in RETRY_STATES:
        raise ValueError("Retry requires a partial, failed or cancelled run")
    expected = load_settings().provenance()
    # Older runs did not record all settings; enforce their known measurement settings.
    required = {"generator", "judge", "inference_backend", "judge_prompt_examples"}
    recorded = run.get("provenance", {})
    if any(recorded.get(k) != v for k, v in expected.items() if k in required or k in recorded):
        raise ValueError("Restore the original model, inference backend and prompt settings before retry")
