"""Promotion must reject unknown usage/metrics, stale evidence and artifact mismatch."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from scripts.quality_gate import gate
from scripts.release_manifest import validate


def fixture():
    metrics = {
        k: 0.5 for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    }
    config = {
        k: "fixed"
        for k in ("dataset_fingerprint", "prompt_fingerprint", "model", "evaluator_fingerprint")
    }
    report = {
        **config,
        "revision": "candidate",
        "image": "digest",
        "sample_count": 10,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exploratory": False,
        "complete": True,
        "scores": metrics,
        "valid_counts": dict.fromkeys(metrics, 10),
        "human_review": {
            "reviewer": "test-reviewer",
            "rubric_version": "ragbench-review-1",
            "decision": "approve",
        },
    }
    policy = {
        **config,
        "reviewer": "test-reviewer",
        "baseline_runs": 3,
        "minimum_samples": 10,
        "maximum_age_hours": 24,
        "minimum_scores": dict.fromkeys(metrics, 0.4),
    }
    return report, policy


def test_complete_measured_evidence():
    report, policy = fixture()
    gate(report, policy, "candidate", "digest")


@pytest.mark.parametrize(
    "patch",
    [
        {"revision": "other"},
        {"image": "other"},
        {"complete": False},
        {"exploratory": True},
        {"sample_count": 1},
        {"scores": {}},
        {"quota_failures": 1},
        {"human_review": {}},
        {"errors": ["timeout"]},
        {"created_at": "2020-01-01T00:00:00+00:00"},
    ],
)
def test_incomplete_or_untrusted_evidence_fails(patch):
    report, policy = fixture()
    with pytest.raises(ValueError):
        gate({**report, **patch}, policy, "candidate", "digest")


def test_manifest_rejects_skipped_validation_and_sqlite_only():
    release = {
        "format": 1,
        "revision": "a" * 40,
        "storage_schema": 1,
        "checkpoint_serializer": "json-v1",
        "storage_backends": ["sqlite", "postgres"],
        "images": {
            k: "ghcr.io/wauul/rag-bench-" + k + "@sha256:" + "b" * 64
            for k in ("cpu", "compact", "dashboard")
        },
        "validation": {
            "revision": "a" * 40,
            "engineering": "success",
            "models": "success",
            "artifacts": "success",
        },
    }
    validate(release)
    copy = deepcopy(release)
    copy["validation"]["models"] = "skipped"
    with pytest.raises(ValueError):
        validate(copy)
    copy = deepcopy(release)
    copy["storage_backends"] = ["sqlite"]
    with pytest.raises(ValueError):
        validate(copy)
