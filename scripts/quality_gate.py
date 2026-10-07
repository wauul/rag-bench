"""Fail closed on stale, incomplete or mismatched measured evaluation evidence.

Policies are reviewed files, derived from repeat baseline evaluations; no universal
Ragas threshold is invented. Smoke/exploratory evidence cannot authorize promotion.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def gate(report: dict, policy: dict, revision: str, image: str) -> None:
    if report.get("revision") != revision or report.get("image") != image:
        raise ValueError("Evaluation does not describe the exact candidate artifact")
    for field in ("dataset_fingerprint", "prompt_fingerprint", "model", "evaluator_fingerprint"):
        if not policy.get(field) or report.get(field) != policy[field]:
            raise ValueError("Evaluation configuration mismatch: " + field)
    if not policy.get("reviewer") or policy.get("baseline_runs", 0) < 3:
        raise ValueError("Policy needs reviewed repeated baseline measurements")
    samples = policy.get("minimum_samples", 0)
    if samples < 2 or report.get("sample_count", 0) < samples or report.get("exploratory", True):
        raise ValueError("Insufficient independent evaluation samples")
    created = datetime.fromisoformat(report["created_at"])
    if created.tzinfo is None:
        raise ValueError("Evaluation timestamp must have a timezone")
    age = (datetime.now(timezone.utc) - created).total_seconds()
    if not 0 <= age <= policy["maximum_age_hours"] * 3600:
        raise ValueError("Stale or future evaluation evidence")
    if report.get("errors") or report.get("quota_failures") or not report.get("complete"):
        raise ValueError("Incomplete scoring or provider/quota failure")
    if set(policy.get("minimum_scores", {})) != {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }:
        raise ValueError("Policy must cover all four fixed Ragas metrics")
    for name, minimum in policy["minimum_scores"].items():
        if (
            not isinstance(minimum, (int, float))
            or isinstance(minimum, bool)
            or not math.isfinite(minimum)
            or not 0 <= minimum <= 1
        ):
            raise ValueError("Invalid measured threshold: " + name)
        score = report.get("scores", {}).get(name)
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not 0 <= score <= 1
            or score < minimum
            or report.get("valid_counts", {}).get(name) != report["sample_count"]
        ):
            raise ValueError("Missing score, incomplete coverage or regression: " + name)
    review = report.get("human_review", {})
    if (
        review.get("reviewer") != policy.get("reviewer")
        or review.get("rubric_version") != "ragbench-review-1"
        or review.get("decision") != "approve"
    ):
        raise ValueError("A provenance-bound human review is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    gate(
        json.loads(args.report.read_text()),
        json.loads(args.policy.read_text()),
        args.revision,
        args.image,
    )
    print("Reviewed evaluation evidence passed the measured promotion policy")


if __name__ == "__main__":
    main()
