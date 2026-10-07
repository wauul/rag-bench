"""On-demand numeric operational readout. No keep-alive polling or provider requests."""

import argparse
import json
from collections import Counter
from pathlib import Path

from backend.models import METRICS
from backend.profiling_report import summarize_profile
from backend.run_state import valid_score
from backend.storage import Store


def summarize(store):
    result = {}
    for workflow, kind in {
        "benchmark": "run",
        "investigation": "investigation",
        "optimization": "optimization",
        "debugger": "debug_run",
    }.items():
        records = store.list(kind)
        # Optimization-owned trials do not count toward independent benchmark accounting.
        if kind == "run":
            records = [r for r in records if not r.get("optimization_id")]
        statuses = Counter(r.get("status", "unknown") for r in records)
        tokens = reports = requests = missing = 0
        latencies = []
        score_rows = []
        for record in records:
            if kind == "investigation":
                requests += len(record.get("attempts", []))
                for attempt in record.get("attempts", []):
                    usage = attempt.get("usage", {})
                    if isinstance(usage.get("total_tokens"), int):
                        tokens += usage["total_tokens"]
                        reports += 1
                    else:
                        missing += 1
                    if "seconds" in attempt:
                        latencies.append(attempt["seconds"])
            elif kind == "optimization":
                usage = record.get("usage", {})
                requests += usage.get("requests_reserved", 0)
                tokens += usage.get("total_tokens", 0)
                reports += usage.get("token_reports", 0)
                missing += max(0, usage.get("requests_reserved", 0) - usage.get("token_reports", 0))
                for trial in record.get("trials", []):
                    try:
                        run = store.get("run", trial["run_id"])
                    except KeyError:
                        continue
                    score_rows.extend(run.get("rows", []))
                    profile = summarize_profile(run.get("profiling", {}))
                    if profile["worker_seconds"]:
                        latencies.append(profile["worker_seconds"])
            else:
                if kind == "run":
                    score_rows.extend(record.get("rows", []))
                summary = summarize_profile(record.get("profiling", {}))
                usage = summary.get("groq", {})
                requests += usage.get("http_requests", 0)
                tokens += usage.get("total_tokens", 0)
                reports += usage.get("total_token_reports", 0)
                missing += usage.get("missing_usage_reports", 0)
                if "worker_seconds" in summary:
                    latencies.append(summary["worker_seconds"])
        terminal = sum(statuses[k] for k in ("completed", "failed", "cancelled"))
        result[workflow] = {
            "records": len(records),
            "statuses": dict(statuses),
            "completion_rate": statuses["completed"] / terminal if terminal else None,
            "error_rate": statuses["failed"] / terminal if terminal else None,
            "provider_attempts_or_reservations": requests,
            "reported_tokens": tokens if reports else None,
            "token_reports": reports,
            "missing_usage_reports_or_reservations": missing,
            "estimated_spend": None,
            "pricing_source": None,
            "pricing_date": None,
            "latency_samples_seconds": latencies,
            "scores": {
                metric: {
                    "valid_count": sum(
                        valid_score(metric, r.get("scores", {}).get(metric)) for r in score_rows
                    ),
                    "expected_count": len(score_rows),
                    "missing_count": sum(
                        not valid_score(metric, r.get("scores", {}).get(metric)) for r in score_rows
                    ),
                }
                for metric in METRICS
            }
            if kind in {"run", "optimization"}
            else None,
        }
    return result


def alerts(readout, policy):
    if policy.get("baseline_runs", 0) < 3 or not policy.get("reviewer"):
        raise ValueError("Alert thresholds require reviewed baseline measurements")
    result = []
    for workflow, thresholds in policy["workflows"].items():
        current = readout[workflow]
        if current["records"] < thresholds["minimum_samples"]:
            continue
        if (
            current["error_rate"] is not None
            and current["error_rate"] > thresholds["maximum_error_rate"]
        ):
            result.append(
                {
                    "workflow": workflow,
                    "alert": "error-rate",
                    "action": "Inspect durable error kinds and provider quota before explicit resume",
                }
            )
        if current["missing_usage_reports_or_reservations"] > thresholds["maximum_missing_usage"]:
            result.append(
                {
                    "workflow": workflow,
                    "alert": "usage-coverage",
                    "action": "Treat spend as unknown; inspect response-before-save and quota evidence",
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/operations.json"))
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    readout = summarize(Store())
    report = {
        "workflows": readout,
        "alerts": alerts(readout, json.loads(args.policy.read_text())) if args.policy else [],
        "alert_status": "configured" if args.policy else "awaiting-measured-baselines",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Numeric operational report written; no provider calls or user content exported")


if __name__ == "__main__":
    main()
