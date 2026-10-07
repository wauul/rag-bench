"""Pure projections of saved profiling data; also usable by the lightweight dashboard."""

from collections import defaultdict

USAGE_FIELDS = (
    "http_requests",
    "http_successes",
    "http_failures",
    "transport_errors",
    "rate_limit_responses",
    "usage_reports",
    "missing_usage_reports",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "prompt_token_reports",
    "completion_token_reports",
    "total_token_reports",
    "reasoning_token_reports",
    "cached_token_reports",
    "http_seconds",
    "provider_seconds",
)


def empty_usage():
    return dict.fromkeys(USAGE_FIELDS, 0)


def spans(profile):
    for attempt in profile.get("attempts", []):
        for span in attempt.get("spans", []):
            yield {"attempt": attempt["number"], **span}


def aggregate(entries):
    usage = empty_usage()
    timings = defaultdict(float)
    peaks = []
    for entry in entries:
        timings[entry["stage"]] += entry.get("seconds", 0)
        if entry.get("rss_peak_mb") is not None:
            peaks.append(entry["rss_peak_mb"])
        for key in USAGE_FIELDS:
            usage[key] += entry.get("groq", {}).get(key, 0)
    return {
        "timings": {key: round(value, 6) for key, value in timings.items()},
        "rss_peak_mb": max(peaks) if peaks else None,
        "groq": usage,
    }


def summarize_profile(profile):
    entries = list(spans(profile))
    summary = aggregate(entries)
    peaks = [
        a["rss_peak_mb"] for a in profile.get("attempts", []) if a.get("rss_peak_mb") is not None
    ]
    summary["rss_peak_mb"] = max(peaks) if peaks else None
    summary["worker_seconds"] = round(
        sum(a.get("seconds", 0) for a in profile.get("attempts", [])), 6
    )
    summary["configurations"] = {
        cid: aggregate(e for e in entries if e.get("configuration_id") == cid)
        for cid in dict.fromkeys(
            e["configuration_id"] for e in entries if e.get("configuration_id")
        )
    }
    return summary


def row_profile(profile, configuration_id, question_index):
    entries = [
        e
        for e in spans(profile)
        if e.get("configuration_id") == configuration_id
        and e.get("question_index") == question_index
    ]
    if not entries:
        return None
    result = aggregate(entries)
    result["metrics"] = {
        metric: aggregate(e for e in entries if e.get("metric") == metric)
        for metric in dict.fromkeys(e["metric"] for e in entries if e.get("metric"))
    }
    return result


def profile_csv_rows(profile):
    rows = []
    for entry in spans(profile):
        usage = entry.get("groq", {})
        flattened = {f"groq_{key}": value for key, value in usage.items()}
        for field in ("prompt", "completion", "total", "reasoning", "cached"):
            if not usage.get(f"{field}_token_reports"):
                flattened[f"groq_{field}_tokens"] = None
        rows.append({**{key: value for key, value in entry.items() if key != "groq"}, **flattened})
    return rows


def interrupt_profile(run):
    """Mark saved open spans incomplete; never invent the time or usage lost in a crash."""
    for attempt in run.get("profiling", {}).get("attempts", []):
        if attempt["status"] == "running":
            attempt["status"] = "interrupted"
            for entry in attempt.get("spans", []):
                if entry["status"] == "running":
                    entry["status"] = "interrupted"
