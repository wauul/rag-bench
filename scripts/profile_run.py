"""Download a saved run's performance report: python -m scripts.profile_run RUN_ID."""
import json
import os
import sys
from pathlib import Path
import httpx
from dotenv import load_dotenv


def main():
    load_dotenv()
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.profile_run RUN_ID")
    run_id = sys.argv[1]
    base = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    headers = {"Authorization": "Bearer " + os.environ["API_TOKEN"]} if os.getenv("API_TOKEN") else {}
    with httpx.Client(base_url=base, headers=headers, timeout=120) as client:
        response = client.get(f"/api/runs/{run_id}/profile")
        response.raise_for_status()
        profile = response.json()["profiling"]
        if not profile:
            raise SystemExit("Profiling was not recorded for this run. Start a new benchmark to collect it.")
        # Run IDs are also accepted from external input; filenames never contain path separators.
        filename = "".join(c for c in run_id if c.isalnum() or c in "-_")
        if not filename:
            raise SystemExit("Run ID must contain letters or digits")
        output = Path("data") / f"profile-{filename}.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
        csv = client.get(f"/api/runs/{run_id}/profile/export")
        if csv.status_code != 409:
            csv.raise_for_status()
            output.with_suffix(".csv").write_bytes(csv.content)
        print(json.dumps(profile.get("summary", {}), indent=2))
        print(f"Saved performance report: {output}")


if __name__ == "__main__":
    main()
