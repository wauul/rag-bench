"""Run the full real demo against a running backend and require complete finite scores."""
import os
import time
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()
base = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
headers = {"Authorization": "Bearer " + os.environ["API_TOKEN"]} if os.getenv("API_TOKEN") else {}
with httpx.Client(base_url=base, headers=headers, timeout=120) as client:
    response = client.post("/api/demo")
    response.raise_for_status()
    demo = response.json()
    response = client.post("/api/runs", json={k: demo[k] for k in ["document_set_id", "test_set_id", "configuration_ids"]})
    response.raise_for_status()
    run_id = response.json()["id"]
    print("Run:", run_id, flush=True)
    deadline = time.monotonic() + 7200
    previous = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        response.raise_for_status()
        run = response.json()
        if run["stage"] != previous:
            print(run["stage"], flush=True)
            previous = run["stage"]
        if run["status"] not in {"queued", "running"}:
            break
        time.sleep(5)
    else:
        raise TimeoutError(f"Run {run_id} still running; poll it in the dashboard")
    out = Path("data/full-demo-results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(response.text, encoding="utf-8")
    assert run["status"] == "completed", f"Incomplete run; inspect {out}"
    assert len(run["rows"]) == 20
    assert any(s["overall"] > 0 for s in run["summary"])
    csv = client.get(f"/api/runs/{run_id}/export")
    csv.raise_for_status()
    print("Full real evaluation passed:", run["summary"], flush=True)
