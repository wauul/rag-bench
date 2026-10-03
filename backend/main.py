import hmac
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from backend.ingestion import MAX_BYTES, MAX_TEXT, extract_document, parse_test_set
from backend.models import Configuration, RunRequest, TestSet, METRICS
from backend.storage import Store
from backend.run_state import ACTIVE_STATES, validate_retry
from backend.settings import load_settings
from backend.profiling_report import interrupt_profile, profile_csv_rows, row_profile

store = Store()
executor = ThreadPoolExecutor(max_workers=1)
run_lock = threading.Lock()
control_lock = threading.Lock()
cancel_events = {}
sample_dir = Path(__file__).resolve().parents[1] / "sample_data"


@asynccontextmanager
async def lifespan(app):
    load_settings()
    # A killed process cannot resume its in-memory worker. Keep partial rows, mark honestly.
    for run in store.list("run"):
        if run["status"] in {"queued", "running"}:
            interrupt_profile(run)
            run.update(status="failed", stage="Server restarted; retry to resume saved work", error="Interrupted by server restart")
            store.save("run", run, run["id"])
    yield


def authorize(authorization: str | None = Header(default=None)):
    token = os.getenv("API_TOKEN", "")
    if token and not hmac.compare_digest(authorization or "", "Bearer " + token):
        raise HTTPException(401, "Invalid API token")


app = FastAPI(title="RAG Bench", version="1.0.0", lifespan=lifespan)
auth = [Depends(authorize)]


def fetch(kind, object_id):
    try:
        return store.get(kind, object_id)
    except KeyError:
        raise HTTPException(404, f"Unknown {kind} ID")


@app.get("/health")
def health():
    return {"status": "ok", "generation_ready": bool(os.getenv("GROQ_API_KEY")),
            "authentication_required": bool(os.getenv("API_TOKEN")),
            "revision": os.getenv("RENDER_GIT_COMMIT", "local")}


@app.post("/api/documents", dependencies=auth, status_code=201)
async def documents(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 10:
        raise HTTPException(422, "Upload between 1 and 10 documents")
    docs = []
    try:
        for file in files:
            docs.extend(extract_document(file.filename or "document.txt", await file.read(MAX_BYTES + 1)))
        if sum(len(d["text"]) for d in docs) > MAX_TEXT:
            raise ValueError("Document set exceeds 150,000 characters")
    except Exception as exc:
        raise HTTPException(422, str(exc) if isinstance(exc, ValueError) else "Document extraction failed")
    saved = store.save("documents", {"documents": docs})
    return {"id": saved["id"], "pages": len(docs), "characters": sum(len(d["text"]) for d in docs)}


@app.post("/api/test-sets", dependencies=auth, status_code=201)
def test_sets(body: TestSet):
    return store.save("test_set", body.model_dump())


@app.post("/api/test-sets/upload", dependencies=auth, status_code=201)
async def upload_test_set(file: UploadFile = File(...)):
    try:
        body = parse_test_set(file.filename or "questions.csv", await file.read(MAX_BYTES + 1))
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, str(exc))
    return store.save("test_set", body.model_dump())


@app.post("/api/configurations", dependencies=auth, status_code=201)
def configurations(body: Configuration):
    return store.save("configuration", body.model_dump())


@app.post("/api/demo", dependencies=auth, status_code=201)
def demo():
    doc = store.save("documents", {"documents": extract_document("harbor-handbook.txt", (sample_dir / "harbor-handbook.txt").read_bytes())})
    test = test_sets(TestSet(**json.loads((sample_dir / "questions.json").read_text(encoding="utf-8"))))
    # Two final passages keep judge quota modest; reranking can select from a larger pool.
    configs = [configurations(Configuration(name="MiniLM · 96 tokens", chunk_size=96, overlap=16, context_k=2)),
               configurations(Configuration(name="BGE · 192 + rerank", embedding_model="BAAI/bge-small-en-v1.5", rerank=True, context_k=2, candidate_k=12))]
    return {"document_set_id": doc["id"], "test_set_id": test["id"],
            "configuration_ids": [c["id"] for c in configs], "configurations": configs,
            "questions": test["questions"]}


def worker(run_id, cancel_event, retry=False):
    try:
        from backend.pipeline import execute_run
        execute_run(store, run_id, cancel_event=cancel_event, retry=retry)
    except Exception as exc:
        run = store.get("run", run_id)
        run.update(status="failed", error=type(exc).__name__ + ": worker could not start", stage="Worker failed")
        store.save("run", run, run_id)
    finally:
        with control_lock:
            cancel_events.pop(run_id, None)
            run_lock.release()


def submit_run(run, retry=False):
    event = threading.Event()
    with control_lock:
        cancel_events[run["id"]] = event
    try:
        executor.submit(worker, run["id"], event, retry)
    except Exception:
        with control_lock:
            cancel_events.pop(run["id"], None)
        run.update(status="failed", stage="Worker could not be queued", error="Worker submission failed")
        store.save("run", run, run["id"])
        raise


def require_generation():
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(503, "Set GROQ_API_KEY on the backend before running evaluations")
    try:
        return load_settings()
    except ValueError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/runs", dependencies=auth, status_code=202)
def start_run(body: RunRequest):
    settings = require_generation()
    fetch("documents", body.document_set_id)
    test = fetch("test_set", body.test_set_id)
    configs = [{**Configuration(**fetch("configuration", cid)).model_dump(), "id": cid}
               for cid in body.configuration_ids]
    if len({c["name"].strip().casefold() for c in configs}) != len(configs):
        raise HTTPException(422, "Configuration names must be distinct for comparison charts")
    if not run_lock.acquire(blocking=False):
        raise HTTPException(409, "An evaluation is already running. Wait for it to finish.")
    try:
        from backend.pipeline import now
        run = store.save("run", {**body.model_dump(), "configurations": configs, "questions": test["questions"],
            "status": "queued", "stage": "Queued", "created_at": now(), "completed": 0,
            "total": len(configs) * len(test["questions"]), "rows": [], "summary": [],
            "attempt": 1, "retrieved": 0, "scored": 0, "valid_scores": 0,
            "provenance": settings.provenance()})
        submit_run(run)
    except Exception:
        run_lock.release()
        raise
    return {"id": run["id"], "status": run["status"]}


@app.get("/api/runs", dependencies=auth)
def list_runs(limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    items = store.run_history(limit + 1, offset)
    return {"runs": items[:limit], "next_offset": offset + limit if len(items) > limit else None}


@app.post("/api/runs/{run_id}/retry", dependencies=auth, status_code=202)
def retry_run(run_id: str):
    require_generation()
    if not run_lock.acquire(blocking=False):
        raise HTTPException(409, "An evaluation is already running. Wait for it to finish.")
    try:
        run = fetch("run", run_id)
        try:
            validate_retry(run)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        if len(run["rows"]) < run["total"]:
            fetch("documents", run["document_set_id"])
        from backend.pipeline import now
        run.update(status="queued", stage="Queued to resume missing work",
                   attempt=run.get("attempt", 1) + 1, retried_at=now())
        run.pop("error", None)
        run.pop("finished_at", None)
        store.save("run", run, run_id)
        submit_run(run, retry=True)
    except Exception:
        run_lock.release()
        raise
    return {"id": run_id, "status": "queued"}


@app.post("/api/runs/{run_id}/cancel", dependencies=auth, status_code=202)
def cancel_run(run_id: str):
    # The worker owns persisted run data; an event avoids overwriting in-flight results.
    with control_lock:
        run = fetch("run", run_id)
        event = cancel_events.get(run_id)
        if run["status"] not in ACTIVE_STATES or event is None:
            raise HTTPException(409, "This run is not active")
        event.set()
    return {"id": run_id, "status": run["status"], "cancel_requested": True}


@app.get("/api/runs/{run_id}", dependencies=auth)
def get_run(run_id: str):
    run = fetch("run", run_id)
    with control_lock:
        event = cancel_events.get(run_id)
        run["cancel_requested"] = event is not None and event.is_set()
    return run


def spreadsheet_safe(value):
    # Prevent uploaded answers/questions becoming formulas when a CSV is opened in Excel.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


@app.get("/api/runs/{run_id}/export", dependencies=auth)
def export(run_id: str):
    run = fetch("run", run_id)
    if not run["rows"]:
        raise HTTPException(409, "No result rows available yet")
    rows = []
    for row in run["rows"]:
        measured = row_profile(run.get("profiling", {}), row.get("configuration_id"), row.get("question_index")) or {}
        timings, usage = measured.get("timings", {}), measured.get("groq", {})
        rows.append({**{k: v for k, v in row.items() if k not in {"scores", "contexts", "errors"}},
            **row["scores"], "contexts": json.dumps(row["contexts"], ensure_ascii=False), "errors": json.dumps(row["errors"]),
            "generation_seconds": timings.get("generation"), "scoring_seconds": timings.get("scoring"),
            **{f"{metric}_seconds": measured.get("metrics", {}).get(metric, {}).get("timings", {}).get("scoring") for metric in METRICS},
            "groq_http_requests": usage.get("http_requests"),
            **{f"groq_{field}_tokens": usage.get(f"{field}_tokens") if usage.get(f"{field}_token_reports") else None
               for field in ("prompt", "completion", "total")}})
    frame = pd.DataFrame(rows).map(spreadsheet_safe)
    return StreamingResponse(io.StringIO(frame.to_csv(index=False)), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="rag-bench-{run["id"]}.csv"'})


@app.get("/api/runs/{run_id}/profile", dependencies=auth)
def get_profile(run_id: str):
    run = fetch("run", run_id)
    return {"run_id": run_id, "profiling": run.get("profiling")}


@app.get("/api/runs/{run_id}/profile/export", dependencies=auth)
def export_profile(run_id: str):
    run = fetch("run", run_id)
    rows = profile_csv_rows(run.get("profiling", {}))
    if not rows:
        raise HTTPException(409, "No profiling measurements available for this run")
    names = {c["id"]: c["name"] for c in run["configurations"]}
    for row in rows:
        row.update(run_id=run_id, configuration_name=names.get(row.get("configuration_id")))
    frame = pd.DataFrame(rows).map(spreadsheet_safe)
    return StreamingResponse(io.StringIO(frame.to_csv(index=False)), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="rag-bench-profile-{run_id}.csv"'})
