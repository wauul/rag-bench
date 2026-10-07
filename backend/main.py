import asyncio
import io
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from backend.debugger_api import install as install_debugger_routes
from backend.ingestion import MAX_BYTES, MAX_TEXT, extract_bounded, extract_document, parse_test_set
from backend.investigation_api import install as install_investigation_routes
from backend.models import METRICS, Configuration, RunRequest, TestSet
from backend.optimization_api import install as install_optimization_routes
from backend.profiling_report import interrupt_profile, profile_csv_rows, row_profile
from backend.run_state import ACTIVE_STATES, validate_retry
from backend.settings import load_settings, validate_operations
from backend.storage import Store

load_dotenv()

store = Store()
executor = ThreadPoolExecutor(max_workers=1)
run_lock = threading.Lock()
control_lock = threading.Lock()
cancel_events = {}
sample_dir = Path(__file__).resolve().parents[1] / "sample_data"


@asynccontextmanager
async def lifespan(app):
    load_settings()
    validate_operations()
    # A killed process cannot resume its in-memory worker. Keep partial rows, mark honestly.
    from backend.execution_lock import ExecutionBusy, execution_lock

    with execution_lock(store, "server.lock"):
        try:
            with execution_lock(store):
                for debug in store.list("debug_run"):
                    if debug["status"] in {"queued", "running"}:
                        debug.update(status="failed", error="Server restarted; resume saved stages")
                        store.save("debug_run", debug, debug["id"])
                for experiment in store.list("optimization"):
                    if experiment["status"] in {"queued", "running"}:
                        experiment.update(
                            status="failed", stage="Server restarted; explicit resume required"
                        )
                        store.save("optimization", experiment, experiment["id"])
                for investigation in store.list("investigation"):
                    if investigation["status"] in {"queued", "running"}:
                        investigation.update(
                            status="failed", error="Server restarted; resume saved stages"
                        )
                        store.save("investigation", investigation, investigation["id"])
                for run in store.list("run"):
                    if run["status"] in {"queued", "running"}:
                        interrupt_profile(run)
                        run.update(
                            status="failed",
                            stage="Server restarted; retry to resume saved work",
                            error="Interrupted by server restart",
                        )
                        store.save("run", run, run["id"])
        except ExecutionBusy:
            if not store.postgres:
                raise
            # Render's old instance may still be finishing its worker during replacement.
        try:
            from backend.maintenance import maintain

            maintenance = asyncio.create_task(maintain(store))
            yield
        finally:
            maintenance.cancel()
            with suppress(asyncio.CancelledError):
                await maintenance
            with control_lock:
                for event in cancel_events.values():
                    event.set()
            for experiment in store.list("optimization"):
                if experiment["status"] in {"queued", "running"}:
                    store.save(
                        "optimization_cancel",
                        {"attempt": experiment["attempt"]},
                        experiment["id"] + "-cancel",
                    )
            if isinstance(executor, ThreadPoolExecutor):
                await asyncio.to_thread(executor.shutdown, wait=True)
                from backend.observability import flush

                flush()


def authorize(authorization: str | None = Header(default=None)):
    from backend.identity import resolve

    token = os.getenv("API_TOKEN", "")
    if not token and os.getenv("RAGBENCH_ALLOW_INSECURE_LOCAL") != "true":
        raise HTTPException(503, "API access is not configured")
    try:
        return resolve(store, authorization)
    except ValueError:
        raise HTTPException(401, "Invalid API token")


app = FastAPI(
    title="RAG Bench",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
auth = [Depends(authorize)]


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Pydantic's default response includes uploaded values and provider-bound inputs.
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                for error in exc.errors()
            ]
        },
    )


class BodyLimitMiddleware:
    """Bound streaming/chunked requests before multipart parsing allocates disk or RAM."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = 12 * 1024 * 1024
        total = 0
        chunks = []
        deadline = asyncio.get_running_loop().time() + 30
        # No downstream parser runs until the bounded request body is complete.
        while True:
            try:
                remaining = deadline - asyncio.get_running_loop().time()
                message = await asyncio.wait_for(receive(), timeout=min(10, remaining))
            except TimeoutError:
                return await JSONResponse(
                    status_code=408, content={"detail": "Body read timed out"}
                )(scope, receive, send)
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > limit:
                response = JSONResponse(
                    status_code=413, content={"detail": "Request exceeds 12 MB"}
                )
                return await response(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break

        async def bounded_receive():
            return chunks.pop(0) if chunks else await receive()

        await self.app(scope, bounded_receive, send)


app.add_middleware(BodyLimitMiddleware)


class AdmissionMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not (
            scope["path"].startswith("/api/") or scope["path"] == "/ready"
        ):
            return await self.app(scope, receive, send)
        from backend.guardrails import check, throttle
        from backend.identity import as_user

        write = scope["method"] not in {"GET", "HEAD", "OPTIONS"}
        try:
            # Socket peer only: never accept arbitrary X-Forwarded-For values.
            peer = (scope.get("client") or ("unknown",))[0]
            if not await asyncio.to_thread(throttle, store, "ip:" + peer, write):
                raise HTTPException(429, "Request limit exceeded")
            headers = dict(scope["headers"])
            user = authorize(headers.get(b"authorization", b"").decode("latin1"))
            if not await asyncio.to_thread(throttle, store, "user:" + user, write):
                raise HTTPException(429, "User request limit exceeded")
            if not await asyncio.to_thread(throttle, store, "workspace", write):
                raise HTTPException(429, "Workspace request limit exceeded")
            if write and scope["method"] != "DELETE" and not scope["path"].endswith("/cancel"):
                await asyncio.to_thread(check, store)
        except HTTPException as exc:
            return await JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})(
                scope, receive, send
            )
        except Exception:
            return await JSONResponse(
                status_code=503, content={"detail": "Workspace policy unavailable or disabled"}
            )(scope, receive, send)
        with as_user(user):
            await self.app(scope, receive, send)


app.add_middleware(AdmissionMiddleware)


@app.get("/api/session", dependencies=auth)
def session():
    from backend.guardrails import locked
    from backend.identity import owner

    with locked(store) as (_, __, policy):
        return {"user_id": owner.get(), "retention_days": policy.retention_days}


@app.get("/api/guardrails", dependencies=auth)
def guardrail_status():
    from backend.guardrails import locked

    with locked(store) as (_, state, policy):
        return {"policy": policy.model_dump(), "usage": state["usage"]}


def fetch(kind, object_id):
    try:
        return store.get(kind, object_id)
    except KeyError:
        raise HTTPException(404, f"Unknown {kind} ID")


@app.get("/api/observability/{workflow}/{record_id}", dependencies=auth)
def observation_link(workflow: str, record_id: str):
    from backend.observability import trace_link

    kinds = {
        "benchmark": "run",
        "investigation": "investigation",
        "optimization": "optimization",
        "debugger": "debug_run",
    }
    if workflow not in kinds:
        raise HTTPException(404, "Unknown workflow")
    fetch(kinds[workflow], record_id)
    return {"url": trace_link(workflow, record_id), "ingestion_verified": False}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "generation_ready": bool(os.getenv("GROQ_API_KEY")),
        "authentication_required": bool(os.getenv("API_TOKEN")),
        "guardrails_version": 2,
        "user_isolation": True,
        "alert_delivery_configured": all(
            os.getenv(n) for n in ("RESEND_API_KEY", "ALERT_FROM", "ALERT_TO")
        ),
        "storage": "postgres" if store.postgres else os.getenv("DATA_STORAGE", "unknown"),
        "revision": os.getenv("RENDER_GIT_COMMIT") or os.getenv("APP_REVISION", "local"),
    }


@app.get("/ready", dependencies=auth)
def ready():
    try:
        with store.connect() as db:
            db.execute("SELECT 1").fetchone()
        if (
            not os.access(store.root, os.W_OK)
            or shutil.disk_usage(store.root).free < 64 * 1024 * 1024
        ):
            raise OSError("Storage unavailable")
    except Exception as exc:
        raise HTTPException(503, "Storage is unavailable or has less than 64 MB free") from exc
    return {
        "status": "ready",
        "revision": health()["revision"],
        "generation_ready": bool(os.getenv("GROQ_API_KEY")),
        "worker_busy": run_lock.locked(),
    }


@app.post("/api/documents", dependencies=auth, status_code=201)
async def documents(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 10:
        raise HTTPException(422, "Upload between 1 and 10 documents")
    docs = []
    try:
        for file in files:
            docs.extend(
                await asyncio.to_thread(
                    extract_bounded, file.filename or "document.txt", await file.read(MAX_BYTES + 1)
                )
            )
        if sum(len(d["text"]) for d in docs) > MAX_TEXT:
            raise ValueError("Document set exceeds 150,000 characters")
    except Exception as exc:
        raise HTTPException(
            422, str(exc) if isinstance(exc, ValueError) else "Document extraction failed"
        )
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
    doc = store.save(
        "documents",
        {
            "documents": extract_document(
                "harbor-handbook.txt", (sample_dir / "harbor-handbook.txt").read_bytes()
            )
        },
    )
    test = test_sets(
        TestSet(**json.loads((sample_dir / "questions.json").read_text(encoding="utf-8")))
    )
    # Two final passages keep judge quota modest; reranking can select from a larger pool.
    configs = [
        configurations(
            Configuration(name="MiniLM · 96 tokens", chunk_size=96, overlap=16, context_k=2)
        ),
        configurations(
            Configuration(
                name="BGE · 192 + rerank",
                embedding_model="BAAI/bge-small-en-v1.5",
                rerank=True,
                context_k=2,
                candidate_k=12,
            )
        ),
    ]
    return {
        "document_set_id": doc["id"],
        "test_set_id": test["id"],
        "configuration_ids": [c["id"] for c in configs],
        "configurations": configs,
        "questions": test["questions"],
    }


def worker(run_id, cancel_event, retry=False):
    from backend.execution_lock import ExecutionBusy, execution_lock

    try:
        from backend.pipeline import execute_run

        execute_run(store, run_id, cancel_event=cancel_event, retry=retry)
    except ExecutionBusy:
        # An external recovery worker may own this run. Never overwrite its live results.
        run = store.get("run", run_id)
        if run["status"] == "queued":
            run.update(
                status="failed",
                stage="Data directory busy; retry when the other worker finishes",
                error="ExecutionBusy: another process owns the data directory",
            )
            store.save("run", run, run_id)
    except Exception as exc:
        # A lost database lock must never overwrite a replacement worker's progress.
        try:
            with execution_lock(store):
                run = store.get("run", run_id)
                if run["status"] in ACTIVE_STATES:
                    run.update(
                        status="failed",
                        error=type(exc).__name__ + ": worker could not start",
                        stage="Worker failed",
                    )
                    store.save("run", run, run_id)
        except ExecutionBusy:
            pass
    finally:
        with control_lock:
            cancel_events.pop(run_id, None)
            run_lock.release()


def submit_run(run, retry=False):
    event = threading.Event()
    with control_lock:
        cancel_events[run["id"]] = event
    try:
        from backend.identity import submit

        submit(executor, worker, run["id"], event, retry)
    except Exception:
        with control_lock:
            cancel_events.pop(run["id"], None)
        run.update(
            status="failed", stage="Worker could not be queued", error="Worker submission failed"
        )
        store.save("run", run, run["id"])
        raise


def require_generation():
    from backend.guardrails import check

    try:
        check(store)
    except Exception:
        raise HTTPException(503, "Workspace policy unavailable or disabled") from None
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(503, "Set GROQ_API_KEY on the backend before running evaluations")
    try:
        return load_settings()
    except ValueError as exc:
        raise HTTPException(503, str(exc))


install_investigation_routes(sys.modules[__name__])
install_optimization_routes(sys.modules[__name__])
install_debugger_routes(sys.modules[__name__])


@app.post("/api/runs", dependencies=auth, status_code=202)
def start_run(body: RunRequest):
    settings = require_generation()
    fetch("documents", body.document_set_id)
    test = fetch("test_set", body.test_set_id)
    configs = [
        {**Configuration(**fetch("configuration", cid)).model_dump(), "id": cid}
        for cid in body.configuration_ids
    ]
    if len({c["name"].strip().casefold() for c in configs}) != len(configs):
        raise HTTPException(422, "Configuration names must be distinct for comparison charts")
    if not run_lock.acquire(blocking=False):
        raise HTTPException(409, "An evaluation is already running. Wait for it to finish.")
    try:
        from backend.pipeline import now

        run = store.save(
            "run",
            {
                **body.model_dump(),
                "configurations": configs,
                "questions": test["questions"],
                "status": "queued",
                "stage": "Queued",
                "created_at": now(),
                "completed": 0,
                "total": len(configs) * len(test["questions"]),
                "rows": [],
                "summary": [],
                "attempt": 1,
                "retrieved": 0,
                "scored": 0,
                "valid_scores": 0,
                "provenance": settings.provenance(),
                "storage": health()["storage"],
            },
        )
        from backend.provenance import snapshot_run

        snapshot_run(store, run)
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
        if run.get("optimization_id"):
            raise HTTPException(
                409, "Resume this trial through its optimization experiment to enforce its budget"
            )
        try:
            validate_retry(run)
            from backend.provenance import validate_snapshot

            validate_snapshot(store, run)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        if len(run["rows"]) < run["total"] and "input_fingerprint" not in run:
            fetch("documents", run["document_set_id"])
        from backend.pipeline import now

        run.update(
            status="queued",
            stage="Queued to resume missing work",
            attempt=run.get("attempt", 1) + 1,
            retried_at=now(),
        )
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
        if run["status"] not in ACTIVE_STATES or (event is None and not store.postgres):
            raise HTTPException(409, "This run is not active")
        if store.postgres:
            store.save(
                "cancellation",
                {"run_id": run_id, "attempt": run.get("attempt", 1)},
                run_id + "-cancel",
            )
        if event is not None:
            event.set()
    return {"id": run_id, "status": run["status"], "cancel_requested": True}


@app.get("/api/runs/{run_id}", dependencies=auth)
def get_run(run_id: str):
    run = fetch("run", run_id)
    with control_lock:
        event = cancel_events.get(run_id)
        run["cancel_requested"] = event is not None and event.is_set()
        if store.postgres and run["status"] in ACTIVE_STATES:
            try:
                request = store.get("cancellation", run_id + "-cancel")
                run["cancel_requested"] |= request.get("attempt") == run.get("attempt", 1)
            except KeyError:
                pass
    return run


@app.delete("/api/runs/{run_id}", dependencies=auth, status_code=204)
def delete_run(run_id: str):
    from backend.execution_lock import ExecutionBusy, execution_lock

    if not run_lock.acquire(blocking=False):
        raise HTTPException(409, "Wait for the active evaluation before deleting data")
    try:
        with execution_lock(store):
            run = fetch("run", run_id)
            if run.get("optimization_id"):
                raise HTTPException(409, "Experiment trial evidence is retained with its report")
            if run["status"] in ACTIVE_STATES:
                raise HTTPException(409, "Cancel the run before deleting it")
            if any(
                i["run_id"] == run_id and i["status"] in ACTIVE_STATES
                for i in store.list("investigation")
            ):
                raise HTTPException(409, "Cancel the investigation before deleting this run")
            store.delete_run(run_id)
    except ExecutionBusy as exc:
        raise HTTPException(409, str(exc))
    finally:
        run_lock.release()


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
        measured = (
            row_profile(
                run.get("profiling", {}), row.get("configuration_id"), row.get("question_index")
            )
            or {}
        )
        timings, usage = measured.get("timings", {}), measured.get("groq", {})
        rows.append(
            {
                **{k: v for k, v in row.items() if k not in {"scores", "contexts", "errors"}},
                "engine": row.get("engine", "existing"),
                "run_provenance": json.dumps(run.get("provenance", {}), ensure_ascii=False),
                "input_fingerprint": run.get("input_fingerprint"),
                **row["scores"],
                "contexts": json.dumps(row["contexts"], ensure_ascii=False),
                "errors": json.dumps(row["errors"]),
                "generation_seconds": timings.get("generation"),
                "scoring_seconds": timings.get("scoring"),
                **{
                    f"{metric}_seconds": measured.get("metrics", {})
                    .get(metric, {})
                    .get("timings", {})
                    .get("scoring")
                    for metric in METRICS
                },
                "groq_http_requests": usage.get("http_requests"),
                **{
                    f"groq_{field}_tokens": usage.get(f"{field}_tokens")
                    if usage.get(f"{field}_token_reports")
                    else None
                    for field in ("prompt", "completion", "total")
                },
            }
        )
    frame = pd.DataFrame(rows).map(spreadsheet_safe)
    return StreamingResponse(
        io.StringIO(frame.to_csv(index=False)),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="rag-bench-{run["id"]}.csv"'},
    )


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
        engines = {c["id"]: c.get("engine", "existing") for c in run["configurations"]}
        row.update(
            run_id=run_id,
            configuration_name=names.get(row.get("configuration_id")),
            engine=engines.get(row.get("configuration_id")),
            run_provenance=json.dumps(run.get("provenance", {})),
        )
    frame = pd.DataFrame(rows).map(spreadsheet_safe)
    return StreamingResponse(
        io.StringIO(frame.to_csv(index=False)),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="rag-bench-profile-{run_id}.csv"'},
    )
