"""Authenticated debugger endpoints using the workspace's single worker."""

import json
import threading
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend import debugger
from backend.execution_lock import ExecutionBusy, execution_lock
from backend.models import Configuration


class DebugRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_set_id: str
    configuration: Configuration
    generate: bool = False


class Action(BaseModel):
    mode: Literal["retrieve", "generate"] = "retrieve"
    context_fingerprint: str | None = None


def install(main):
    router = APIRouter(prefix="/api/debugger", dependencies=main.auth)

    def launch(make_record, generate=False, expected=None):
        if generate:
            main.require_generation()
        if not main.run_lock.acquire(blocking=False):
            raise HTTPException(409, "Workspace worker is busy")
        try:
            with execution_lock(main.store):
                if any(
                    r["status"] in {"queued", "running"}
                    for kind in ("run", "investigation", "optimization", "debug_run")
                    for r in main.store.list(kind)
                ):
                    raise HTTPException(409, "Workspace worker is busy")
                record = make_record()
                event = threading.Event()
                record.update(status="queued")
                main.store.save("debug_run", record, record["id"])
                with main.control_lock:
                    main.cancel_events[record["id"]] = event

            def worker():
                try:
                    debugger.execute(main.store, record["id"], event, generate, expected)
                except Exception:
                    # Atomic update only while still queued: never clobber another worker's commit.
                    patch = json.dumps(
                        {"status": "failed", "error": "Worker interrupted; resume explicitly"}
                    )
                    with main.store.connect() as db:
                        db.execute(
                            "UPDATE objects SET payload=payload || %s::jsonb WHERE id=%s AND kind='debug_run' AND payload->>'status'='queued'"
                            if main.store.postgres
                            else "UPDATE objects SET payload=json_patch(payload, ?) WHERE id=? AND kind='debug_run' AND json_extract(payload, '$.status')='queued'",
                            (patch, record["id"]),
                        )
                finally:
                    with main.control_lock:
                        main.cancel_events.pop(record["id"], None)
                    main.run_lock.release()

            try:
                from backend.identity import submit

                submit(main.executor, worker)
            except Exception:
                record.update(status="failed", error="Worker submission failed; resume explicitly")
                main.store.save("debug_run", record, record["id"])
                with main.control_lock:
                    main.cancel_events.pop(record["id"], None)
                raise
            return debugger.public(record)
        except (ExecutionBusy, ValueError, KeyError) as exc:
            main.run_lock.release()
            raise HTTPException(409, "Worker busy or immutable inputs unavailable") from exc
        except Exception:
            main.run_lock.release()
            raise

    @router.get("/inputs")
    def inputs():
        return {
            "document_sets": [
                {"id": d["id"], "sources": sorted({p["source"] for p in d["documents"]})}
                for d in main.store.list("documents", limit=100)
            ],
            "configurations": main.store.list("configuration", limit=100),
        }

    @router.post("", status_code=202)
    def start(body: DebugRequest):
        main.fetch("documents", body.document_set_id)
        return launch(
            lambda: debugger.create(
                main.store, body.question, body.document_set_id, body.configuration.model_dump()
            ),
            body.generate,
        )

    @router.get("")
    def history():
        return {
            "runs": [
                {k: r[k] for k in ("id", "question", "status", "created_at", "document_set_id")}
                for r in main.store.list("debug_run", limit=100)
            ]
        }

    @router.get("/historical/{run_id}/{configuration_id}/{question_index}")
    def historical(run_id: str, configuration_id: str, question_index: int):
        try:
            return debugger.historical(
                main.store, main.fetch("run", run_id), configuration_id, question_index
            )
        except (StopIteration, KeyError, ValueError) as exc:
            raise HTTPException(422, "Recorded evidence unavailable or inconsistent") from exc

    @router.post("/replay/{run_id}/{configuration_id}/{question_index}", status_code=202)
    def replay(run_id: str, configuration_id: str, question_index: int):
        evidence = historical(run_id, configuration_id, question_index)
        return launch(lambda: debugger.replay(main.store, evidence))

    @router.get("/{debug_id}/export")
    @router.get("/{debug_id}")
    def get(debug_id: str):
        record = debugger.public(main.fetch("debug_run", debug_id))
        if len(json.dumps(record).encode()) > 4 * 1024 * 1024:
            raise HTTPException(413, "Trace exceeds 4 MB export limit")
        return record

    @router.post("/{debug_id}/resume", status_code=202)
    def resume(debug_id: str, body: Action):
        def ready():
            record = main.fetch("debug_run", debug_id)
            if record["status"] not in {"paused", "failed"}:
                raise HTTPException(409, "Only paused or failed runs can resume")
            if record["attempt"] >= 4:
                raise HTTPException(409, "Retry budget exhausted; create a new run")
            if body.mode == "generate" and (
                record["selected"] is None
                or body.context_fingerprint != record["context_fingerprint"]
            ):
                raise HTTPException(409, "Saved context fingerprint required")
            debugger.validate(record)
            return record

        return launch(ready, body.mode == "generate", body.context_fingerprint)

    @router.post("/{debug_id}/cancel", status_code=202)
    def cancel(debug_id: str):
        record = main.fetch("debug_run", debug_id)
        if record["status"] not in {"queued", "running"}:
            raise HTTPException(409, "Run is not active")
        main.store.save(
            "debug_cancel", {"requested_at": debugger.timestamp()}, debug_id + "-cancel"
        )
        with main.control_lock:
            if event := main.cancel_events.get(debug_id):
                event.set()
        return {"cancel_requested": True}

    @router.get("/{left_id}/compare/{right_id}")
    def compare(left_id: str, right_id: str):
        try:
            return debugger.compare(
                main.fetch("debug_run", left_id), main.fetch("debug_run", right_id)
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.delete("/{debug_id}", status_code=204)
    def delete(debug_id: str):
        if not main.run_lock.acquire(blocking=False):
            raise HTTPException(409, "Workspace worker is busy")
        try:
            with execution_lock(main.store):
                record = main.fetch("debug_run", debug_id)
                if record["status"] in {"queued", "running"}:
                    raise HTTPException(409, "Cancel active work before deleting")
                from backend.graph_pipeline import checkpointer

                with checkpointer(main.store) as saver:
                    saver.delete_thread("debug-" + debug_id)
                with main.store.connect() as db:
                    db.execute(
                        "DELETE FROM objects WHERE id IN (%s, %s) AND kind IN ('debug_run', 'debug_cancel')"
                        if main.store.postgres
                        else "DELETE FROM objects WHERE id IN (?, ?) AND kind IN ('debug_run', 'debug_cancel')",
                        (debug_id, debug_id + "-cancel"),
                    )
        except ExecutionBusy as exc:
            raise HTTPException(409, "Workspace worker is busy") from exc
        finally:
            main.run_lock.release()

    main.app.include_router(router)
