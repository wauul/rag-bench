"""Routes share benchmark authorization, admission and the single worker executor."""

import threading

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from backend.investigation_models import InvestigationRequest
from backend.investigator import configuration_draft, create_record, execute, markdown_report


def install(main):
    router = APIRouter(dependencies=main.auth)

    def submit(record):
        event = threading.Event()
        with main.control_lock:
            main.cancel_events[record["id"]] = event

        def worker():
            try:
                execute(main.store, record["id"], event)
            except Exception:
                from backend.execution_lock import ExecutionBusy, execution_lock

                try:
                    with execution_lock(main.store):
                        saved = main.store.get("investigation", record["id"])
                        if saved["status"] in {"queued", "running"}:
                            saved.update(
                                status="failed", error="Worker interrupted; resume saved stages"
                            )
                            main.store.save("investigation", saved, record["id"])
                except ExecutionBusy:
                    main.store.fail_queued_investigation(record["id"])
            finally:
                with main.control_lock:
                    main.cancel_events.pop(record["id"], None)
                    main.run_lock.release()

        try:
            main.executor.submit(worker)
        except Exception:
            with main.control_lock:
                main.cancel_events.pop(record["id"], None)
            record.update(status="failed", error="Worker submission failed")
            main.store.save("investigation", record, record["id"])
            raise

    @router.post("/api/runs/{run_id}/investigations", status_code=202)
    def start(run_id: str, body: InvestigationRequest):
        # Serialize dedup and admission across API processes using the storage lock.
        from backend.execution_lock import ExecutionBusy, execution_lock

        run = main.fetch("run", run_id)
        try:
            with execution_lock(main.store, "investigation-admission.lock"):
                for record in main.store.list("investigation"):
                    if record["run_id"] != run_id:
                        continue
                    same = (
                        record["configuration_id"] == body.configuration_id
                        and record["question_index"] == body.question_index
                    )
                    if record["request_key"] == body.request_key and not same:
                        raise HTTPException(409, "Request key belongs to another result")
                    if record["request_key"] == body.request_key or (
                        same and (not body.run_again or record["status"] in {"queued", "running"})
                    ):
                        return public(record)
                main.require_generation()
                try:
                    record = create_record(main.store, run, body)
                except (ValueError, StopIteration, IndexError, KeyError) as exc:
                    raise HTTPException(422, "Result or immutable evidence is unavailable") from exc
                if not main.run_lock.acquire(blocking=False):
                    raise HTTPException(409, "The workspace worker is busy; try when it finishes")
                try:
                    # Cross-process benchmark/investigator admission uses the same execution lock.
                    with execution_lock(main.store):
                        if any(
                            r["status"] in {"queued", "running"}
                            for r in main.store.list("run") + main.store.list("investigation")
                        ):
                            raise HTTPException(409, "The workspace worker is busy")
                        record["version"] = 1 + sum(
                            r["run_id"] == run_id
                            and r["configuration_id"] == body.configuration_id
                            and r["question_index"] == body.question_index
                            for r in main.store.list("investigation")
                        )
                        record = main.store.save("investigation", record)
                    submit(record)
                except Exception:
                    main.run_lock.release()
                    raise
                return public(record)
        except ExecutionBusy as exc:
            raise HTTPException(409, "The workspace worker is busy") from exc

    @router.get("/api/runs/{run_id}/investigations")
    def history(run_id: str, configuration_id: str = Query(), question_index: int = Query(ge=0)):
        main.fetch("run", run_id)
        return {
            "investigations": [
                public(r)
                for r in main.store.list("investigation")
                if r["run_id"] == run_id
                and r["configuration_id"] == configuration_id
                and r["question_index"] == question_index
            ]
        }

    @router.get("/api/investigations/{investigation_id}")
    def get(investigation_id: str):
        return public(main.fetch("investigation", investigation_id))

    @router.post("/api/investigations/{investigation_id}/cancel", status_code=202)
    def cancel(investigation_id: str):
        from backend.investigator import now

        record = main.fetch("investigation", investigation_id)
        if record["status"] not in {"queued", "running"}:
            raise HTTPException(409, "Investigation is not active")
        # Separate durable signal avoids overwriting the worker's in-flight model commit.
        main.store.save(
            "investigation_cancel",
            {"requested_at": now(), "run_id": record["run_id"]},
            investigation_id + "-cancel",
        )
        with main.control_lock:
            event = main.cancel_events.get(investigation_id)
            if event:
                event.set()
        return {"id": investigation_id, "cancel_requested": True}

    @router.post("/api/investigations/{investigation_id}/resume", status_code=202)
    def resume(investigation_id: str):
        from backend.execution_lock import ExecutionBusy, execution_lock

        record = main.fetch("investigation", investigation_id)
        if record["status"] != "failed":
            raise HTTPException(409, "Only interrupted or failed investigations can resume")
        if record.get("stage") == "persist_report" and record.get("validation_error"):
            raise HTTPException(409, "Repair budget exhausted; use Run again")
        main.require_generation()
        if not main.run_lock.acquire(blocking=False):
            raise HTTPException(409, "The workspace worker is busy")
        try:
            with execution_lock(main.store):
                # Another API process may have completed a resume since the initial read.
                record = main.fetch("investigation", investigation_id)
                if record["status"] != "failed":
                    raise HTTPException(409, "Investigation was already resumed")
                if any(
                    r["status"] in {"queued", "running"}
                    for r in main.store.list("run") + main.store.list("investigation")
                ):
                    raise HTTPException(409, "The workspace worker is busy")
                record.update(status="queued")
                record.pop("error", None)
                main.store.save("investigation", record, investigation_id)
            submit(record)
        except ExecutionBusy as exc:
            main.run_lock.release()
            raise HTTPException(409, "The workspace worker is busy") from exc
        except Exception:
            main.run_lock.release()
            raise
        return public(record)

    @router.post("/api/investigations/{investigation_id}/experiments/{index}/draft")
    def draft(investigation_id: str, index: int):
        record = main.fetch("investigation", investigation_id)
        if record["status"] != "completed" or not 0 <= index < len(record["experiments"]):
            raise HTTPException(409, "No completed experiment at this index")
        return {"configuration": configuration_draft(record, index), "launched": False}

    @router.get("/api/investigations/{investigation_id}/export")
    def export(investigation_id: str, format: str = Query("json", pattern="^(json|markdown)$")):
        record = public(main.fetch("investigation", investigation_id))
        if format == "markdown":
            return PlainTextResponse(markdown_report(record), media_type="text/markdown")
        return record

    main.app.include_router(router)


def public(record):
    # Preserve report evidence/provenance while omitting source corpus and raw model payloads.
    return {
        **{k: v for k, v in record.items() if k != "inputs"},
        "attempts": [
            {k: v for k, v in a.items() if k not in {"raw_content", "output"}}
            for a in record["attempts"]
        ],
    }
