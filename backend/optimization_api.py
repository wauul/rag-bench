"""Authenticated plan/review/start lifecycle; no provider work during planning."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.execution_lock import ExecutionBusy, execution_lock
from backend.models import Configuration
from backend.optimization import PlanRequest, create_plan, public, validate_plan


class Approval(BaseModel):
    plan_fingerprint: str


class Amendment(BaseModel):
    max_requests: int = Field(ge=1, le=10000)
    reason: str = Field(min_length=3, max_length=300)


class SaveCandidate(BaseModel):
    configuration_id: str
    name: str = Field(min_length=1, max_length=60)


def now():
    return datetime.now(timezone.utc).isoformat()


def install(main):
    router = APIRouter(prefix="/api/optimizations", dependencies=main.auth)

    @router.post("", status_code=201)
    def plan(body: PlanRequest):
        try:
            return public(create_plan(main.store, body))
        except KeyError as exc:
            raise HTTPException(404, "Dataset or baseline not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("")
    def history():
        return {"experiments": [public(r) for r in main.store.list("optimization", limit=50)]}

    @router.get("/inputs")
    def inputs():
        return {
            "documents": [
                {"id": d["id"], "name": d.get("name", "Document set"), "pages": len(d["documents"])}
                for d in main.store.list("documents")
            ],
            "test_sets": [
                {"id": t["id"], "name": t["name"], "count": len(t["questions"])}
                for t in main.store.list("test_set")
            ],
            "configurations": main.store.list("configuration"),
        }

    @router.get("/{experiment_id}")
    def get(experiment_id: str):
        record = main.fetch("optimization", experiment_id)
        if record["trials"]:
            try:
                active = main.store.get("run", record["trials"][-1]["run_id"])
                record["current_trial"] = {
                    key: active.get(key)
                    for key in ("id", "status", "stage", "completed", "scored", "total")
                }
            except KeyError:
                pass
        try:
            signal = main.store.get("optimization_cancel", experiment_id + "-cancel")
            if record["status"] in {"queued", "running"} and signal["attempt"] == record["attempt"]:
                record["status"] = "cancelling"
        except KeyError:
            pass
        return public(record)

    def worker(identity):
        from backend.optimization import execute_experiment

        try:
            execute_experiment(main.store, identity)
        except Exception as exc:
            # Admission/startup failures must not leave a queued experiment stuck.
            try:
                with execution_lock(main.store):
                    record = main.store.get("optimization", identity)
                    record.update(
                        status="failed", error=type(exc).__name__, stage="Explicit resume required"
                    )
                    main.store.save("optimization", record, identity)
            except ExecutionBusy:
                # Another workspace worker holds the lock, but cannot own this queued
                # experiment (admission committed before dispatch). Update only queued state.
                main.store.fail_queued_optimization(identity)
        finally:
            main.run_lock.release()

    def launch(identity, fingerprint, resume):
        main.require_generation()
        if not main.run_lock.acquire(blocking=False):
            raise HTTPException(409, "The workspace worker is busy")
        try:
            with execution_lock(main.store):
                record = main.fetch("optimization", identity)
                allowed = {"failed", "cancelled", "budget_exhausted"} if resume else {"planned"}
                if record["status"] not in allowed:
                    raise HTTPException(409, "Experiment cannot start from its current state")
                if fingerprint != record["plan_fingerprint"]:
                    raise HTTPException(409, "Review the current frozen plan before starting")
                try:
                    validate_plan(record)
                except ValueError as exc:
                    raise HTTPException(409, str(exc)) from exc
                if record["usage"]["requests_reserved"] >= record["max_requests"]:
                    raise HTTPException(
                        409, "Request budget exhausted; explicitly amend it before resuming"
                    )
                if any(
                    r["status"] in {"queued", "running"}
                    for kind in ("run", "investigation", "optimization")
                    for r in main.store.list(kind)
                ):
                    raise HTTPException(409, "The workspace worker is busy")
                record.update(
                    status="queued",
                    approved_at=record.get("approved_at", now()),
                    attempt=record["attempt"] + 1,
                )
                main.store.save("optimization", record, identity)
            try:
                from backend.identity import submit

                submit(main.executor, worker, identity)
            except Exception:
                main.store.fail_queued_optimization(identity)
                raise
        except ExecutionBusy as exc:
            main.run_lock.release()
            raise HTTPException(409, "The workspace worker is busy") from exc
        except Exception:
            main.run_lock.release()
            raise
        return public(record)

    @router.post("/{experiment_id}/start", status_code=202)
    def start(experiment_id: str, body: Approval):
        return launch(experiment_id, body.plan_fingerprint, False)

    @router.post("/{experiment_id}/resume", status_code=202)
    def resume(experiment_id: str, body: Approval):
        return launch(experiment_id, body.plan_fingerprint, True)

    @router.post("/{experiment_id}/cancel", status_code=202)
    def cancel(experiment_id: str):
        record = main.fetch("optimization", experiment_id)
        if record["status"] not in {"queued", "running"}:
            raise HTTPException(409, "Experiment is not active")
        main.store.save(
            "optimization_cancel",
            {"attempt": record["attempt"], "requested_at": now()},
            experiment_id + "-cancel",
        )
        return {
            "status": "cancelling",
            "message": "Stopping new work; an in-flight request may finish",
        }

    @router.post("/{experiment_id}/budget")
    def amend(experiment_id: str, body: Amendment):
        try:
            with execution_lock(main.store):
                record = main.fetch("optimization", experiment_id)
                if (
                    record["status"] not in {"failed", "cancelled", "budget_exhausted"}
                    or body.max_requests <= record["max_requests"]
                ):
                    raise HTTPException(
                        409, "Only stopped experiments can receive an explicit budget increase"
                    )
                record["amendments"].append(
                    {"at": now(), "previous": record["max_requests"], **body.model_dump()}
                )
                record["max_requests"] = body.max_requests
                main.store.save("optimization", record, experiment_id)
                return public(record)
        except ExecutionBusy as exc:
            raise HTTPException(409, "The workspace worker is busy") from exc

    @router.get("/{experiment_id}/export")
    def export(experiment_id: str):
        return public(main.fetch("optimization", experiment_id))

    @router.post("/{experiment_id}/configuration", status_code=201)
    def save_candidate(experiment_id: str, body: SaveCandidate):
        record = main.fetch("optimization", experiment_id)
        observed = {
            t["configuration_id"] for t in record.get("tuning_results", []) if t["eligible"]
        }
        if body.configuration_id not in observed:
            raise HTTPException(409, "Choose a candidate with complete observed tuning results")
        configs = [record["plan"]["baseline"], *record["plan"]["candidates"]]
        config = next(c for c in configs if c["id"] == body.configuration_id)
        return main.store.save(
            "configuration", Configuration(**{**config, "name": body.name}).model_dump()
        )

    main.app.include_router(router)
