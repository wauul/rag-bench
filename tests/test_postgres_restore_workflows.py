"""A stopped PostgreSQL snapshot restores real paused work without automatic spending."""

from threading import Event
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from backend import debugger, investigator, optimization
from backend.storage import Store
from scripts.postgres_backup import backup, restore
from tests import test_investigator, test_optimization
from tests.graph_fakes import create_run, install
from tests.test_postgres import pg_url as postgres_url_fixture


@pytest.fixture
def pg_url():
    yield from postgres_url_fixture.__wrapped__()


def test_real_paused_workflows_restore_then_explicit_resume(pg_url, tmp_path, monkeypatch):
    def remote(root):
        return Store(root, database_url=pg_url)

    calls, _ = install(monkeypatch)
    store = remote(tmp_path / "source")
    run = create_run(store, ("langgraph",))
    debug = debugger.create(store, "Question 0?", run["document_set_id"], run["configurations"][0])
    paused = debugger.execute(store, debug["id"], Event())
    assert paused["status"] == "paused" and calls["generation"] == 0
    monkeypatch.setattr(test_investigator, "Store", remote)
    inv_store, _, inv = test_investigator.fixture.__wrapped__(tmp_path / "inv")
    event = Event()

    def model_call(record):
        event.set()
        return {"output": test_investigator.diagnosis(), "usage": {"total_tokens": 15}}

    interrupted = investigator.execute(inv_store, inv["id"], event, model_call=model_call)
    assert interrupted["status"] == "cancelled" and len(interrupted["attempts"]) == 1

    monkeypatch.setattr(test_optimization, "Store", remote)
    opt_store, request = test_optimization.setup.__wrapped__(tmp_path / "opt", monkeypatch)
    plan = test_optimization.approve(opt_store, optimization.create_plan(opt_store, request))
    test_optimization.providers(monkeypatch, opt_store, plan["id"], cancel_at=1)
    interrupted_opt = optimization.execute_experiment(opt_store, plan["id"])
    assert (
        interrupted_opt["status"] == "cancelled"
        and interrupted_opt["usage"]["requests_reserved"] == 1
    )
    name = "restored_" + uuid4().hex
    with psycopg.connect(pg_url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    destination = urlunsplit(urlsplit(pg_url)._replace(path="/" + name))
    try:
        snapshot = tmp_path / "snapshot"
        backup(pg_url, snapshot)
        restore(snapshot, destination)
        fresh = Store(tmp_path / "restored-cache", database_url=destination)
        # Restore and opening storage never execute provider work.
        assert fresh.get("debug_run", debug["id"])["answer"] is None
        assert len(fresh.get("investigation", inv["id"])["attempts"]) == 1
        assert fresh.get("optimization", plan["id"])["usage"]["requests_reserved"] == 1
        completed_inv = investigator.execute(
            fresh, inv["id"], model_call=lambda _: pytest.fail("Repeated saved provider response")
        )
        assert completed_inv["status"] == "completed" and len(completed_inv["attempts"]) == 1
        install(monkeypatch)
        completed_debug = debugger.execute(
            fresh, debug["id"], Event(), True, paused["context_fingerprint"]
        )
        assert completed_debug["status"] == "completed"
        # Explicit resume creates a new attempt; the old cancellation remains audit evidence.
        restored_opt = fresh.get("optimization", plan["id"])
        restored_opt["attempt"] += 1
        fresh.save("optimization", restored_opt, plan["id"])
        test_optimization.providers(monkeypatch, fresh, plan["id"])
        completed_opt = optimization.execute_experiment(fresh, plan["id"])
        assert completed_opt["status"] == "completed", completed_opt.get("error")
        assert completed_opt["usage"]["requests_reserved"] <= completed_opt["max_requests"]
    finally:
        with psycopg.connect(pg_url, autocommit=True) as db:
            db.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
