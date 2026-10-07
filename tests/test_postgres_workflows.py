"""All durable workflow kinds and checkpoints survive a PostgreSQL dump/restore."""

from threading import Event

import pytest

from backend.storage import Store
from scripts.postgres_backup import backup, restore
from tests.test_postgres import pg_url as postgres_url_fixture


@pytest.fixture
def pg_url(request):
    yield from postgres_url_fixture.__wrapped__()


def test_postgres_workflows_resume(pg_url, tmp_path, monkeypatch):
    from backend import debugger, investigator, optimization
    from tests import test_debugger, test_investigator, test_optimization

    def remote(root):
        return Store(root, database_url=pg_url)

    monkeypatch.setattr(test_debugger, "Store", remote)
    store, run, record, calls, _ = test_debugger.setup.__wrapped__(tmp_path / "debug", monkeypatch)
    paused = debugger.execute(store, record["id"], Event())
    assert paused["status"] == "paused", paused.get("error")
    fresh = Store(tmp_path / "fresh", database_url=pg_url)
    completed = debugger.execute(fresh, record["id"], Event(), True, paused["context_fingerprint"])
    assert completed["status"] == "completed" and calls["generation"] == 1
    assert not fresh.get("run", run["id"])["rows"]

    monkeypatch.setattr(test_investigator, "Store", remote)
    inv_store, _, inv = test_investigator.fixture.__wrapped__(tmp_path / "inv")
    result = investigator.execute(
        inv_store,
        inv["id"],
        model_call=lambda _: {
            "output": test_investigator.diagnosis(),
            "usage": {"total_tokens": 15},
        },
    )
    assert result["status"] == "completed", result.get("error")
    assert len(result["attempts"]) == 1
    again = investigator.execute(
        fresh, inv["id"], model_call=lambda _: pytest.fail("repeated model call")
    )
    assert again["status"] == "completed"

    monkeypatch.setattr(test_optimization, "Store", remote)
    opt_store, request = test_optimization.setup.__wrapped__(tmp_path / "opt", monkeypatch)
    plan = optimization.create_plan(opt_store, request)
    approved = test_optimization.approve(opt_store, plan)
    result = optimization.execute_experiment(opt_store, approved["id"])
    assert result["status"] == "completed", result.get("error")
    assert (
        result["selection"] == optimization.execute_experiment(fresh, approved["id"])["selection"]
    )


def test_dump_restore_preserves_all_records_and_checkpoints(pg_url, tmp_path, monkeypatch):
    from backend.graph_pipeline import checkpointer, thread_id
    from backend.pipeline import execute_run
    from tests.graph_fakes import create_run, install

    store = Store(tmp_path / "source", database_url=pg_url)
    run = create_run(store, ("langgraph",), questions=2)
    event = Event()
    calls, _ = install(monkeypatch, event=event)
    result = execute_run(store, run["id"], event)
    assert result["status"] == "cancelled"
    for kind in (
        "investigation",
        "optimization",
        "debug_run",
        "cancellation",
        "debug_cancel",
        "optimization_cancel",
        "investigation_cancel",
    ):
        store.save(kind, {"fixture": kind}, kind + "-fixture")
    # A second disposable database using the same validated localhost connection.
    from urllib.parse import urlsplit, urlunsplit
    from uuid import uuid4

    import psycopg
    from psycopg import sql

    name = "restore_" + uuid4().hex
    with psycopg.connect(pg_url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    restored_url = urlunsplit(urlsplit(pg_url)._replace(path="/" + name))
    try:
        snapshot = tmp_path / "dump"
        backup(pg_url, snapshot)
        restore(snapshot, restored_url)
        restored = Store(tmp_path / "restored", database_url=restored_url)
        for kind in (
            "investigation",
            "optimization",
            "debug_run",
            "cancellation",
            "debug_cancel",
            "optimization_cancel",
            "investigation_cancel",
        ):
            assert restored.get(kind, kind + "-fixture")["fixture"] == kind
        with checkpointer(restored) as saver:
            assert saver.get_tuple({"configurable": {"thread_id": thread_id(run["id"], "0", 0)}})
        event.clear()
        recovered = execute_run(restored, run["id"], retry=True)
        assert recovered["status"] == "completed" and calls["generation"] == 2
        with pytest.raises(ValueError, match="empty"):
            restore(snapshot, restored_url)
        (snapshot / "database.dump").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum"):
            restore(snapshot, restored_url)
    finally:
        with psycopg.connect(pg_url, autocommit=True) as db:
            db.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
