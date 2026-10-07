"""Real PostgreSQL tests; set RAGBENCH_TEST_POSTGRES_URL to a disposable local cluster."""

import os
import subprocess
import sys
from collections import Counter
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from backend.execution_lock import execution_lock
from backend.postgres import validate_database_url
from backend.storage import Store


@pytest.fixture
def pg_url():
    url = os.getenv("RAGBENCH_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Disposable PostgreSQL not configured")
    options = conninfo_to_dict(url)
    if options.get("host") not in {"127.0.0.1", "localhost"}:
        pytest.fail("These tests require a disposable localhost PostgreSQL cluster")
    name = "test_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    # Store accepts URLs, not libpq keyword strings.
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    result = urlunsplit(parts._replace(path="/" + name))
    try:
        yield result
    finally:
        with psycopg.connect(url, autocommit=True) as db:
            db.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def test_remote_url_requires_tls_and_direct_endpoint():
    for bad in (
        "sqlite:///data",
        "postgresql://u:p@ep-test.neon.tech/db",
        "postgresql://u:p@ep-test-pooler.neon.tech/db?sslmode=require",
    ):
        with pytest.raises(ValueError):
            validate_database_url(bad)
    validate_database_url("postgresql://u:p@ep-test.neon.tech/db?sslmode=require")


def test_crud_history_and_local_default(pg_url, tmp_path, monkeypatch):
    store = Store(tmp_path / "one", database_url=pg_url)
    saved = store.save("documents", {"documents": [{"text": "UTF-8 café 漢字"}]})
    fresh = Store(tmp_path / "empty", database_url=pg_url)
    assert fresh.get("documents", saved["id"]) == saved
    run = fresh.save(
        "run",
        {
            "status": "completed",
            "created_at": "2026-10-07",
            "configurations": [],
            "rows": [{"answer": "large"}],
        },
    )
    fresh.save("run", {**run, "status": "failed"}, run["id"])
    assert len(fresh.list("run")) == 1
    history = fresh.run_history()
    assert history[0]["status"] == "failed" and "rows" not in history[0]
    assert fresh.list("run", limit=1, offset=1) == []
    with pytest.raises(KeyError):
        fresh.get("configuration", saved["id"])
    with pytest.raises(ValueError):
        fresh.save("run", {"score": float("nan")})
    monkeypatch.setenv("DATABASE_URL", pg_url)
    assert not Store(tmp_path / "explicit-local").postgres


def test_database_lock_spans_different_local_roots(pg_url, tmp_path):
    store = Store(tmp_path / "one", database_url=pg_url)
    code = """
import os, sys
from backend.storage import Store
from backend.execution_lock import execution_lock, ExecutionBusy
try:
    with execution_lock(Store(sys.argv[1], database_url=os.environ['TEST_DATABASE_URL'])):
        pass
except ExecutionBusy:
    sys.exit(23)
"""
    env = {**os.environ, "TEST_DATABASE_URL": pg_url}

    def child():
        return subprocess.run(
            [sys.executable, "-c", code, str(tmp_path / "two")],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    with execution_lock(store), execution_lock(store):
        assert child().returncode == 23
    assert child().returncode == 0


def test_lost_lock_session_cannot_write_through_new_connection(pg_url, tmp_path):
    from backend.execution_lock import execution_connection

    store = Store(tmp_path, database_url=pg_url)
    with execution_lock(store):
        connection = execution_connection(store)
        pid = connection.execute("SELECT pg_backend_pid()").fetchone()[0]
        with psycopg.connect(pg_url, autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        with pytest.raises(psycopg.Error):
            store.save("run", {"status": "incorrect overwrite"}, "lost-session")
    with pytest.raises(KeyError):
        store.get("run", "lost-session")


@pytest.mark.parametrize(
    "crash", ["before_call", "after_answer_save", "between_metrics", "indexing"]
)
def test_fresh_process_empty_cache_recovery(pg_url, tmp_path, crash):
    from graph_fakes import create_run

    from backend.graph_pipeline import checkpointer, thread_id

    store = Store(tmp_path / "first", database_url=pg_url)
    run = create_run(store, ("langgraph",), questions=2)
    code = """
import os, sys
sys.path.insert(0, 'tests')
from pytest import MonkeyPatch
from graph_fakes import install
from backend.storage import Store
from backend.pipeline import execute_run
store = Store(sys.argv[1], database_url=os.environ['TEST_DATABASE_URL'])
install(MonkeyPatch(), log_path=sys.argv[3], crash=sys.argv[4] or None)
result = execute_run(store, sys.argv[2], retry=not bool(sys.argv[4]))
assert result['status'] == 'completed', result.get('error')
"""
    log = tmp_path / "calls.txt"
    env = {**os.environ, "TEST_DATABASE_URL": pg_url}
    for key in ("GROQ_MODEL", "INFERENCE_BACKEND", "GROQ_REQUEST_INTERVAL", "JUDGE_EXAMPLES"):
        env.pop(key, None)
    first = subprocess.run(
        [sys.executable, "-c", code, str(store.root), run["id"], str(log), crash],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert first.returncode == 73, first.stdout + first.stderr
    before = store.get("run", run["id"])
    resumed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "fresh-empty"), run["id"], str(log), ""],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = store.get("run", run["id"])
    calls = Counter(log.read_text().splitlines())
    assert result["status"] == "completed" and calls["generation"] == 2
    from backend.models import METRICS

    assert all(calls[m] == 2 for m in METRICS)
    for row in before["rows"]:
        if row.get("answer"):
            assert result["rows"][row["question_index"]]["answer"] == row["answer"]
    with checkpointer(store) as saver:
        identity = {"configurable": {"thread_id": thread_id(run["id"], "0", 0)}}
        assert saver.get_tuple(identity) is not None
    with execution_lock(store):
        store.delete_run(run["id"])
    with checkpointer(store) as saver:
        assert saver.get_tuple(identity) is None
    assert store.list("snapshot") == [] and store.list("graph_artifact") == []


def test_remote_cancellation_and_retry(pg_url, tmp_path, monkeypatch):
    from graph_fakes import create_run, install

    from backend.pipeline import execute_run
    from scripts.retry_failed import recover

    store = Store(tmp_path, database_url=pg_url)
    run = create_run(store, ("langgraph",))
    calls, _ = install(monkeypatch)
    store.save("cancellation", {"run_id": run["id"], "attempt": 1}, run["id"] + "-cancel")
    assert execute_run(store, run["id"])["status"] == "cancelled"
    assert not calls
    recover(store, run["id"])
    assert store.get("run", run["id"])["status"] == "completed"
    assert calls["generation"] == 1


def test_offline_sqlite_migration_preserves_checkpoints(pg_url, tmp_path, monkeypatch):
    from threading import Event

    from graph_fakes import create_run, install

    from backend.graph_pipeline import checkpointer, thread_id
    from backend.pipeline import execute_run
    from scripts.migrate_postgres import migrate
    from scripts.retry_failed import recover

    source = Store(tmp_path / "sqlite")
    run = create_run(source, ("langgraph",))
    event = Event()
    calls, _ = install(monkeypatch, event=event)
    execute_run(source, run["id"], cancel_event=event)
    report = migrate(source.root, pg_url)
    assert report["objects"] > 0 and report["checkpoints"] > 0
    remote = Store(tmp_path / "empty", database_url=pg_url)
    assert remote.get("run", run["id"])["rows"] == source.get("run", run["id"])["rows"]
    with checkpointer(remote) as saver:
        assert saver.get_tuple({"configurable": {"thread_id": thread_id(run["id"], "0", 0)}})
    event.clear()
    recover(remote, run["id"])
    assert calls["generation"] == 1
    assert remote.get("run", run["id"])["status"] == "completed"
    with pytest.raises(ValueError, match="empty"):
        migrate(source.root, pg_url)
