"""Temporary private data only: consistent snapshots, restore/recovery and security."""

from threading import Event

import pytest
from fastapi.testclient import TestClient
from graph_fakes import create_run, install

from backend.execution_lock import execution_lock
from backend.settings import validate_operations
from backend.storage import Store
from scripts.backup import backup, restore


def test_production_fails_closed(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="API_TOKEN"):
        validate_operations()
    monkeypatch.setenv("API_TOKEN", "x" * 32)
    monkeypatch.setenv("DATA_STORAGE", "ephemeral")
    with pytest.raises(ValueError, match="persistent"):
        validate_operations()
    monkeypatch.setenv("DATA_STORAGE", "persistent")
    validate_operations()


def test_streaming_upload_and_validation_redaction(tmp_path, monkeypatch):
    from backend import main

    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setenv("API_TOKEN", "temporary-test-token")
    client = TestClient(main.app)
    assert client.get("/ready").status_code == 401
    client.headers["Authorization"] = "Bearer temporary-test-token"
    assert client.get("/ready").status_code == 200
    response = client.post(
        "/api/test-sets", json={"questions": [{"question": "private-value", "reference": "x"}]}
    )
    assert response.status_code == 422
    assert "private-value" not in response.text and '"input"' not in response.text
    response = client.post("/api/documents", content=iter([b"a" * 1024 * 1024] * 13))
    assert response.status_code == 413


def test_snapshot_restore_resumes_graph_without_repeating_answer(tmp_path, monkeypatch):
    from backend.pipeline import execute_run

    for key in ("GROQ_MODEL", "INFERENCE_BACKEND", "JUDGE_EXAMPLES"):
        monkeypatch.delenv(key, raising=False)
    source = tmp_path / "data"
    store = Store(source)
    run = create_run(store, ("langgraph",), questions=2)
    event = Event()
    calls, _ = install(monkeypatch, event=event)
    interrupted = execute_run(store, run["id"], cancel_event=event)
    assert interrupted["status"] == "cancelled"
    assert calls["generation"] == 1
    snapshot = tmp_path / "snapshot"
    backup(source, snapshot)
    restored = tmp_path / "restored"
    restore(snapshot, restored)
    event.clear()
    result = execute_run(Store(restored), run["id"], retry=True)
    assert result["status"] == "completed", result.get("error")
    assert calls["generation"] == 2
    assert result["rows"][0]["answer"] == interrupted["rows"][0]["answer"]
    with pytest.raises(ValueError, match="NEW"):
        restore(snapshot, restored)
    (snapshot / "bench.sqlite3").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        restore(snapshot, tmp_path / "corrupt-restore")


def test_backup_refuses_live_api(tmp_path):
    # A separate process owns server.lock; thread-local lock reentrancy is deliberate.
    import subprocess
    import sys

    store = Store(tmp_path / "data")
    with execution_lock(store, "server.lock"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.backup",
                "backup",
                str(store.root),
                str(tmp_path / "snapshot"),
            ],
            capture_output=True,
        )
    assert result.returncode != 0
    assert b"ExecutionBusy" in result.stderr
