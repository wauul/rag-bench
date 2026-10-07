import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import alerts
from backend.identity import as_user, resolve, submit
from backend.retention import sweep
from backend.storage import Store
from scripts.users import provision


def test_ownership_and_worker_context(tmp_path):
    store = Store(tmp_path)
    with as_user("alice"):
        item = store.save("documents", {"text": "private"})
        with ThreadPoolExecutor() as executor:
            assert (
                submit(executor, store.get, "documents", item["id"]).result()["owner_id"] == "alice"
            )
    with as_user("bob"):
        assert store.list("documents") == []
        with pytest.raises(KeyError):
            store.get("documents", item["id"])
        with pytest.raises(KeyError):
            store.save("documents", {"text": "replace"}, item["id"])


def test_api_user_cannot_read_cancel_delete_other_run(tmp_path, monkeypatch):
    from backend import main

    store = Store(tmp_path)
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setenv("API_TOKEN", "owner-secret")
    provision(store, "bob", "b" * 32)
    with as_user("alice"):
        run = store.save(
            "run", {"status": "completed", "created_at": "2026-10-01", "configurations": []}
        )
    client = TestClient(main.app)
    headers = {"Authorization": "Bearer " + "b" * 32}
    assert client.get("/api/session", headers=headers).json()["user_id"] == "bob"
    for method, path in [("GET", "/api/runs/"), ("POST", "/api/runs/"), ("DELETE", "/api/runs/")]:
        suffix = "/cancel" if method == "POST" else ""
        assert client.request(method, path + run["id"] + suffix, headers=headers).status_code == 404
    assert client.get("/api/runs", headers=headers).status_code == 200


def test_expired_and_revoked_keys_fail_closed(tmp_path, monkeypatch):
    store = Store(tmp_path)
    monkeypatch.setenv("API_TOKEN", "owner-secret")
    key = "b" * 32
    provision(store, "bob", key)
    assert resolve(store, "Bearer " + key) == "bob"
    with store.connect(operator=True) as db:
        db.execute(
            "UPDATE users SET expires_at=?",
            ((datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),),
        )
    with pytest.raises(ValueError):
        resolve(store, "Bearer " + key)
    provision(store, "bob", key)
    with store.connect(operator=True) as db:
        db.execute("UPDATE users SET enabled=0")
    with pytest.raises(ValueError):
        resolve(store, "Bearer " + key)


def test_retention_preserves_recent_dependencies_and_quota(tmp_path):
    store = Store(tmp_path)
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=40)).isoformat()
    documents = store.save("documents", {"retained_at": old})
    expired = store.save("test_set", {"retained_at": old})
    with store.connect() as db:
        db.execute(
            "UPDATE objects SET payload=json_set(payload,'$.retained_at',?) WHERE id IN (?,?)",
            (old, documents["id"], expired["id"]),
        )
    store.save(
        "run",
        {"status": "completed", "document_set_id": documents["id"], "retained_at": now.isoformat()},
    )
    assert sweep(store, now) == 1
    assert store.get("documents", documents["id"])
    with pytest.raises(KeyError):
        store.get("test_set", expired["id"])
    assert store.get("guardrails", "workspace-guardrails")["usage"] == {}


def test_email_outbox_deduplicates_bounds_retries_and_sanitizes(tmp_path, monkeypatch):
    store = Store(tmp_path)
    for name, value in {
        "RESEND_API_KEY": "secret",
        "ALERT_FROM": "alerts@example.com",
        "ALERT_TO": "owner@example.com",
    }.items():
        monkeypatch.setenv(name, value)
    calls = []

    def send(request):
        calls.append(request)
        return httpx.Response(503)

    original = httpx.Client
    monkeypatch.setattr(
        alerts.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(send), **kw)
    )
    alerts.enqueue(store, "quota-blocked")
    alerts.enqueue(store, "quota-blocked")
    for _ in range(5):
        assert alerts.deliver(store) == 0
    assert len(calls) == 3
    assert len({r.headers["Idempotency-Key"] for r in calls}) == 1
    assert json.loads(calls[0].content)["to"] == ["owner@example.com"]
    assert len(store.list("alert")) == 1
    with pytest.raises(ValueError):
        alerts.enqueue(store, "user supplied content")


def test_email_missing_credentials_never_sends(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    store = Store(tmp_path)
    alerts.enqueue(store, "quota-warning")
    assert alerts.deliver(store) == 0
    assert store.list("alert")[0]["attempts"] == 0
