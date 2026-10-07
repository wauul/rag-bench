import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.guardrails import POLICY_ID, locked, persist, policy_context, reserve, throttle
from backend.profiling import ProfiledAsyncClient, ProfiledClient
from backend.storage import Store


def amend(store, **changes):
    with locked(store) as (db, state, policy):
        state["policy"] = {**policy.model_dump(), **changes}
        persist(store, db, state)


def request():
    return httpx.Request("POST", "https://provider.test/completions", json={"max_tokens": 2048})


def test_atomic_shared_cap_and_restart(tmp_path):
    store = Store(tmp_path)
    amend(store, daily_requests=3)

    def attempt(_):
        try:
            with policy_context(store):
                reserve(request())
            return 1
        except ValueError:
            return 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(12))) == 3
    with policy_context(Store(tmp_path)), pytest.raises(ValueError, match="quota"):
        reserve(request())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_transport_failure_consumes_quota_and_switch_blocks_send(tmp_path, asynchronous):
    store = Store(tmp_path)
    amend(store, daily_requests=1)
    sent = []

    def transport(req):
        sent.append(req)
        raise httpx.ConnectError("fixture")

    async def async_run():
        async with ProfiledAsyncClient(transport=httpx.MockTransport(transport)) as client:
            with pytest.raises(httpx.ConnectError):
                await client.send(request())
            with pytest.raises(ValueError):
                await client.send(request())

    with policy_context(store):
        if asynchronous:
            asyncio.run(async_run())
        else:
            with ProfiledClient(transport=httpx.MockTransport(transport)) as client:
                with pytest.raises(httpx.ConnectError):
                    client.send(request())
                with pytest.raises(ValueError):
                    client.send(request())
    assert len(sent) == 1
    amend(store, enabled=False)
    with pytest.raises(ValueError, match="disabled"), policy_context(store):
        reserve(request())


def test_missing_policy_and_token_allowance_fail_closed(tmp_path):
    store = Store(tmp_path)
    amend(store, daily_token_allowance=1)
    with policy_context(store), pytest.raises(ValueError, match="quota"):
        reserve(request())
    assert store.get("guardrails", POLICY_ID)["usage"] == {}
    with store.connect() as db:
        db.execute("DELETE FROM objects WHERE id=?", (POLICY_ID,))
    with pytest.raises(ValueError, match="unavailable"), policy_context(store):
        reserve(request())


def test_limits_and_storage_are_durable(tmp_path):
    store = Store(tmp_path)
    amend(store, requests_per_minute=2, storage_bytes=1048576)
    assert throttle(store, "ip:fixture")
    assert throttle(store, "ip:fixture")
    assert not throttle(Store(tmp_path), "ip:fixture")
    with pytest.raises(ValueError, match="storage quota"):
        store.save("documents", {"text": "a" * 1048576})
    assert store.list("documents") == []


def test_default_authentication_and_ingress_fail_closed(tmp_path, monkeypatch):
    from backend import main

    store = Store(tmp_path)
    monkeypatch.setattr(main, "store", store)
    monkeypatch.delenv("RAGBENCH_ALLOW_INSECURE_LOCAL")
    monkeypatch.delenv("API_TOKEN", raising=False)
    client = TestClient(main.app)
    assert client.post("/api/demo").status_code == 503
    monkeypatch.setenv("API_TOKEN", "test-token")
    assert client.post("/api/documents", content=b"unparsed").status_code == 401
    client.headers["Authorization"] = "Bearer test-token"
    amend(store, enabled=False)
    assert client.post("/api/demo").status_code == 503
    amend(store, enabled=True, requests_per_minute=1)
    # New address remains subject to the workspace-wide counter too.
    assert client.get("/api/runs", headers={"X-Forwarded-For": "new-address"}).status_code == 429
    with store.connect() as db:
        db.execute("UPDATE objects SET payload=? WHERE id=?", (json.dumps({}), POLICY_ID))
    assert client.get("/api/runs").status_code == 503


def test_default_dashboard_denies_unconfigured_access(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("RAGBENCH_ALLOW_INSECURE_LOCAL")
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    app = AppTest.from_file(str(Path("dashboard/app.py").resolve()), default_timeout=20).run()
    assert not app.exception
    assert "Dashboard access is not configured" in app.error[0].value
    assert not app.get("file_uploader")


def test_purge_preserves_consumed_quota_and_requires_switch(tmp_path):
    from scripts.purge_workspace import purge

    store = Store(tmp_path)
    store.save("documents", {"text": "private"})
    with policy_context(store):
        reserve(request())
    usage = store.get("guardrails", POLICY_ID)["usage"]
    with pytest.raises(ValueError, match="Disable"):
        purge(store)
    amend(store, enabled=False)
    purge(store)
    assert store.list("documents") == []
    assert store.get("guardrails", POLICY_ID)["usage"] == usage


def test_pdf_timeout_is_terminated_and_slot_released(monkeypatch):
    import subprocess

    from backend import ingestion

    def expired(*args, **kwargs):
        assert kwargs["timeout"] == 25
        raise subprocess.TimeoutExpired(args[0], 25)

    monkeypatch.setattr(ingestion.subprocess, "run", expired)
    for _ in range(3):
        with pytest.raises(ValueError, match="timed out"):
            ingestion.extract_bounded("fixture.pdf", b"fixture")


def test_monthly_cap_survives_day_rollover(tmp_path):
    store = Store(tmp_path)
    amend(store, monthly_requests=1)
    with policy_context(store):
        reserve(request())
        with locked(store) as (db, state, _):
            state["usage"]["day"]["period"] = "2000-01-01"
            persist(store, db, state)
        with pytest.raises(ValueError, match="quota"):
            reserve(request())


def test_persistence_error_prevents_provider_send(tmp_path, monkeypatch):
    from backend import guardrails

    store = Store(tmp_path)
    sent = []
    with policy_context(store):

        def fail(*args):
            raise OSError("fixture database unavailable")

        monkeypatch.setattr(guardrails, "persist", fail)
        with ProfiledClient(transport=httpx.MockTransport(lambda req: sent.append(req))) as client:
            with pytest.raises(OSError):
                client.send(request())
    assert not sent


def test_disabled_switch_keeps_read_cancel_and_delete_available(tmp_path, monkeypatch):
    from backend import main

    store = Store(tmp_path)
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setenv("API_TOKEN", "test-only")
    run = store.save(
        "run", {"status": "completed", "rows": [], "configurations": [], "questions": []}
    )
    amend(store, enabled=False)
    client = TestClient(main.app, headers={"Authorization": "Bearer test-only"})
    assert client.get("/api/guardrails").status_code == 200
    assert client.post(f"/api/runs/{run['id']}/cancel").status_code == 409
    assert client.delete(f"/api/runs/{run['id']}").status_code == 204


def test_manual_retry_lifetime_limit():
    from backend.run_state import validate_retry

    with pytest.raises(ValueError, match="Retry budget exhausted"):
        validate_retry({"status": "failed", "attempt": 4})
