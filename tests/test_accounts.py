import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from backend import accounts
from backend.identity import as_user
from backend.storage import Store


@pytest.fixture
def account_client(tmp_path, monkeypatch):
    from backend import main

    store = Store(tmp_path)
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setenv("API_TOKEN", "o" * 32)
    monkeypatch.setenv("NEON_AUTH_URL", "https://test.neonauth.neon.tech/neondb/auth")
    monkeypatch.setenv("AUTH_GATEWAY_URL", "https://ragbench.example")
    monkeypatch.setenv("DASHBOARD_URL", "https://ragbench.streamlit.app")
    return TestClient(main.app), store


def start(client, verifier="v" * 32):
    return client.post(
        "/api/auth/start",
        headers={"Authorization": "Bearer " + "o" * 32},
        json={"challenge": accounts.digest(verifier)},
    ).json()["flow"]


def test_flow_requires_authenticated_start_and_private_verifier(account_client):
    client, store = account_client
    assert client.post("/api/auth/start", json={"challenge": "f" * 64}).status_code == 401
    flow = start(client)
    response = client.get("/auth/login", params={"flow": flow})
    assert response.status_code == 200
    assert (
        "HttpOnly" in response.headers["set-cookie"] and "Secure" in response.headers["set-cookie"]
    )
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    # CSP paths without a trailing slash match only that exact path, blocking
    # /sign-in/social, /get-session and /token beneath the managed Auth API.
    assert (
        "connect-src 'self' https://test.neonauth.neon.tech/neondb/auth/;"
        in response.headers["content-security-policy"]
    )
    assert (
        client.post(
            "/auth/redeem", json={"flow": flow, "verifier": "x" * 32, "ticket": "t" * 32}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/auth/redeem", json={"flow": flow, "verifier": "v" * 32, "ticket": "t" * 32}
        ).status_code
        == 403
    )
    other = TestClient(client.app)
    assert other.get("/auth/login", params={"flow": flow}).status_code == 400
    with store.connect(operator=True) as db:
        db.execute("UPDATE account_flows SET expires_at=0")
    assert (
        client.post(
            "/auth/redeem", json={"flow": flow, "verifier": "v" * 32, "ticket": "t" * 32}
        ).status_code
        == 400
    )


def test_identity_proof_and_origin_fail_closed(account_client, monkeypatch):
    client, _ = account_client
    flow = start(client)
    assert (
        client.post(
            "/auth/finish", json={"flow": flow}, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/auth/finish", json={"flow": flow}, headers={"Origin": "https://ragbench.example"}
        ).status_code
        == 401
    )
    monkeypatch.setattr(
        accounts, "verified_subject", lambda _: (_ for _ in ()).throw(TimeoutError())
    )
    assert (
        client.post(
            "/auth/finish",
            json={"flow": flow},
            headers={"Origin": "https://ragbench.example", "Authorization": "Bearer invalid"},
        ).status_code
        == 401
    )


def test_password_reset_page_does_not_depend_on_dashboard_flow(account_client):
    client, store = account_client
    with store.connect(operator=True) as db:
        db.execute("DELETE FROM account_flows")
    response = client.get("/auth/reset", params={"token": "provider-controlled-token"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert '"flow": ""' in response.text
    assert "redirectTo:config.origin+'/auth/reset'" in response.text
    assert "history.replaceState(null,'',config.origin+'/auth/reset')" in response.text
    assert "provider-controlled-token" not in response.text
    assert "set-cookie" not in response.headers


def test_single_use_redemption_stores_only_hash_and_revokes(account_client, monkeypatch):
    client, store = account_client
    flow = start(client)
    monkeypatch.setattr(accounts, "verified_user", lambda db, subject: None)
    with store.connect(operator=True) as db:
        db.execute(
            "INSERT INTO account_links(subject,user_id) VALUES (?,?)", ("verified-account", "alice")
        )
        db.execute(
            "UPDATE account_flows SET subject=?,ticket_hash=? WHERE id=?",
            ("verified-account", accounts.digest("t" * 32), flow),
        )
    body = {"flow": flow, "verifier": "v" * 32, "ticket": "t" * 32}
    result = client.post("/auth/redeem", json=body)
    assert result.status_code == 200
    token = result.json()["token"]
    assert token.startswith("rb_session_") and result.json()["user_id"] == "alice"
    with store.connect(operator=True) as db:
        row = db.execute("SELECT token_hash,expires_at FROM account_sessions").fetchone()
        assert row[0] == accounts.digest(token) and token not in str(row)
        assert 3500 < row[1] - time.time() <= 3600
    assert client.post("/auth/redeem", json=body).status_code == 400
    # SQLite intentionally cannot authorize production managed accounts.
    with pytest.raises(ValueError):
        accounts.resolve(store, token)
    monkeypatch.setattr(accounts, "resolve", lambda store, value: "alice")
    assert (
        client.post("/api/auth/logout", headers={"Authorization": "Bearer " + token}).status_code
        == 200
    )
    with store.connect(operator=True) as db:
        assert db.execute("SELECT COUNT(*) FROM account_sessions").fetchone()[0] == 0


def test_disabled_account_cannot_redeem(account_client, monkeypatch):
    client, store = account_client
    flow = start(client)
    monkeypatch.setattr(accounts, "verified_user", lambda db, subject: None)
    with store.connect(operator=True) as db:
        db.execute(
            "INSERT INTO account_links(subject,user_id,enabled) VALUES ('account','alice',0)"
        )
        db.execute(
            "UPDATE account_flows SET subject='account',ticket_hash=?", (accounts.digest("t" * 32),)
        )
    # Failure never returns a session, even when operator persistence is unavailable.
    assert (
        client.post(
            "/auth/redeem", json={"flow": flow, "verifier": "v" * 32, "ticket": "t" * 32}
        ).status_code
        == 401
    )


def test_jwt_signature_issuer_audience_expiration_and_subject(monkeypatch):
    provider = "https://test.neonauth.neon.tech/neondb/auth"
    monkeypatch.setenv("NEON_AUTH_URL", provider)
    monkeypatch.setenv("AUTH_GATEWAY_URL", "https://ragbench.example")
    monkeypatch.setenv("DASHBOARD_URL", "https://ragbench.streamlit.app")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(
        accounts,
        "key_client",
        lambda _: SimpleNamespace(
            get_signing_key_from_jwt=lambda token: SimpleNamespace(key=key.public_key())
        ),
    )
    claims = {
        "sub": "user-a",
        "iss": provider,
        "aud": provider,
        "iat": int(time.time()),
        "exp": int(time.time()) + 900,
    }
    token = jwt.encode(claims, key, algorithm="RS256")
    assert accounts.verified_subject(token) == "user-a"
    for patch in [
        {"iss": "https://evil.example"},
        {"aud": "other-app"},
        {"exp": 1},
        {"sub": "../owner"},
    ]:
        with pytest.raises((ValueError, jwt.PyJWTError)):
            accounts.verified_subject(jwt.encode({**claims, **patch}, key, algorithm="RS256"))
    with pytest.raises(jwt.PyJWTError):
        accounts.verified_subject(jwt.encode(claims, "untrusted" * 8, algorithm="HS256"))
    monkeypatch.setattr(accounts, "key_client", lambda _: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):
        accounts.verified_subject(token)


def test_account_api_keys_cannot_access_other_users(account_client):
    client, store = account_client
    from scripts.users import provision

    provision(store, "alice", "a" * 32)
    provision(store, "bob", "b" * 32)
    with as_user("bob"):
        private = store.save("run", {"status": "completed", "configurations": []})
    headers = {"Authorization": "Bearer " + "a" * 32}
    key = client.post("/api/access-key", headers=headers).json()["key"]
    headers = {"Authorization": "Bearer " + key}
    assert client.get("/api/runs/" + private["id"], headers=headers).status_code == 404
    assert client.delete("/api/access-key", headers=headers).status_code == 200
    assert client.get("/api/session", headers=headers).status_code == 401


def test_public_auth_throttle_denies_on_error(account_client, monkeypatch):
    from backend import guardrails

    client, _ = account_client
    monkeypatch.setattr(guardrails, "throttle", lambda *args: (_ for _ in ()).throw(OSError()))
    assert client.get("/auth/login?flow=" + "a" * 43).status_code == 503
