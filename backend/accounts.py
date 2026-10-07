"""Managed identity verification and short-lived, hashed dashboard sessions.

The browser proves identity to Neon. Only a private Streamlit verifier can redeem
the result. Neither provider JWTs nor user passwords are stored by Ragbench.
"""

import hashlib
import json
import os
import re
import secrets
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import jwt
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def configuration():
    provider = os.getenv("NEON_AUTH_URL", "").rstrip("/")
    origin = os.getenv("AUTH_GATEWAY_URL", "").rstrip("/")
    issuer = os.getenv("NEON_AUTH_ISSUER") or provider
    audience = os.getenv("NEON_AUTH_AUDIENCE") or provider
    for value in (provider, origin):
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Account authentication is not configured")
    if not urlsplit(provider).hostname.endswith(".neon.tech"):
        raise ValueError("Untrusted identity provider")
    return provider, origin, issuer, audience


@lru_cache(maxsize=2)
def key_client(provider):
    return jwt.PyJWKClient(provider + "/.well-known/jwks.json", lifespan=300, timeout=5)


def verified_subject(token):
    provider, _, issuer, audience = configuration()
    if len(token) > 16384:
        raise ValueError("Invalid identity token")
    key = key_client(provider).get_signing_key_from_jwt(token).key
    claims = jwt.decode(
        token,
        key,
        algorithms=["RS256", "ES256", "EdDSA"],
        issuer=issuer,
        audience=audience,
        options={"require": ["exp", "iat", "sub", "iss", "aud"]},
    )
    subject = claims["sub"]
    if not isinstance(subject, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", subject):
        raise ValueError("Invalid account identity")
    return subject


def verified_user(db, subject):
    # Auth state is authoritative, not user-editable JWT profile metadata.
    row = db.execute(
        'SELECT "emailVerified", banned FROM neon_auth."user" WHERE id=%s',
        (subject,),
    ).fetchone()
    if not row or row[0] is not True or row[1] is True:
        raise ValueError("A verified, active account is required")


def initialize(store):
    with store.connect(operator=True) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS account_links (
            subject TEXT PRIMARY KEY, user_id TEXT UNIQUE NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1)""")
        db.execute("""CREATE TABLE IF NOT EXISTS account_sessions (
            token_hash TEXT PRIMARY KEY, subject TEXT NOT NULL,
            expires_at BIGINT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS account_flows (
            id TEXT PRIMARY KEY, challenge TEXT NOT NULL, cookie_hash TEXT,
            subject TEXT, expires_at BIGINT NOT NULL)""")


def sql(store, text):
    # Only fixed literals are passed by this module, never user-controlled SQL.
    return text.replace("?", "%s") if store.postgres else text


def principal(store, db, subject):
    row = db.execute(
        sql(store, "SELECT user_id,enabled FROM account_links WHERE subject=?"), (subject,)
    ).fetchone()
    if not row or row[1] != 1:
        raise ValueError("Account access is unavailable")
    return row[0]


def resolve(store, supplied):
    if not store.postgres:
        raise ValueError("Managed accounts require the production identity store")
    with store.connect(operator=True) as db:
        row = db.execute(
            "SELECT subject FROM account_sessions WHERE token_hash=%s AND expires_at>%s",
            (digest(supplied), int(time.time())),
        ).fetchone()
        if not row:
            raise ValueError("Invalid or expired session")
        verified_user(db, row[0])
        return principal(store, db, row[0])


class Start(BaseModel):
    challenge: str = Field(pattern=r"^[a-f0-9]{64}$")


class Finish(BaseModel):
    flow: str = Field(pattern=r"^[a-zA-Z0-9_-]{32,80}$")


class Redeem(Finish):
    verifier: str = Field(min_length=32, max_length=100)


def live_flow(store, db, flow):
    row = db.execute(
        sql(
            store,
            "SELECT challenge,cookie_hash,subject FROM account_flows WHERE id=? AND expires_at>?",
        ),
        (flow, int(time.time())),
    ).fetchone()
    if not row:
        raise HTTPException(400, "Sign-in expired. Start again from the dashboard.")
    return row


def login_page(flow=""):
    provider, origin, _, __ = configuration()
    nonce = secrets.token_urlsafe(24)
    config = json.dumps({"provider": provider, "origin": origin, "flow": flow}).replace(
        "<", "\\u003c"
    )
    html = Path(__file__).with_name("account_login.html").read_text(encoding="utf-8")
    response = HTMLResponse(html.replace("__CONFIG__", config).replace("__NONCE__", nonce))
    response.headers.update(
        {
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; connect-src 'self' {provider}; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        }
    )
    return response


def install(app, get_store, auth):
    @app.get("/auth/reset")
    def reset_page():
        # Reset tokens remain governed by the provider, independently of login flows.
        return login_page()

    @app.post("/api/auth/start", dependencies=auth)
    def start(body: Start):
        configuration()
        store = get_store()
        flow = secrets.token_urlsafe(32)
        with store.connect(operator=True) as db:
            now = int(time.time())
            db.execute(sql(store, "DELETE FROM account_flows WHERE expires_at<=?"), (now,))
            db.execute(sql(store, "DELETE FROM account_sessions WHERE expires_at<=?"), (now,))
            if db.execute("SELECT COUNT(*) FROM account_flows").fetchone()[0] >= 100:
                raise HTTPException(429, "Too many sign-ins in progress")
            db.execute(
                sql(store, "INSERT INTO account_flows(id,challenge,expires_at) VALUES (?,?,?)"),
                (flow, body.challenge, now + 600),
            )
        return {
            "flow": flow,
            "url": os.environ["AUTH_GATEWAY_URL"].rstrip("/") + "/auth/login?flow=" + flow,
        }

    @app.get("/auth/login")
    def login(flow: str, request: Request):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{32,80}", flow):
            raise HTTPException(400, "Start sign-in from the dashboard")
        configuration()
        store = get_store()
        cookie = request.cookies.get("rb_login", "")
        with store.connect(operator=True) as db:
            row = live_flow(store, db, flow)
            if not row[1]:
                cookie = secrets.token_urlsafe(32)
                db.execute(
                    sql(
                        store,
                        "UPDATE account_flows SET cookie_hash=? WHERE id=? AND cookie_hash IS NULL",
                    ),
                    (digest(cookie), flow),
                )
            elif not cookie or not secrets.compare_digest(row[1], digest(cookie)):
                raise HTTPException(400, "Return to the browser where you started sign-in")
        response = login_page(flow)
        response.set_cookie(
            "rb_login",
            cookie,
            max_age=600,
            httponly=True,
            secure=True,
            samesite="lax",
            path="/auth",
        )
        return response

    @app.post("/auth/finish")
    def finish(body: Finish, request: Request):
        provider, origin, _, __ = configuration()
        if request.headers.get("origin") != origin:
            raise HTTPException(403, "Invalid sign-in origin")
        authorization = request.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Identity proof required")
        try:
            subject = verified_subject(authorization[7:])
            store = get_store()
            if not store.postgres:
                raise ValueError("Production identity store required")
            with store.connect(operator=True) as db:
                row = live_flow(store, db, body.flow)
                cookie = request.cookies.get("rb_login", "")
                if (
                    not cookie
                    or not row[1]
                    or not secrets.compare_digest(row[1], digest(cookie))
                    or row[2]
                ):
                    raise ValueError("Invalid sign-in state")
                verified_user(db, subject)
                user_id = "acct_" + digest(provider + ":" + subject)[:40]
                db.execute(
                    "INSERT INTO account_links(subject,user_id) VALUES (%s,%s) ON CONFLICT(subject) DO NOTHING",
                    (subject, user_id),
                )
                principal(store, db, subject)
                db.execute("UPDATE account_flows SET subject=%s WHERE id=%s", (subject, body.flow))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(401, "Sign-in could not be verified") from None
        return {"complete": True}

    @app.post("/auth/redeem")
    def redeem(body: Redeem):
        store = get_store()
        with store.connect(operator=True) as db:
            # Row lock makes redemption single-use across all service instances.
            if store.postgres:
                db.execute("SELECT id FROM account_flows WHERE id=%s FOR UPDATE", (body.flow,))
            row = live_flow(store, db, body.flow)
            if not secrets.compare_digest(row[0], digest(body.verifier)):
                raise HTTPException(403, "Invalid sign-in verifier")
            if not row[2]:
                return JSONResponse({"pending": True}, status_code=202)
            try:
                verified_user(db, row[2])
                user_id = principal(store, db, row[2])
            except Exception:
                raise HTTPException(401, "Account access could not be verified") from None
            token = "rb_session_" + secrets.token_urlsafe(32)
            db.execute(
                sql(
                    store,
                    "INSERT INTO account_sessions(token_hash,subject,expires_at) VALUES (?,?,?)",
                ),
                (digest(token), row[2], int(time.time()) + 3600),
            )
            db.execute(sql(store, "DELETE FROM account_flows WHERE id=?"), (body.flow,))
        return {"token": token, "user_id": user_id, "expires_in": 3600}

    @app.post("/api/auth/logout", dependencies=auth)
    def logout(request: Request):
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        store = get_store()
        with store.connect(operator=True) as db:
            db.execute(
                sql(store, "DELETE FROM account_sessions WHERE token_hash=?"), (digest(token),)
            )
        return {"signed_out": True}

    @app.post("/api/access-key", dependencies=auth)
    def access_key():
        from backend.identity import owner
        from scripts.users import provision

        user = owner.get()
        if user == "owner":
            raise HTTPException(403, "Use operator provisioning for the owner")
        key = secrets.token_urlsafe(32)
        provision(get_store(), user, key)
        return {"key": key, "expires_in_days": 30}

    @app.delete("/api/access-key", dependencies=auth)
    def revoke_key():
        from backend.identity import owner

        store = get_store()
        with store.connect(operator=True) as db:
            db.execute(
                sql(store, "UPDATE users SET enabled=false WHERE id=?")
                if store.postgres
                else "UPDATE users SET enabled=0 WHERE id=?",
                (owner.get(),),
            )
        return {"revoked": True}
