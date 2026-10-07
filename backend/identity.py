"""Per-user access keys and request/worker identity. Operator code is explicitly unscoped."""

import hashlib
import hmac
import os
from contextlib import contextmanager
from contextvars import ContextVar, copy_context

owner = ContextVar("ragbench_owner", default=None)


@contextmanager
def as_user(user):
    token = owner.set(user)
    try:
        yield
    finally:
        owner.reset(token)


def resolve(store, authorization):
    secret = os.getenv("API_TOKEN", "")
    if secret and hmac.compare_digest(authorization or "", "Bearer " + secret):
        return "owner"
    if not secret and os.getenv("RAGBENCH_ALLOW_INSECURE_LOCAL") == "true":
        return "owner"
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("Bearer access key required")
    supplied = authorization.removeprefix("Bearer ")
    if supplied.startswith("rb_session_"):
        from backend.accounts import resolve as resolve_session

        return resolve_session(store, supplied)
    digest = hashlib.sha256(supplied.encode()).hexdigest()
    with store.connect(operator=True) as db:
        row = db.execute(
            "SELECT id FROM users WHERE token_hash=%s AND enabled=true AND expires_at > CURRENT_TIMESTAMP"
            if store.postgres
            else "SELECT id FROM users WHERE token_hash=? AND enabled=1 AND datetime(expires_at)>datetime('now')",
            (digest,),
        ).fetchone()
    if row is None:
        raise ValueError("Invalid or expired access key")
    with store.connect(operator=True) as db:
        linked = db.execute(
            "SELECT subject,enabled FROM account_links WHERE user_id=%s"
            if store.postgres
            else "SELECT subject,enabled FROM account_links WHERE user_id=?",
            (row[0],),
        ).fetchone()
        if linked:
            if linked[1] != 1 or not store.postgres:
                raise ValueError("Account access is unavailable")
            from backend.accounts import verified_user

            verified_user(db, linked[0])
    return row[0]


def submit(executor, function, *args):
    context = copy_context()
    return executor.submit(context.run, function, *args)
