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
    return row[0]


def submit(executor, function, *args):
    context = copy_context()
    return executor.submit(context.run, function, *args)
