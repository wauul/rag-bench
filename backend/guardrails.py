"""Durable workspace policy. Reservations commit before any provider HTTP attempt."""

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

_store = ContextVar("guardrail_store", default=None)
POLICY_ID = "workspace-guardrails"
log = logging.getLogger(__name__)


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = True
    daily_requests: int = Field(default=500, ge=1, le=100000)
    monthly_requests: int = Field(default=5000, ge=1, le=1000000)
    daily_token_allowance: int = Field(default=2000000, ge=1)
    monthly_token_allowance: int = Field(default=20000000, ge=1)
    requests_per_minute: int = Field(default=120, ge=1, le=10000)
    writes_per_minute: int = Field(default=20, ge=1, le=1000)
    storage_bytes: int = Field(default=100 * 1024 * 1024, ge=1048576)
    retention_days: int = Field(default=30, ge=1, le=365)


def initialize(store):
    encoded = json.dumps({"policy": Policy().model_dump(), "usage": {}, "rates": {}})
    with store.connect(operator=True) as db:
        db.execute(
            "INSERT INTO objects(id,kind,payload) VALUES (%s,'guardrails',%s::jsonb) ON CONFLICT(id) DO NOTHING"
            if store.postgres
            else "INSERT OR IGNORE INTO objects(id,kind,payload) VALUES (?,'guardrails',?)",
            (POLICY_ID, encoded),
        )


@contextmanager
def locked(store):
    with store.connect(operator=True) as db:
        if store.postgres:
            # Also serialize storage admission across independent API processes.
            db.execute("SELECT pg_advisory_xact_lock(727416391)")
        else:
            db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT payload FROM objects WHERE id=%s AND kind='guardrails'"
            if store.postgres
            else "SELECT payload FROM objects WHERE id=? AND kind='guardrails'",
            (POLICY_ID,),
        ).fetchone()
        if row is None:
            raise ValueError("Workspace policy unavailable")
        state = row[0] if store.postgres else json.loads(row[0])
        policy = Policy.model_validate(state["policy"])
        yield db, state, policy


def persist(store, db, state):
    db.execute(
        "UPDATE objects SET payload=%s::jsonb WHERE id=%s"
        if store.postgres
        else "UPDATE objects SET payload=? WHERE id=?",
        (json.dumps(state, allow_nan=False), POLICY_ID),
    )


def check(store):
    with locked(store) as (_, __, policy):
        if not policy.enabled:
            raise ValueError("Workspace expensive features disabled")


@contextmanager
def policy_context(store):
    check(store)
    token = _store.set(store)
    try:
        yield
    finally:
        _store.reset(token)


def reserve(request):
    try:
        return _reserve(request)
    except ValueError as exc:
        if "quota exhausted" in str(exc):
            from backend.alerts import enqueue

            store = _store.get()
            if store is not None:
                try:
                    enqueue(store, "quota-blocked")
                except Exception:
                    log.error("Operational alert could not be queued")
        raise


def _reserve(request):
    store = _store.get()
    if store is None:
        from backend.optimization_budget import current_budget
        from backend.storage import Store

        budget = current_budget()
        store = budget.store if budget else Store()
    # Transport-level fixtures and health probes do not generate tokens. All Groq
    # generation requests are POSTs and must declare their output bound.
    if request.method != "POST":
        return
    payload = json.loads(request.content)
    output = payload.get("max_completion_tokens", payload.get("max_tokens"))
    if not isinstance(output, int) or isinstance(output, bool) or not 1 <= output <= 4096:
        raise ValueError("Provider output allowance must be bounded")
    # Conservative input-byte allowance, protocol overhead and maximum output;
    # do not refund unknown/failed responses. This is not a monetary price estimate.
    allowance = len(request.content) + output + 1024
    now = datetime.now(timezone.utc)
    warning = False
    with locked(store) as (db, state, policy):
        if not policy.enabled:
            raise ValueError("Workspace expensive features disabled")
        periods = (
            ("day", now.strftime("%Y-%m-%d"), policy.daily_requests, policy.daily_token_allowance),
            (
                "month",
                now.strftime("%Y-%m"),
                policy.monthly_requests,
                policy.monthly_token_allowance,
            ),
        )
        for key, period, requests, tokens in periods:
            usage = state["usage"].get(key, {})
            if usage.get("period") != period:
                usage = {"period": period, "requests": 0, "allowance": 0}
            if usage["requests"] >= requests or usage["allowance"] + allowance > tokens:
                log.warning("Workspace %s provider quota exhausted; outbound attempt denied", key)
                raise ValueError("Workspace provider quota exhausted")
            if usage["requests"] < requests * 0.8 <= usage["requests"] + 1 or (
                usage["allowance"] < tokens * 0.8 <= usage["allowance"] + allowance
            ):
                log.warning("Workspace %s provider allowance reached 80 percent", key)
                warning = True
            state["usage"][key] = {
                **usage,
                "requests": usage["requests"] + 1,
                "allowance": usage["allowance"] + allowance,
            }
        persist(store, db, state)
    if warning:
        from backend.alerts import enqueue

        enqueue(store, "quota-warning")


def throttle(store, identity, write=False):
    minute = int(time.time() // 60)
    key = hashlib.sha256(identity.encode()).hexdigest()
    with locked(store) as (db, state, policy):
        rates = {k: v for k, v in state["rates"].items() if v["minute"] == minute}
        if key not in rates and len(rates) >= 10000:
            raise ValueError("Request limiter capacity reached")
        usage = rates.get(key, {"minute": minute, "requests": 0, "writes": 0})
        if usage["requests"] >= policy.requests_per_minute or (
            write and usage["writes"] >= policy.writes_per_minute
        ):
            return False
        rates[key] = {
            **usage,
            "requests": usage["requests"] + 1,
            "writes": usage["writes"] + int(write),
        }
        state["rates"] = rates
        persist(store, db, state)
    return True
