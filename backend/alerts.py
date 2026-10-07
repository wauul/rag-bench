"""Fixed-destination, durable operational email outbox; never includes user content."""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import httpx

from backend.guardrails import locked, persist

log = logging.getLogger(__name__)
EVENTS = {"quota-warning", "quota-blocked", "maintenance-failed", "deployment-test"}


def enqueue(store, event):
    if event not in EVENTS:
        raise ValueError("Unknown operational event")
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    identity = "alert-" + hashlib.sha256((event + day).encode()).hexdigest()
    with locked(store) as (db, _, __):
        db.execute(
            "INSERT INTO objects(id,kind,payload,owner_id) VALUES (%s,'alert',%s::jsonb,'owner') ON CONFLICT(id) DO NOTHING"
            if store.postgres
            else "INSERT OR IGNORE INTO objects(id,kind,payload,owner_id) VALUES (?,'alert',?,'owner')",
            (identity, json.dumps({"event": event, "day": day, "attempts": 0, "sent": False})),
        )


def deliver(store):
    key, sender, recipient = (
        os.getenv(n, "") for n in ("RESEND_API_KEY", "ALERT_FROM", "ALERT_TO")
    )
    if not all((key, sender, recipient)):
        return 0
    if (
        any("\n" in v or "\r" in v for v in (sender, recipient))
        or "@" not in sender
        or "@" not in recipient
    ):
        raise ValueError("Invalid alert destination")
    delivered = 0
    # One attempt per event per maintenance tick, at most 3 attempts total.
    with locked(store) as (db, state, _):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        budget = state.get("email_usage", {})
        if budget.get("day") != day:
            budget = {"day": day, "attempts": 0}
        pending = []
        rows = db.execute("SELECT id,payload FROM objects WHERE kind='alert'").fetchall()
        for identity, raw in rows:
            record = raw if store.postgres else json.loads(raw)
            if record["sent"] or record["attempts"] >= 3 or budget["attempts"] >= 20:
                continue
            if record["event"] not in EVENTS:
                raise ValueError("Invalid operational event")
            record["attempts"] += 1
            budget["attempts"] += 1
            db.execute(
                "UPDATE objects SET payload=%s::jsonb WHERE id=%s"
                if store.postgres
                else "UPDATE objects SET payload=? WHERE id=?",
                (json.dumps(record), identity),
            )
            pending.append((identity, record))
        state["email_usage"] = budget
        persist(store, db, state)
    for identity, record in pending:
        try:
            with httpx.Client(timeout=5, follow_redirects=False) as client:
                response = client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": "Bearer " + key, "Idempotency-Key": identity},
                    json={
                        "from": sender,
                        "to": [recipient],
                        "subject": "Ragbench: " + record["event"],
                        "text": "Ragbench operational event: "
                        + record["event"]
                        + ". UTC day: "
                        + record["day"]
                        + ". Check the operator usage dashboard. No user content is included.",
                    },
                )
                response.raise_for_status()
            with locked(store) as (db, _, __):
                record["sent"] = True
                db.execute(
                    "UPDATE objects SET payload=%s::jsonb WHERE id=%s"
                    if store.postgres
                    else "UPDATE objects SET payload=? WHERE id=?",
                    (json.dumps(record), identity),
                )
            delivered += 1
        except (httpx.HTTPError, ValueError):
            log.error("Operational email delivery failed; bounded retry retained")
    return delivered
