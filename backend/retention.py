"""Daily automatic retention. Keep active records and dependencies of retained work."""

import json
from datetime import datetime, timedelta, timezone

from backend.execution_lock import execution_lock
from backend.guardrails import locked


def sweep(store, now=None, days=None):
    now = now or datetime.now(timezone.utc)
    with execution_lock(store), locked(store) as (db, _, policy):
        cutoff = now - timedelta(days=days or policy.retention_days)
        db.execute(
            "DELETE FROM objects WHERE kind='alert' AND payload->>'day' < %s"
            if store.postgres
            else "DELETE FROM objects WHERE kind='alert' AND json_extract(payload,'$.day') < ?",
            (cutoff.strftime("%Y-%m-%d"),),
        )
        rows = db.execute(
            "SELECT id,kind,payload FROM objects WHERE kind NOT IN ('guardrails','alert')"
        ).fetchall()
        records = {r[0]: (r[1], r[2] if store.postgres else json.loads(r[2])) for r in rows}
        protected = set()
        for identity, (kind, record) in records.items():
            date = (
                record.get("finished_at") or record.get("retained_at") or record.get("created_at")
            )
            if date is None:
                record["retained_at"] = now.isoformat()
                db.execute(
                    "UPDATE objects SET payload=%s::jsonb WHERE id=%s"
                    if store.postgres
                    else "UPDATE objects SET payload=? WHERE id=?",
                    (json.dumps(record), identity),
                )
                protected.add(identity)
                continue
            try:
                stamp = (
                    datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
                    if datetime.fromisoformat(date).tzinfo is None
                    else datetime.fromisoformat(date)
                )
            except (ValueError, TypeError):
                protected.add(identity)
                continue
            if stamp >= cutoff or record.get("status") in {"queued", "running"}:
                protected.add(identity)

        def references(value):
            if isinstance(value, dict):
                return (
                    set().union(
                        *(references(v) for k, v in value.items() if k not in {"id", "owner_id"})
                    )
                    if value
                    else set()
                )
            if isinstance(value, list):
                return set().union(*(references(v) for v in value)) if value else set()
            return {value} if isinstance(value, str) and value in records else set()

        while True:
            extra = (
                set().union(*(references(records[i][1]) for i in protected)) if protected else set()
            )
            extra.update(
                i
                for i, (_, r) in records.items()
                if any(r.get(k) in protected for k in ("run_id", "optimization_id", "debug_id"))
            )
            extra -= protected
            if not extra:
                break
            protected.update(extra)
        expired = set(records) - protected
        if not expired:
            return 0
        # Checkpoint IDs are independent of object ownership; only delete expired roots.
        if store.postgres or (store.root / "langgraph.sqlite3").exists():
            from backend.graph_pipeline import checkpointer, thread_id

            with checkpointer(store) as saver:
                for identity in expired:
                    kind, record = records[identity]
                    if kind == "run":
                        for config in record["configurations"]:
                            for index in range(len(record["questions"])):
                                saver.delete_thread(thread_id(identity, config["id"], index))
                    elif kind in {"investigation", "optimization", "debug_run"}:
                        prefix = {
                            "investigation": "investigation-",
                            "optimization": "optimization-",
                            "debug_run": "debug-",
                        }[kind]
                        saver.delete_thread(prefix + identity)
        if (store.root / "chroma").exists():
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(store.root / "chroma"), settings=Settings(anonymized_telemetry=False)
            )
            kept_indexes = {
                records[i][1].get("index", {}).get("collection")
                for i in protected
                if isinstance(records[i][1].get("index"), dict)
            }
            for collection in client.list_collections():
                if (
                    collection.name.startswith("run-debug-") and collection.name not in kept_indexes
                ) or any(
                    collection.name.startswith("run-" + identity + "-") for identity in expired
                ):
                    client.delete_collection(collection.name)
        for identity in expired:
            db.execute(
                "DELETE FROM objects WHERE id=%s"
                if store.postgres
                else "DELETE FROM objects WHERE id=?",
                (identity,),
            )
        return len(expired)
