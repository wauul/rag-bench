"""Authoritative JSON storage: PostgreSQL when configured, SQLite for local use.

Chroma is a disposable cache; documents, snapshots and retrieved passages live here.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class Store:
    def __init__(self, root=None, database_url=None):
        # Explicit local roots remain local unless a database URL is explicitly supplied.
        self.database_url = (
            database_url
            if database_url is not None
            else (os.getenv("DATABASE_URL", "") if root is None else "")
        )
        self.postgres = bool(self.database_url)
        self.root = Path(root or os.getenv("DATA_DIR", "data")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "bench.sqlite3"
        if self.postgres:
            from backend.postgres import validate_database_url

            validate_database_url(self.database_url)
            with self.connect() as db:
                db.execute("""CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload JSONB NOT NULL,
                    sequence BIGSERIAL NOT NULL)""")
                db.execute("CREATE INDEX IF NOT EXISTS objects_kind ON objects(kind)")
            from backend.guardrails import initialize

            initialize(self)
            self.initialize_identity()
            return
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, kind TEXT, payload TEXT)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS objects_kind ON objects(kind)")
        from backend.guardrails import initialize

        initialize(self)
        self.initialize_identity()

    def initialize_identity(self):
        with self.connect(operator=True) as db:
            if self.postgres:
                db.execute(
                    "ALTER TABLE objects ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'owner'"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL, enabled BOOLEAN NOT NULL DEFAULT true, expires_at TIMESTAMPTZ NOT NULL)"
                )
                db.execute("ALTER TABLE objects ENABLE ROW LEVEL SECURITY")
                db.execute("ALTER TABLE objects FORCE ROW LEVEL SECURITY")
                if not db.execute(
                    "SELECT 1 FROM pg_policies WHERE schemaname=current_schema() AND tablename='objects' AND policyname='owner_access'"
                ).fetchone():
                    db.execute(
                        "CREATE POLICY owner_access ON objects USING (current_setting('ragbench.operator',true)='true' OR owner_id=current_setting('ragbench.owner',true) OR kind='guardrails') WITH CHECK (current_setting('ragbench.operator',true)='true' OR owner_id=current_setting('ragbench.owner',true))"
                    )
            else:
                columns = {r[1] for r in db.execute("PRAGMA table_info(objects)")}
                if "owner_id" not in columns:
                    db.execute(
                        "ALTER TABLE objects ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'owner'"
                    )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, expires_at TEXT NOT NULL)"
                )

    @contextmanager
    def connect(self, operator=False):
        from backend.identity import owner

        def scope(db):
            db.execute(
                "SELECT set_config('ragbench.owner',%s,true), set_config('ragbench.operator',%s,true)",
                (owner.get() or "", "true" if operator or owner.get() is None else "false"),
            )

        if self.postgres:
            from backend.execution_lock import execution_connection
            from backend.postgres import connect

            owned = execution_connection(self)
            if owned is not None:
                # Use the lock-owning session: a lost lock cannot write through a new connection.
                with owned.transaction():
                    scope(owned)
                    yield owned
                return
            with connect(self.database_url) as db:
                scope(db)
                yield db
        else:
            db = sqlite3.connect(self.path, timeout=30)
            try:
                with db:
                    yield db
            finally:
                db.close()

    def save(self, kind, payload, object_id=None):
        from datetime import datetime, timezone

        from backend.execution_lock import assert_execution_lock
        from backend.identity import owner

        assert_execution_lock(self)
        object_id = object_id or uuid4().hex
        payload = {**payload, "id": object_id}
        from backend.guardrails import locked

        with locked(self) as (db, _, policy):
            existing = db.execute(
                "SELECT owner_id,payload FROM objects WHERE id=%s"
                if self.postgres
                else "SELECT owner_id,payload FROM objects WHERE id=?",
                (object_id,),
            ).fetchone()
            principal = owner.get()
            if existing and principal is not None and existing[0] != principal:
                raise KeyError(object_id)
            principal = principal or (existing[0] if existing else payload.get("owner_id", "owner"))
            previous = (
                (existing[1] if self.postgres else json.loads(existing[1])) if existing else {}
            )
            payload.update(
                owner_id=principal,
                retained_at=previous.get("retained_at", datetime.now(timezone.utc).isoformat()),
            )
            encoded = json.dumps(payload, allow_nan=False)
            # Count authoritative JSON, including snapshots. Updates replace their old size.
            total = db.execute(
                "SELECT COALESCE(SUM(octet_length(payload::text)),0) FROM objects"
                if self.postgres
                else "SELECT COALESCE(SUM(length(CAST(payload AS BLOB))),0) FROM objects"
            ).fetchone()[0]
            old = db.execute(
                "SELECT octet_length(payload::text) FROM objects WHERE id=%s"
                if self.postgres
                else "SELECT length(CAST(payload AS BLOB)) FROM objects WHERE id=?",
                (object_id,),
            ).fetchone()
            incoming = (
                db.execute("SELECT octet_length(%s::jsonb::text)", (encoded,)).fetchone()[0]
                if self.postgres
                else len(encoded.encode())
            )
            if total - (old[0] if old else 0) + incoming > policy.storage_bytes:
                raise ValueError("Workspace storage quota exhausted")
            db.execute(
                """INSERT INTO objects(id, kind, payload, owner_id) VALUES (%s, %s, %s::jsonb, %s)
                ON CONFLICT(id) DO UPDATE SET kind=EXCLUDED.kind, payload=EXCLUDED.payload"""
                if self.postgres
                else "INSERT OR REPLACE INTO objects(id,kind,payload,owner_id) VALUES (?, ?, ?, ?)",
                (object_id, kind, encoded, principal),
            )
        return payload

    def get(self, kind, object_id):
        from backend.identity import owner

        principal = owner.get()
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM objects WHERE id=%s AND kind=%s AND (owner_id=%s OR %s::text IS NULL OR kind='guardrails')"
                if self.postgres
                else "SELECT payload FROM objects WHERE id=? AND kind=? AND (owner_id=? OR ? IS NULL OR kind='guardrails')",
                (object_id, kind, principal, principal),
            ).fetchone()
        if row is None:
            raise KeyError(object_id)
        return row[0] if self.postgres else json.loads(row[0])

    def fail_queued_investigation(self, object_id):
        """Atomic admission failure; never overwrite a worker that has begun running."""
        patch = json.dumps({"status": "failed", "error": "Worker busy; resume saved stages"})
        with self.connect() as db:
            db.execute(
                "UPDATE objects SET payload=payload || %s::jsonb "
                "WHERE id=%s AND kind='investigation' AND payload->>'status'='queued'"
                if self.postgres
                else "UPDATE objects SET payload=json_patch(payload, ?) "
                "WHERE id=? AND kind='investigation' AND json_extract(payload, '$.status')='queued'",
                (patch, object_id),
            )

    def fail_queued_optimization(self, object_id):
        patch = json.dumps({"status": "failed", "stage": "Worker busy; explicit resume required"})
        with self.connect() as db:
            db.execute(
                "UPDATE objects SET payload=payload || %s::jsonb WHERE id=%s AND kind='optimization' AND payload->>'status'='queued'"
                if self.postgres
                else "UPDATE objects SET payload=json_patch(payload, ?) WHERE id=? AND kind='optimization' AND json_extract(payload, '$.status')='queued'",
                (patch, object_id),
            )

    def list(self, kind, limit=None, offset=0):
        from backend.identity import owner

        principal = owner.get()
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM objects WHERE kind=%s AND (owner_id=%s OR %s::text IS NULL) ORDER BY sequence DESC LIMIT %s OFFSET %s"
                if self.postgres
                else "SELECT payload FROM objects WHERE kind=? AND (owner_id=? OR ? IS NULL) ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (
                    kind,
                    principal,
                    principal,
                    limit if self.postgres else (-1 if limit is None else limit),
                    offset,
                ),
            ).fetchall()
        return [row[0] if self.postgres else json.loads(row[0]) for row in rows]

    def run_history(self, limit=50, offset=0):
        from backend.identity import owner

        principal = owner.get()
        if self.postgres:
            fields = (
                "status",
                "stage",
                "created_at",
                "finished_at",
                "completed",
                "total",
                "scored",
                "configurations",
            )
            projection = ", ".join(f"'{field}', payload->'{field}'" for field in fields)
            # Projection uses only the literal field allowlist above; values remain bound.
            with self.connect() as db:
                rows = db.execute(
                    "SELECT jsonb_build_object('id', id, " + projection + ") FROM objects "  # nosec B608
                    "WHERE kind='run' AND (owner_id=%s OR %s::text IS NULL) ORDER BY payload->>'created_at' DESC, id DESC LIMIT %s OFFSET %s",
                    (principal, principal, limit, offset),
                ).fetchall()
            return [row[0] for row in rows]
        # Project compact fields inside SQLite: polling history does not decode passages/answers.
        with self.connect() as db:
            rows = db.execute(
                """SELECT json_object(
                'id', id, 'status', json_extract(payload, '$.status'),
                'stage', json_extract(payload, '$.stage'),
                'created_at', json_extract(payload, '$.created_at'),
                'finished_at', json_extract(payload, '$.finished_at'),
                'completed', json_extract(payload, '$.completed'),
                'total', json_extract(payload, '$.total'),
                'scored', json_extract(payload, '$.scored'),
                'configurations', json_extract(payload, '$.configurations'))
                FROM objects WHERE kind='run' AND (owner_id=? OR ? IS NULL)
                ORDER BY json_extract(payload, '$.created_at') DESC, id DESC LIMIT ? OFFSET ?""",
                (principal, principal, limit, offset),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete_run(self, run_id):
        """Call only while holding worker admission and the DATA_DIR execution lock.

        Keep the run until external artifacts are removed, making interrupted deletion retryable.
        Shared uploaded inputs/configurations remain available to other runs.
        """
        run = self.get("run", run_id)
        if self.postgres or (self.root / "langgraph.sqlite3").exists():
            from backend.graph_pipeline import checkpointer, thread_id

            with checkpointer(self) as saver:
                for investigation in self.list("investigation"):
                    if investigation["run_id"] == run_id:
                        saver.delete_thread("investigation-" + investigation["id"])
                for config in run["configurations"]:
                    for index in range(len(run["questions"])):
                        saver.delete_thread(thread_id(run_id, config["id"], index))
        if (self.root / "chroma").exists():
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(self.root / "chroma"), settings=Settings(anonymized_telemetry=False)
            )
            names = {collection.name for collection in client.list_collections()}
            for config in run["configurations"]:
                name = f"run-{run_id}-cfg-{config['id']}"
                if name in names:
                    client.delete_collection(name)
        with self.connect() as db:
            db.execute(
                "DELETE FROM objects WHERE kind IN ('snapshot', 'graph_artifact', 'retrieval_trace', 'cancellation', 'investigation', 'investigation_cancel') "
                "AND payload->>'run_id'=%s"
                if self.postgres
                else "DELETE FROM objects WHERE kind IN ('snapshot', 'graph_artifact', 'retrieval_trace', 'cancellation', 'investigation', 'investigation_cancel') AND json_extract(payload, '$.run_id')=?",
                (run_id,),
            )
            db.execute(
                "DELETE FROM objects WHERE kind='run' AND id=%s"
                if self.postgres
                else "DELETE FROM objects WHERE kind='run' AND id=?",
                (run_id,),
            )
