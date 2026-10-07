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
            return
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, kind TEXT, payload TEXT)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS objects_kind ON objects(kind)")

    @contextmanager
    def connect(self):
        if self.postgres:
            from backend.execution_lock import execution_connection
            from backend.postgres import connect

            owned = execution_connection(self)
            if owned is not None:
                # Use the lock-owning session: a lost lock cannot write through a new connection.
                with owned.transaction():
                    yield owned
                return
            with connect(self.database_url) as db:
                yield db
        else:
            db = sqlite3.connect(self.path, timeout=30)
            try:
                with db:
                    yield db
            finally:
                db.close()

    def save(self, kind, payload, object_id=None):
        from backend.execution_lock import assert_execution_lock

        assert_execution_lock(self)
        object_id = object_id or uuid4().hex
        payload = {**payload, "id": object_id}
        encoded = json.dumps(payload, allow_nan=False)
        with self.connect() as db:
            db.execute(
                """INSERT INTO objects(id, kind, payload) VALUES (%s, %s, %s::jsonb)
                ON CONFLICT(id) DO UPDATE SET kind=EXCLUDED.kind, payload=EXCLUDED.payload"""
                if self.postgres
                else "INSERT OR REPLACE INTO objects VALUES (?, ?, ?)",
                (object_id, kind, encoded),
            )
        return payload

    def get(self, kind, object_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM objects WHERE id=%s AND kind=%s"
                if self.postgres
                else "SELECT payload FROM objects WHERE id=? AND kind=?",
                (object_id, kind),
            ).fetchone()
        if row is None:
            raise KeyError(object_id)
        return row[0] if self.postgres else json.loads(row[0])

    def list(self, kind, limit=None, offset=0):
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM objects WHERE kind=%s ORDER BY sequence DESC LIMIT %s OFFSET %s"
                if self.postgres
                else "SELECT payload FROM objects WHERE kind=? ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (kind, limit if self.postgres else (-1 if limit is None else limit), offset),
            ).fetchall()
        return [row[0] if self.postgres else json.loads(row[0]) for row in rows]

    def run_history(self, limit=50, offset=0):
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
            with self.connect() as db:
                rows = db.execute(
                    "SELECT jsonb_build_object('id', id, " + projection + ") FROM objects "
                    "WHERE kind='run' ORDER BY payload->>'created_at' DESC, id DESC LIMIT %s OFFSET %s",
                    (limit, offset),
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
                FROM objects WHERE kind='run'
                ORDER BY json_extract(payload, '$.created_at') DESC, id DESC LIMIT ? OFFSET ?""",
                (limit, offset),
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
                "DELETE FROM objects WHERE kind IN ('snapshot', 'graph_artifact', 'cancellation') "
                "AND payload->>'run_id'=%s"
                if self.postgres
                else "DELETE FROM objects WHERE kind IN ('snapshot', 'graph_artifact', 'cancellation') AND json_extract(payload, '$.run_id')=?",
                (run_id,),
            )
            db.execute(
                "DELETE FROM objects WHERE kind='run' AND id=%s"
                if self.postgres
                else "DELETE FROM objects WHERE kind='run' AND id=?",
                (run_id,),
            )
