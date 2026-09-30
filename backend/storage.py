"""Small SQLite metadata store; Chroma owns its separate vector persistence directory."""
import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4


class Store:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv("DATA_DIR", "data")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "bench.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, kind TEXT, payload TEXT)")
            db.execute("CREATE INDEX IF NOT EXISTS objects_kind ON objects(kind)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def save(self, kind, payload, object_id=None):
        object_id = object_id or uuid4().hex
        payload = {**payload, "id": object_id}
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO objects VALUES (?, ?, ?)",
                       (object_id, kind, json.dumps(payload, allow_nan=False)))
        return payload

    def get(self, kind, object_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM objects WHERE id=? AND kind=?", (object_id, kind)).fetchone()
        if row is None:
            raise KeyError(object_id)
        return json.loads(row[0])

    def list(self, kind, limit=None, offset=0):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM objects WHERE kind=? ORDER BY rowid DESC LIMIT ? OFFSET ?",
                              (kind, -1 if limit is None else limit, offset)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def run_history(self, limit=50, offset=0):
        # Project compact fields inside SQLite: polling history does not decode passages/answers.
        with self.connect() as db:
            rows = db.execute("""SELECT json_object(
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
                (limit, offset)).fetchall()
        return [json.loads(row[0]) for row in rows]
