"""Copy a stopped local SQLite workspace into an empty PostgreSQL database.

Source files are never modified or deleted. DATABASE_URL is read from the environment.
Run after stopping the source API and before starting the PostgreSQL-backed API.
"""

import argparse
import json
import os
from pathlib import Path

from backend.execution_lock import execution_lock
from backend.storage import Store


def migrate(source: Path, database_url: str) -> dict:
    if not (source / "bench.sqlite3").is_file():
        raise ValueError("Source has no bench.sqlite3")
    local = Store(source, database_url="")
    remote = Store(source / "migration-cache", database_url=database_url)
    with execution_lock(local, "server.lock"), execution_lock(local), execution_lock(remote):
        with remote.connect() as db:
            if db.execute("SELECT COUNT(*) FROM objects").fetchone()[0]:
                raise ValueError(
                    "Migration target must be empty; existing data will not be overwritten"
                )
        with local.connect() as db:
            rows = db.execute("SELECT id, kind, payload FROM objects ORDER BY rowid").fetchall()
        checkpoints = 0
        if (source / "langgraph.sqlite3").exists():
            from backend.graph_pipeline import checkpointer

            with checkpointer(local) as old, checkpointer(remote) as new:
                for item in reversed(list(old.list(None))):
                    config = item.config
                    parent = item.parent_config
                    incoming = {
                        "configurable": {
                            **config["configurable"],
                            "checkpoint_id": parent["configurable"]["checkpoint_id"]
                            if parent
                            else None,
                        }
                    }
                    saved = new.put(
                        incoming,
                        item.checkpoint,
                        item.metadata,
                        item.checkpoint["channel_versions"],
                    )
                    tasks = {}
                    for task, channel, value in item.pending_writes or []:
                        tasks.setdefault(task, []).append((channel, value))
                    for task, writes in tasks.items():
                        new.put_writes(saved, writes, task)
                    checkpoints += 1
        with remote.connect() as db:
            for identity, kind, encoded in rows:
                json.loads(encoded)  # Validate before copying; preserve source payload exactly.
                db.execute(
                    "INSERT INTO objects(id, kind, payload) VALUES (%s, %s, %s::jsonb)",
                    (identity, kind, encoded),
                )
        return {"objects": len(rows), "checkpoints": checkpoints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, os.environ["DATABASE_URL"])))


if __name__ == "__main__":
    main()
