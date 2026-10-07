"""Offline content deletion. Stop the API and disable workspace features first.

Usage counters/policy survive deletion. External provider records and backups do not
live here and require operator cleanup under their own retention settings.
"""

import argparse
import shutil

from dotenv import load_dotenv

from backend.execution_lock import execution_lock
from backend.guardrails import locked
from backend.storage import Store


def purge(store):
    with execution_lock(store, "server.lock"), execution_lock(store):
        with locked(store) as (_, __, policy):
            if policy.enabled:
                raise ValueError("Disable workspace features before purging content")
        cache = store.root / "chroma"
        if cache.is_symlink() or cache.resolve().parent != store.root.resolve():
            raise ValueError("Unsafe cache path")
        if store.postgres or (store.root / "langgraph.sqlite3").exists():
            from backend.graph_pipeline import checkpointer

            with checkpointer(store) as saver:
                threads = {item.config["configurable"]["thread_id"] for item in saver.list(None)}
                for thread in threads:
                    saver.delete_thread(thread)
        if cache.exists():
            shutil.rmtree(cache)
        with locked(store) as (db, _, __):
            db.execute("DELETE FROM objects WHERE kind != 'guardrails'")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True, choices=["DELETE_WORKSPACE_CONTENT"])
    parser.parse_args()
    purge(Store())
    print("Workspace content deleted; policy and consumed quotas retained")


if __name__ == "__main__":
    main()
