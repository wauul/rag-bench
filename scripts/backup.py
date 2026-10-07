"""Offline, checksummed snapshots. Stop the API before backup; restore to a NEW directory."""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from backend.execution_lock import execution_lock
from backend.storage import Store

FORMAT = 1


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Snapshots cannot contain symlinks")
        if path.is_file() and path.name != "manifest.json":
            result[path.relative_to(root).as_posix()] = checksum(path)
    return result


def backup(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or source in destination.parents:
        raise ValueError("Backup target must be new and outside DATA_DIR")
    if not (source / "bench.sqlite3").is_file():
        raise ValueError("DATA_DIR has no bench.sqlite3")
    store = Store(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with execution_lock(store, "server.lock"), execution_lock(store):
        temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=destination.parent))
        try:
            for name in ("bench.sqlite3", "langgraph.sqlite3", "chroma"):
                path = source / name
                if not path.exists():
                    continue
                if path.is_symlink() or any(p.is_symlink() for p in path.rglob("*")):
                    raise ValueError("Durable storage cannot contain symlinks")
                if path.is_dir():
                    shutil.copytree(path, temporary / name)
                else:
                    # SQLite backup API includes committed WAL frames in the snapshot.
                    with (
                        closing(sqlite3.connect(path)) as db,
                        closing(sqlite3.connect(temporary / name)) as out,
                    ):
                        db.backup(out)
            (temporary / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": FORMAT,
                        "storage_schema": 1,
                        "revision": os.getenv("APP_REVISION", "local"),
                        "files": inventory(temporary),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.rename(destination)
        except BaseException:
            shutil.rmtree(temporary)
            raise


def restore(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or source in destination.parents:
        raise ValueError("Restore target must be a NEW directory outside the snapshot")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("storage_schema") != 1:
        raise ValueError("Unsupported snapshot format/storage schema")
    if inventory(source) != manifest["files"] or "bench.sqlite3" not in manifest["files"]:
        raise ValueError("Snapshot checksum mismatch or missing database")
    for path in source.rglob("*.sqlite3"):
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
            if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("Snapshot database integrity check failed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".restore-", dir=destination.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        (temporary / "manifest.json").unlink()
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["backup", "restore"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if os.getenv("DATABASE_URL"):
        raise ValueError("DATABASE_URL is set: use scripts.postgres_backup for authoritative data")
    (backup if args.operation == "backup" else restore)(args.source, args.destination)
    print(f"{args.operation} complete; private snapshot contents are not logged")


if __name__ == "__main__":
    main()
