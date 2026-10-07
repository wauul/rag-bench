"""Consistent PostgreSQL custom dumps; restore only into an empty destination.

Use a direct TLS DATABASE_URL. Credentials are passed via libpq environment,
never command arguments or output. Requires PostgreSQL client tools matching the server.
Stop API writes for a maintenance recovery point; caches are intentionally excluded.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from psycopg.conninfo import conninfo_to_dict

from backend.postgres import connect, validate_database_url
from scripts.backup import checksum, inventory


def tool(name: str, url: str, *args: str) -> None:
    validate_database_url(url)
    options = conninfo_to_dict(url)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
    for key, variable in {
        "host": "PGHOST",
        "port": "PGPORT",
        "user": "PGUSER",
        "password": "PGPASSWORD",
        "dbname": "PGDATABASE",
        "sslmode": "PGSSLMODE",
        "sslrootcert": "PGSSLROOTCERT",
    }.items():
        if key in options:
            if options[key] is not None:
                environment[variable] = str(options[key])
    environment["PGCONNECT_TIMEOUT"] = "15"
    command = [name, *args]
    if os.getenv("RAGBENCH_PG_DOCKER") == "1":
        # Cross-platform client tools without installing PostgreSQL on the host.
        # Mount only the private snapshot directory; docker inherits libpq variables by name.
        file_arg = next(a for a in args if a.startswith("--file=") or a.endswith("database.dump"))
        path = Path(file_arg.removeprefix("--file=")).resolve()
        command = [
            "docker",
            "run",
            "--rm",
            "--network=host",
            "--mount",
            f"type=bind,source={path.parent},target=/snapshot",
        ]
        for key in environment:
            if key.startswith("PG"):
                command.extend(["--env", key])
        if os.name == "nt" and environment.get("PGHOST") in {"localhost", "127.0.0.1"}:
            environment["PGHOST"] = "host.docker.internal"
            command.remove("--network=host")
        command.extend(["postgres:17-alpine", name])
        command.extend(a.replace(str(path), "/snapshot/database.dump") for a in args)
    result = subprocess.run(command, env=environment, capture_output=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f"{name} failed; private database diagnostics suppressed")


def backup(url: str, destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Backup destination must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pg-backup-", dir=destination.parent) as temporary:
        root = Path(temporary)
        tool(
            "pg_dump",
            url,
            "--format=custom",
            "--no-owner",
            "--no-acl",
            "--table=public.*",
            "--file=" + str(root / "database.dump"),
        )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "format": 1,
                    "storage": "postgresql",
                    "storage_schema": 1,
                    "revision": os.getenv("APP_REVISION", "local"),
                    "files": {"database.dump": checksum(root / "database.dump")},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Copy out atomically; TemporaryDirectory remains responsible for its own tree.
        import shutil

        staged = destination.with_name(destination.name + ".partial")
        if staged.exists():
            raise ValueError("Staging destination already exists")
        try:
            shutil.copytree(root, staged)
            staged.rename(destination)
        finally:
            if staged.exists():
                shutil.rmtree(staged)


def restore(source: Path, url: str) -> None:
    validate_database_url(url)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format"), manifest.get("storage"), manifest.get("storage_schema")) != (
        1,
        "postgresql",
        1,
    ):
        raise ValueError("Unsupported PostgreSQL backup format")
    if inventory(source) != manifest["files"] or set(manifest["files"]) != {"database.dump"}:
        raise ValueError("PostgreSQL backup checksum mismatch")
    with connect(url) as db:
        if db.execute(
            "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname NOT IN ('pg_catalog', 'information_schema') "
            "AND n.nspname NOT LIKE 'pg_toast%' AND c.relkind IN ('r','v','m','S','f') LIMIT 1"
        ).fetchone():
            raise ValueError("Restore database must be empty; existing data is never overwritten")
    tool(
        "pg_restore",
        url,
        "--exit-on-error",
        "--single-transaction",
        "--no-owner",
        "--no-acl",
        "--dbname=" + str(conninfo_to_dict(url)["dbname"]),
        str(source / "database.dump"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["backup", "restore"])
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    if args.operation == "backup":
        backup(os.environ["DATABASE_URL"], args.snapshot)
    else:
        restore(args.snapshot, os.environ["RESTORE_DATABASE_URL"])
    print("PostgreSQL operation complete; no private connection details logged")


if __name__ == "__main__":
    main()
