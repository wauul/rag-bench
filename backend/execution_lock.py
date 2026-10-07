"""One benchmark worker per DATA_DIR, including local recovery commands.

OS advisory locks release on process death; no stale leases or quota-consuming takeover.
"""

import hashlib
import os
from contextlib import contextmanager
from threading import local

_owned = local()


class ExecutionBusy(RuntimeError):
    pass


@contextmanager
def execution_lock(store, name="execution.lock"):
    if store.postgres:
        # Idle API instances must not keep Neon awake with a lifetime server session.
        if name == "server.lock":
            yield
        else:
            with postgres_lock(store, name):
                yield
        return
    key = str(store.root / name)
    if key in getattr(_owned, "paths", set()):
        yield
        return
    with (store.root / name).open("a+b") as handle:
        try:
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ExecutionBusy("An evaluation already owns this data directory") from None
        try:
            _owned.paths = getattr(_owned, "paths", set()) | {key}
            yield
        finally:
            _owned.paths.remove(key)
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def postgres_lock(store, name):
    from backend.postgres import connect

    key = (store.database_url, name)
    connections = getattr(_owned, "connections", {})
    if key in connections:
        assert_execution_lock(store, name)
        yield
        return
    lock_id = int.from_bytes(
        hashlib.sha256(("ragbench:" + name).encode()).digest()[:8], "big", signed=True
    )
    with connect(store.database_url, autocommit=True) as db:
        if not db.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,)).fetchone()[0]:
            raise ExecutionBusy("Another worker owns this database")
        _owned.connections = {**connections, key: db}
        try:
            yield
        finally:
            _owned.connections.pop(key, None)
            # Closing the session releases the lock after exceptions/process death too.


def assert_execution_lock(store, name="execution.lock"):
    """Abort before the next stage/write if the lock's original session was lost."""
    if store.postgres:
        db = getattr(_owned, "connections", {}).get((store.database_url, name))
        if db is not None:
            db.execute("SELECT 1")


def execution_connection(store):
    return getattr(_owned, "connections", {}).get((store.database_url, "execution.lock"))
