"""Resume a local run using the same checkpoints as API retry. Stop the local API first."""

import sys

from dotenv import load_dotenv

load_dotenv()


def recover(store, run_id):
    from backend.execution_lock import execution_lock

    with execution_lock(store):
        return _recover(store, run_id)


def _recover(store, run_id):
    from backend.pipeline import execute_run, now
    from backend.run_state import validate_retry

    run = store.get("run", run_id)
    validate_retry(run)
    from backend.provenance import validate_snapshot

    validate_snapshot(store, run)
    run.update(
        status="queued",
        stage="Queued to resume missing work",
        attempt=run.get("attempt", 1) + 1,
        retried_at=now(),
    )
    store.save("run", run, run_id)
    result = execute_run(store, run_id, retry=True)
    print(
        result["status"],
        "prior attempts retained:",
        len(result.get("retry_history", [])),
        flush=True,
    )


if __name__ == "__main__":
    from backend.storage import Store

    recover(Store(), sys.argv[1])
