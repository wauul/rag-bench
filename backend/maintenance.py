"""Background maintenance uses operator scope and remains independent of generation."""

import asyncio
import logging
import time

from backend.alerts import deliver, enqueue
from backend.execution_lock import ExecutionBusy
from backend.retention import sweep

log = logging.getLogger(__name__)


async def maintain(store):
    next_retention = 0.0
    while True:
        try:
            if time.monotonic() >= next_retention:
                try:
                    await asyncio.to_thread(sweep, store)
                    next_retention = time.monotonic() + 86400
                except ExecutionBusy:
                    pass
            await asyncio.to_thread(deliver, store)
        except Exception:
            log.error("Workspace maintenance failed; will retry")
            try:
                await asyncio.to_thread(enqueue, store, "maintenance-failed")
            except Exception:
                log.error("Maintenance alert could not be queued")
        await asyncio.sleep(60)
