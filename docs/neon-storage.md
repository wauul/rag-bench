# Durable storage on Render Free

Use Render Free for the compact ONNX API and Neon Free PostgreSQL for authoritative
storage. No Render disk or paid database plan is required. The Streamlit dashboard
continues to use the same authenticated API.

## Configuration

- Create a dedicated Neon Free project near the API region (Oregon for this service).
- Set `DATABASE_URL` on the API to the **direct** connection URL with
  `sslmode=require` (or stricter). Do not use a `-pooler` endpoint: worker locks are
  session advisory locks, which require a stable PostgreSQL session.
- Preserve the existing `API_TOKEN` and `GROQ_API_KEY`. Set `APP_ENV=production`.
- Leave `DATA_STORAGE=ephemeral` for the local filesystem. `/health` and new runs
  report `storage: postgres` when the configured authoritative store is PostgreSQL.
- Keep one API service / one Uvicorn worker. Locks prevent concurrent benchmark
  execution across replacement instances and local recovery commands.
- Keep `RAGBENCH_LANGSMITH=false`. Database credentials never enter run provenance.

The `objects` table holds uploaded document text, tests, configurations, immutable
snapshots, answers, scores, profiling and retrieval artifacts. LangGraph's native
PostgresSaver owns checkpoint tables, using the same JSON-only serializer as local
SQLite. Chroma data under `DATA_DIR` is disposable: lost indexes are rebuilt from
saved document snapshots only when missing work needs retrieval. Completed answers
and scores are not recomputed. Lost checkpoints reconcile against saved results.

Connections close after API operations and benchmark execution so an idle database
can suspend. Health probes do not query PostgreSQL; authenticated readiness does.
Active runs hold a direct session lock; losing that session aborts subsequent work
and prevents writes through a replacement connection. Cancellation requests are
persisted by attempt so a request reaching a replacement API can stop the worker.

Free plans impose storage, compute and transfer limits and may sleep. This provides
durability, not uninterrupted execution or an uptime guarantee. Startup marks
interrupted work failed; users explicitly retry missing work. Do not use keep-alive
pings to consume the free compute allowance. Delete unwanted runs through the API
to remove results, snapshots, retrieval artifacts and graph checkpoints together.
Uploaded inputs shared with other runs are retained.

## Existing SQLite data

Stop the source API and workers. Preserve a whole-directory backup first. With the
destination `DATABASE_URL` in the environment, run:

```powershell
uv run --no-sync python -m scripts.migrate_postgres C:\path\to\saved-data
```

The destination application table must be empty. The tool copies object IDs and
JSON payloads in one transaction, and converts SQLite checkpoints through the saver
APIs. It refuses to overwrite existing application records and leaves source files
available for rollback. Interrupted checkpoint copying can be rerun while the
application table remains empty. Rebuildable Chroma indexes are not uploaded.
Dependency/prompt compatibility checks still apply to migrated unfinished runs.

The old hosted release has no history-list or database-download endpoint. Its
ephemeral files cannot be recovered after Render discards them. Export any known
run IDs before replacing that release; this migration cannot promise preservation
of inaccessible historical files. Local test fixtures are not imported into the
production database.

## Verification and rollback

`tests/test_postgres.py` runs against real disposable PostgreSQL. Set
`RAGBENCH_TEST_POSTGRES_URL` to a localhost administrator URL; the tests create and
drop unique test databases. Coverage includes empty-cache fresh-process recovery
before generation, after answer persistence, between metrics, during indexing,
cross-process locking, cancellation, history, deletion and SQLite migration.
External providers are faked in these repeatable failure tests.

Before releasing, test the compact Linux image, then verify a hosted result remains
available after service replacement. A prior SQLite-only release cannot read Neon:
rollback to a PostgreSQL-aware revision, or restore the offline SQLite backup on a
host with a persistent local volume. Removing `DATABASE_URL` selects an independent
local database and must not be presented as a successful rollback of hosted data.

Sources: [Render Free](https://render.com/docs/free),
[Neon Free](https://neon.com/blog/neon-free-plan-1-gb-per-project),
[PostgresSaver](https://reference.langchain.com/python/langgraph.checkpoint.postgres).

## Local verification on October 7, 2026

The full suite passed 96 tests with 2 optional real-model tests skipped. After adding
lock-session-loss coverage, the targeted PostgreSQL/API/recovery suite passed 19 tests.
The compact Linux image ran under a 512 MB container limit with real ONNX models,
Neon PostgreSQL, Groq and Ragas: 2 engine answers, 8 finite metric results, 18 HTTP
requests, no provider errors. Replacing that container with a fresh filesystem
preserved the completed run and both-engine CSV export without new provider calls.
These measurements are a smoke test, not an engine quality or speed comparison.
Hosted deployment verification is recorded separately from these local checks.
