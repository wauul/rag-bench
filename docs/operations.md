# Operations and recovery

Authoritative data is PostgreSQL when `DATABASE_URL` is configured: documents, questions, configurations, benchmark rows/snapshots/cancellation, investigations, experiments, debugger records and native graph checkpoints. SQLite is the local fallback. Chroma indexes and model caches are rebuildable, but model revisions/ONNX manifests must still match provenance. Shared API/dashboard secrets authorize the entire workspace; there is no per-user/tenant isolation.

## Backup and restore

`python -m scripts.backup` protects local SQLite/files only and now refuses PostgreSQL mode. PostgreSQL backup uses a consistent `pg_dump` custom archive of public application/checkpoint tables and their sequences/indexes, checksummed with a schema manifest. It intentionally excludes database roles, provider extension setup and caches. Use PostgreSQL clients compatible with the server (17 in tests), a direct TLS connection and an empty isolated destination. Stop writes and workers to establish a recovery point; a dump during writes is transaction-consistent but is not a zero-loss handoff. Secrets go through libpq environment, not command arguments/logs.

```powershell
# DATABASE_URL is supplied privately by the operator; do not print it.
python -m scripts.postgres_backup backup PRIVATE_SNAPSHOT_DIRECTORY
# RESTORE_DATABASE_URL targets a disposable empty database.
python -m scripts.postgres_backup restore PRIVATE_SNAPSHOT_DIRECTORY
# Optional official Docker client when native pg_dump/pg_restore are unavailable:
$env:RAGBENCH_PG_DOCKER='1'
# Local SQLite: stop the API first.
python -m scripts.backup backup data PRIVATE_SQLITE_SNAPSHOT
python -m scripts.backup restore PRIVATE_SQLITE_SNAPSHOT NEW_EMPTY_DATA_DIRECTORY
```

Restore validates the archive checksum and refuses any destination with user tables/views/sequences. Restore runs transactionally without `--clean`. Do not connect other application writers until it finishes. The disposable recovery suite verifies all four record kinds, cancelled work and paused/interrupted checkpoints followed by explicit resume; tests never restore into production.

## Runbooks

| Trigger | Action |
| --- | --- |
| Deployment fails | Keep the last known live revision, inspect sanitized build/deploy status, verify credential/auto-deploy/ref setup, rerun only after identifying failure. Do not switch storage to make health pass. |
| Lock loss / replacement process | Stop provider work; persisted workflow becomes interrupted/failed. Confirm previous worker has stopped and DB is reachable, inspect saved attempts/reservations, then explicitly resume missing work. PostgreSQL session/advisory locks require a direct connection, not transaction pooling. |
| Interrupted benchmark | Preserve completed passages/answers/metric checkpoints. Retry missing work at the original compatible snapshot. Resume is not permission to change the generator or judge. |
| Investigation failure | Inspect saved structured hypothesis/validation; repair calls are bounded. Cancel and explicitly resume, preserving citations and separate usage. Suggestions remain hypotheses. |
| Optimizer interruption/quota | Inspect durable reservations and plan approval. Old cancellation guards remain tied to their attempt; a new explicit attempt does not reset spent reservations. Incomplete candidates cannot win. |
| Debugger paused | Review saved context before explicit generation. Resume uses the saved exact prompt/context. Historical replay is new debug evidence and never overwrites benchmark scores. |
| Database outage | Stop writes/spending; restore connectivity without clearing DATABASE_URL. Inspect locks and interrupted records before explicit recovery. |
| Provider model unavailable / quota | Check the configured model against provider availability/deprecation docs, fail without substitution. New model means a new approved snapshot/evaluation. Wait for quota reset instead of unbounded retries. |
| Disk full | Stop writes, preserve authoritative database/backup, measure caches and rebuild only disposable indexes/models after space is available. Never purge evidence to hide errors. |
| Credential exposure | Disable external tracing/provider operations, revoke the exposed credential at its provider, rotate the corresponding backend/GitHub/Streamlit secret, scan repository/history and review logs. Do not paste credentials into issues. |
| Restore | Validate checksum into an empty disposable DB, validate four workflows/checkpoints at matching revision, then review any production connection switch separately. |
| Rollback | Follow deployment compatibility checks; preserve PostgreSQL and the original environment for interrupted snapshots. Code rollback is separate from data rollback. |

There is a provider-response-before-durable-save uncertainty window. A crash can leave a reservation without a saved response, or require repeating an unsaved call after explicit resume. Exactly-once billing is not promised. Optimizer HTTP attempts include durable reservations and SDK retries. Per-call output settings and investigator repair bounds are not aggregate token or wall-time ceilings; no hard aggregate token/time limit is claimed.

## Retention and monitoring

| Evidence | Current deletion behavior |
| --- | --- |
| Independent benchmarks | Authenticated DELETE /api/runs/{id}, only after active work stops; removes associated snapshots, retrieval artifacts, cancellation, investigator records and graph checkpoints. Shared inputs/configurations remain. |
| Investigator | Associated evidence is removed with its originating independent benchmark; no separate automatic TTL. |
| Optimization | Reports and trial evidence are retained; trial deletion is rejected. No automatic TTL or destructive experiment-delete command is introduced. Offline retention needs a reviewed coherent archive of experiment, trials, snapshots and checkpoints. |
| Debugger | Authenticated debugger DELETE removes the debug record and its checkpoint when admission allows; retained replay evidence is separate from the originating benchmark. |
| Langfuse/backups | Separate operator retention/deletion; local API deletion does not propagate. |


Local benchmark/investigation/optimization/debugger records have no automatic TTL: they remain until an operator uses supported application deletion or an approved offline retention procedure. Shared uploaded inputs/configurations and immutable copies are retained independently of benchmark deletion. Backups also contain sensitive evidence; restrict access, encrypt at rest in operator storage and apply an explicit retention schedule (recommended starting policy: 7 daily/4 weekly backups after a restore drill). This recommendation is not configured automation. Do not delete dependency records of active work or claim local deletion propagates to Langfuse.

```powershell
python -m scripts.monitor --output reports/operations.json
python -m scripts.monitor --policy REVIEWED_ALERT_POLICY.json --output reports/operations.json
```

On-demand numeric monitoring separates the four workflows and optimizer-owned benchmark trials. It reports completion/error rates, measured latency samples, attempts/reservations and reported-token coverage. Spend remains unknown without reviewed provider pricing; no invented dollar estimate is emitted. Alert policies require a reviewer and at least three measured baseline runs, per-workflow sample minimums, error-rate and missing-usage limits. Investigate triggered record types and quota before explicit resume. There is no keep-alive schedule or notification integration. Langfuse retention/access/deletion is a separate operator policy (see LLMOps).
