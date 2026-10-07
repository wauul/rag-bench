# Workspace protection

Personal access keys isolate users' objects, history, mutations and background workers.
Legacy data and the workspace password belong to the owner. PostgreSQL enables and
forces RLS; application ownership checks also apply to SQLite. Database credentials
remain operator credentials: users never receive SQL access. Public content moderation
is not provided; this remains a private benchmark tool.

Provision a 30-day key with `python -m scripts.users alice` (one-time secret output).
Revoke with `python -m scripts.users alice --revoke`; rotating a user replaces the key.
Sign in using the personal key. Sessions expire in one hour and clear cached data at
each sign-in. Revoked/expired keys are rejected by every API request.

Both API and dashboard now require access configuration by default. Set API_TOKEN
and DASHBOARD_PASSWORD (production requires at least 32 characters). Compose requires
both and sets production mode. Only an isolated local development environment may
explicitly set RAGBENCH_ALLOW_INSECURE_LOCAL=true. Use HTTPS at public ingress.

The server disables proxy header processing. Limits use the socket peer, never an
unverified forwarded address. Behind a proxy, its clients conservatively share a
limit. There is also a workspace-wide limit: 120 requests/minute, 20 writes/minute.
Dashboard login allows 20 attempts/minute per process; sessions expire after one hour.
Login throttling restarts with the dashboard process; API counters are durable.

## Operator controls

Policy lives in the authoritative database. Changes do not require a redeploy:

```powershell
python -m scripts.guardrails
python -m scripts.guardrails --set enabled=false
python -m scripts.guardrails --set enabled=true
python -m scripts.guardrails --set daily_requests=500 --set monthly_requests=5000
```

Defaults reserve at most 500 provider attempts/day and 5,000/month, across benchmark,
optimization, investigation, debugger and direct workers. Every HTTP retry consumes
another reservation, even if it fails. UTC calendar periods reset independently;
restarts, new experiments and content deletion do not reset usage. Exhausted quotas,
disabled features, corrupt policy or failed persistence deny provider sends.

Each attempt also reserves request JSON bytes plus output tokens plus 1,024 protocol
units against a conservative allowance (2,000,000/day, 20,000,000/month). This is an
admission allowance, not measured tokens or a guaranteed dollar billing ceiling.
Actual token reports remain in performance profiles. No refunds are made for missing
responses. Use provider-side spending controls too if the account is billed.

The switch stops new mutations/expensive jobs and subsequent worker stages/provider
attempts. An in-flight operation can finish. Read, cancel and deletion remain possible.
Authenticated GET /api/guardrails exposes policy/usage but cannot increase caps.
Operator changes retain the last 50 audit entries; approaching 80 percent and denial
emit content-free warnings. Set RESEND_API_KEY, ALERT_FROM and ALERT_TO in backend
secrets for email delivery. A durable outbox deduplicates events per UTC day, uses
Resend idempotency keys, limits retries to three and total attempts to 20/day. Emails
contain only fixed event names and UTC dates, never prompts, identities or credentials.

## Uploads and retention

Authoritative JSON is capped at 100 MiB across uploaded inputs, snapshots and results.
This excludes local index/model caches, database overhead, checkpoints and backups;
those still require disk monitoring and operator retention. A full authoritative
store rejects writes; cancel/worker status updates may also need space freed first.
The request body is bounded to 12 MiB and 30 seconds overall, with a 10-second idle
timeout. PDF parsing runs outside the API event loop in a disposable subprocess with
a 25-second deadline, two parser slots, and POSIX CPU/address-space limits. Windows
enforces the process deadline and concurrency limit, not an address-space limit.

Uploaded data is scoped to its owner. Selected passages, questions,
references and generated answers reach Groq. Optional telemetry has its separate
documented boundaries. Use non-sensitive benchmark data. Background maintenance runs
at startup and every day, expiring inactive data after 30 days (operator policy
retention_days, 1-365). Active work and dependencies of retained records are protected.
Legacy records without dates receive a 30-day grace period. Expired checkpoints and
run indexes are removed under the execution lock; busy workers postpone cleanup.
Backups/exports and external providers have separate retention policies.

Ordinary run/debugger deletion remains available. To remove all corpora, test sets,
optimization evidence, results, checkpoints and local Chroma caches, stop the API,
disable features, then explicitly run:

```powershell
python -m scripts.purge_workspace --confirm DELETE_WORKSPACE_CONTENT
```

This retains policy and consumed quotas. Delete offline backups/exports separately;
external provider/telemetry retention is outside this command. Purging logical data
does not guarantee secure erasure from storage media or provider backups.

Automated tests cover atomic concurrent reservations, restart persistence, failed
transport accounting, sync/async denial before send, corrupt/missing policy, token
allowances, ingress/authentication denial, storage rejection, dashboard access,
parser timeouts and quota-preserving purge. They do not establish hosted deployment
configuration or live PostgreSQL correctness unless its integration tests run.
