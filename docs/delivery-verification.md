# Delivery verification — 2026-10-07

## Audit and changes

Baseline main was `d5ea194cf7a62d623e37dfb046f9b915a3920eb9`. Existing uncommitted debugger/backend/dashboard/documentation changes were preserved and included, not reset. Existing pipeline/LangGraph engines, CPU/ONNX inference, Groq/Ragas, snapshots, locks/checkpoints, investigator and optimizer remain the execution system. The existing uv, Ruff/mypy, security, coverage and container tools were extended rather than replaced.

Missing boundaries addressed: PostgreSQL was a separate workflow outside the aggregate gate; SQLite backup did not protect Neon; deployment assumed image-backed Render while the live service was source-backed; no reviewed immutable image publication/quality promotion path or shared Langfuse layer existed. New backup, recovery tests, source deployment, release/evaluation/deployment workflows, metadata tracing, frozen prompt provenance, native experiment export and operations/security/contribution policies address those gaps.

## Local evidence

| Check | Observed result |
| --- | --- |
| Windows Python 3.11 engineering | 170 passed, 14 opt-in skips, 72.33% coverage; unchanged coverage gate passed. Lock/exports/dependency compatibility/Ruff/scoped mypy passed. |
| PostgreSQL 17 recovery | 13 passed, including locks, replacement processes, all four workflows, real dumps/restores, interrupted checkpoints, explicit resume, checksum/nonempty rejection. Synthetic disposable databases only. |
| Tracing/prompts/quality | 22 passed; actual SDK OTLP payload masking, disabled/debugger opt-ins, transport failure isolation, correlation, score IDs, missing usage, frozen remote label and fail-closed quality evidence. |
| Native experiment SDK | 1 passed with actual installed SDK + controlled HTTP transport; fingerprint-only payload and original benchmark trace association. No external ingestion. |
| Real CPU models | 2 passed on Windows and 2 passed in Linux CPU runtime; actual pinned MiniLM/BGE embeddings, reranker, Chroma and both engines. Provider/judge fixtures, not measured Ragas quality. |
| Real compact models | 2 passed in actual Linux compact runtime; prepared pinned ONNX models, reranker and both engines, with controlled provider/judge fixtures. |
| Actual CPU/compact/dashboard containers | Both backend variants passed UID 10001, writable volumes, auth/readiness, changed port, persisted restart interruption, dashboard login/render/authenticated API history and offline SQLite backup. |
| Compose/action syntax | Compose configuration and actionlint passed. Action commits and current tool releases verified against official sources. |
| Secret/security | Gitleaks scanned 28 existing commits with no leaks; Bandit passed. Dependency audit found 9 known findings in 3 packages covered by unchanged documented, owned, expiring exceptions. Container scans apply the existing actionable finding policy, not a claim of zero vulnerabilities. |
| Live Groq availability | Authenticated model-list check returned 200; configured model available, no substitution. |
| Bounded live sample | Actual compact runtime, fictional data, 48-attempt ceiling; 33 reserved attempts, 30 token reports, 11,787 reported tokens. Three rate-limit responses left the baseline incomplete. The runner exited unsuccessfully and optimizer selected no winner. Candidate-only complete scores do not repair the baseline. Smoke evidence only. |

The first Windows real-model invocation failed because the local environment lacked the CPU/model extras. A locked sync installed the missing extras and the rerun passed. The first simultaneous Trivy invocation hit its shared-cache lock; scans were rerun sequentially. These were resolved, not suppressed. Ragas emits existing embedding-wrapper deprecation warnings; semantics/fixed judge embeddings were preserved.

Artifacts above were local images labeled `local-delivery-final`, not published GHCR releases. A final script formatting change requires a cached image refresh before exact-tree certification. Runtime tests added pytest only in disposable test containers; application runtime dependencies came from the built images. Raw machine evidence remains under ignored `reports/` and contains no production dump. Historical reports certify only their stated revisions.

## Remote verification and activation boundaries

GitHub repository admin access and owner `wauul` were verified. Main protection and reviewer-controlled release/evaluation/production environments were configured. Private vulnerability reporting was verified enabled. Remote Actions for the changed branch must be recorded separately; earlier successful main Actions do not certify these changes.

Render was visibly Free/source-backed (`Dockerfile.render`), auto-deploy Off, with the baseline commit live. The Streamlit dashboard rendered existing workspace controls against the hosted API; this proves the prior app connection, not a new deployment. No production record was changed and no paid resource/storage replacement was provisioned.

Langfuse project identity, EU region, Hobby plan, authenticated browser access and **no API keys** were observed. The owner confirmed this project for metadata-only tracing. Production tracing, actual trace/score/experiment ingestion and configured retention/access remain unverified until a scoped backend credential is provisioned and tested.

Required external work: controlled Streamlit release-branch configuration/build evidence; Render/GitHub/Streamlit secret wiring; reviewed repeated-baseline quality policy and independent evaluation evidence; protected release publication/deployment approval; Langfuse credential/retention setup. A failed or skipped activation is not working CD. See the setup and command details in [CI/CD](cicd.md), [deployment](deployment.md), [operations](operations.md) and [LLMOps](llmops.md).
