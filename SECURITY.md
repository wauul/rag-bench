# Security policy

This is a shared-workspace application. Production API requests require a shared bearer token; the dashboard requires its configured password. Approved users can access the workspace's records and exports. This is not tenant isolation or a public multi-user service.

## Reporting

Private vulnerability reporting is enabled and was verified for [wauul/rag-bench](https://github.com/wauul/rag-bench/security/advisories/new). Report privately through GitHub with a minimal synthetic reproducer, affected revision and impact. Do not include tokens, database URLs, documents or personal data. No dedicated security mailbox or guaranteed response SLA is claimed. The owner aims to acknowledge reports when available; this is a solo-maintained project.

Only current main and its latest reviewed release receive fixes; older releases have no guaranteed backport policy. Interrupted immutable snapshots may require their original environment for recovery. Do not expose older releases without reviewing current advisories.

## Boundaries

Keep one backend worker. Use TLS direct PostgreSQL sessions for durable state/locks; never remove DATABASE_URL as a rollback. Upload parsing is limited by type/count/bytes/pages/extracted text; exports and evidence are shared-workspace data. Render untrusted documents/answers as plain text, never executable HTML or instructions. Local evidence/export permission does not authorize external tracing.

Groq receives generation context/question and judge inputs/references needed for Ragas; investigator calls receive selected saved evidence. Those provider flows are part of explicitly started AI work. Optional Langfuse accepts only allowlisted operational metadata; debugger and experiment export have separate opt-ins. Never configure LangSmith and Langfuse together. Keys stay backend-side, are absent from CI fixtures and must not be pasted into issues or commits.

CI runs secret scanning, Bandit, dependency audits and container scans. Exceptions in the existing checked-in security policy require justification, verified owner and expiry; passing scans does not certify absence of vulnerabilities. Follow [operations](docs/operations.md) for exposure, outage and recovery response.

No legal license is selected or replaced by this work.
