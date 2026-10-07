# Failure Investigator

Open Results → Inspect answers → Failure Investigator → Investigate result.
An answer or saved context is enough; complete Ragas metrics are not required.
Review the additional usage disclosure first. Optional source search uses a single
question query with at most 12 chunks, confined to the frozen document snapshot.
Saved versions reopen automatically. Refresh to see the stage, cancel active work,
resume interrupted work, or use Run again to create another report version.
Each suggested experiment can open an editable configuration draft in Setup.
Creating a draft neither saves a configuration nor launches a benchmark.

## Diagnostic scope

The report distinguishes retrieval misses, context selection loss, unused useful
context, unsupported claims, apparently inadequate sources, reference ambiguity,
technical/incomplete evaluation, and insufficient evidence. These are hypotheses,
not established root causes. Qualitative strength includes a rationale. Missing
scores are not treated as low scores. Ragas internal reasoning is not available.

New runs in both engines persist fingerprinted retrieval traces: original candidate
order, complete post-rerank order and scores, and the selected context. Reranking-loss
hypotheses must cite a candidate moved from inside the original context cutoff to
outside it. Context-selection hypotheses must cite an excluded candidate. Historical
graph candidate artifacts remain usable, but missing historical ordering cannot be
reconstructed; the report discloses that gap and rejects reranking-loss assertions.
An investigation search result is labeled `search`, never original `context`.
Failure to find a passage with one query does not establish absence from the corpus.
Older runs without an immutable source snapshot can inspect saved context but
cannot search historical sources. No source material is silently taken from a
different uploaded document set.

Model explanations remain fallible. Exact citation checks establish that quotes
exist, not that a diagnosis logically follows. Source instructions remain untrusted
data in the system prompt. Schema checks, exact quotation checks, trace availability
guards, and unsupported-certainty/experiment-claim checks reject invalid reports;
they cannot fully verify arbitrary natural-language reasoning. Conditional
experiment wording and configuration changes are application-generated, never
accepted from the model as claims of successful trials or guaranteed improvement.

## Data and schema

`investigation` objects use the existing authoritative SQLite/PostgreSQL Store.
They copy the benchmark row, original configuration, available candidates, and
source snapshot. The SHA-256 fingerprint covers these inputs. They never edit
benchmark answers, contexts, configuration, scores, completion, or profiling.
References are diagnostic inputs only; benchmark generation's existing input
allowlist remains unchanged.

Report schema version `2` includes identity (`id`, `run_id`, `configuration_id`,
`question_index`, report `version`), input fingerprint, source snapshot identity,
status/stage/timestamps/sanitized errors, `diagnosis` (summary, hypotheses with
category/strength/rationale/supporting/contradicting quotes), evidence catalog,
limitations, conditional experiments, provenance and separate request attempts.
Catalog keys such as `context:0`, `candidate:0`, `search:0`, `answer`, `question`,
`reference`, and sanitized `evaluation` resolve exact excerpts. Answer-use and
unsupported-claim hypotheses must cite an exact answer excerpt. Chunks preserve chunk ID, source, page,
token offsets and distance when available; `source_id` is derived from snapshot
identity and source name. Their origin is explicit. Null/missing original IDs are
not invented; catalog keys identify the exact stored passage.

Provenance records model, graph/prompt/schema version, prompt fingerprint and
installed dependency versions. Attempts store start time, elapsed seconds and
provider-reported usage when available; unavailable usage stays unknown. Usage
does not enter benchmark profiling or metric averages. Public JSON/Markdown
exports omit the frozen full corpus and raw provider output. Technical details
are expandable in the dashboard.

## Execution, limits and recovery

The existing one-thread executor, worker admission lock, and cross-process data
execution lock are reused. Requests have an idempotency key; repeated starts
reopen the existing report, and explicit Run again creates a new version. Routes
share the existing bearer-token authorization boundary: Ragbench is a shared
workspace, not a multi-tenant application. Production requires its existing API
token and dashboard password. Source search cannot select another corpus.

The typed LangGraph runs load snapshot → validate evidence → inspect original →
optional source search → generate hypotheses → validate references → persist.
SQLite/PostgreSQL checkpointers use the existing JSON-only serializer and
`durability="sync"` ([LangGraph durability reference](https://reference.langchain.com/python/langgraph/types/Durability)).
Application stage commits and saved model outputs are authoritative on resume.
Completed retrieval/inspection/search stages are skipped; a saved model response
is validated before any further request. Graph state holds references, not a
second mutable report. Graph cursors never override the frozen application inputs.

Each investigation permits at most two provider requests total (initial plus one
repair/retry), no SDK retries, 60 seconds per request, and 3,000 output tokens per
request. Input snapshot JSON is limited to 1.5 MB, answer to 24,000 characters,
and model evidence JSON to 80,000 characters; oversized model inputs fail honestly.
Search does not generate model queries or rerank. Local model/index preparation
uses existing bounded inputs and cancellation checkpoints; a native embedding
operation cannot be preempted mid-call. LangSmith tracing is disabled unless the
operator explicitly sets `RAGBENCH_LANGSMITH=true`, matching benchmark execution.
Model failures retain sanitized messages, not raw provider errors.

An attempt is committed before the provider call, and its output immediately
afterward. A crash after provider response but before save is an uncertainty
window: the reserved attempt remains consumed, but the response/usage may be
unknown. One remaining request may repair/retry it. There is no promise of
exactly-once provider billing. A crash after a durable response does not repeat
it. Restart reconciliation marks active investigations failed with an explicit
Resume action; it does not silently spend quota. A cancellation signal is durable
and checked at stage boundaries. In-flight provider output is saved before
cancellation is observed. Cancelled reports are not resumed by the UI; Run again
creates a new version. Persistent disk or the configured PostgreSQL database is
required to retain reports/checkpoints across host replacement. Ephemeral storage
cannot provide durable recovery.

## API

- `POST /api/runs/{run_id}/investigations`: configuration ID, question index,
  request key, optional search_sources and run_again.
- `GET /api/runs/{run_id}/investigations?configuration_id=...&question_index=...`
- `GET /api/investigations/{id}`
- `POST /api/investigations/{id}/cancel` or `/resume`
- `POST /api/investigations/{id}/experiments/{index}/draft`: returns editable
  settings and `launched: false`; does not create a benchmark.
- `GET /api/investigations/{id}/export?format=json|markdown`

Existing deployments already install the LangChain/LangGraph dependencies.
No new service, tracing subscription or migration is needed. Deploy the backend
and dashboard together. Deleting a run deletes its associated investigation
reports and graph threads. Keep the existing persistent-storage settings.

## Verification boundaries

`tests/test_investigator.py` covers deterministic diagnoses, citation rejection,
bounded repair, source isolation, separate accounting, duplicate requests,
authorization, cancellation, original-result preservation, and fresh-process
recovery without repeating model work. Controlled HTTP tests exercise the real
LangChain/Groq structured-output adapter and its request budget. A real Chroma
index with deterministic embeddings verifies frozen-source retrieval. Dashboard
tests include the real sidebar and editable configuration form. These tests establish behavior, not real model diagnostic
accuracy. Real-provider verification requires existing GROQ_API_KEY and quota;
inspect report claims manually against original and additional source evidence.

On 2026-10-07, a bounded live Groq smoke check using `qwen/qwen3.8-27b`
completed on the synthetic refund-policy fixture: two requests including validation
repair, 2,455 total tokens. The accepted report identified the answer's unsupported
one-day claim against seven-day context. Its broader causal explanation remained
imperfect despite valid citations; this verifies provider integration and bounded
repair, not general diagnostic accuracy. No production benchmark was modified.
