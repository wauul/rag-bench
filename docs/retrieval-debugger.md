# Retrieval debugger

Open **Retrieval debugger** in the Streamlit workspace, choose an uploaded document
set and configuration, enter a question, and click **Run retrieval**. Inspect the
candidates and final context, then explicitly **Generate answer from saved context**.
The optional combined action performs both steps. Changing settings shows a warning;
the old trace and answer remain associated with their original settings until rerun.

Debug runs are stored as `debug_run`, never `run`; they have separate profiling and
never invoke Ragas, require references, or change benchmark scores. They share the
workspace bearer-token authorization and single-worker executor/execution lock.
This is a shared workspace, not a multi-tenant application: authorized users can
inspect all workspace data. No new per-user isolation is claimed.

## Evidence and schema

Version 1 records immutable question/configuration and document hashes, optional
benchmark origin, candidate order, reranked order, selected passages, context hash,
actual system/human prompt messages, answer, stage timings, model/pipeline/package
provenance, index reuse, reported usage, HTTP attempts through the existing profiler,
sanitized errors, and cancellation status. Full corpus inputs remain private API
storage and are excluded from exports. Documents and generated content use plain
text rendering. The shared system prompt treats passages as untrusted data; this
instruction mitigates prompt injection but cannot guarantee model compliance.

`distance` is Chroma cosine distance (lower is better). `rerank_score` is the shared
CrossEncoder output (higher is better), not a calibrated probability. Missing values
are unavailable, never zero. Scores across embedding models/rerankers are not directly
comparable. The shared selector takes the first `context_k` passages after optional
reranking; there is no additional context-budget truncation. The exact messages are
persisted before invocation and checked against the invocation. No local token
estimates are presented as provider measurements. Provider usage can be unavailable;
HTTP attempts and tokens, including retries, remain in the separate debug profile.

Source markers such as `[1]` link to supplied passages; invalid markers are flagged.
A valid marker does not establish that a claim is supported.

## Persistence, pause, interruption, cancellation

The typed LangGraph validates, prepares/reuses the shared Chroma index, retrieves,
optionally reranks, selects, commits retrieval, interrupts for review, constructs the
prompt, generates and commits the answer. SQLite/PostgreSQL checkpoints use the same
backend as benchmark graphs. Checkpoints contain primitive IDs/hashes/stages and
an explicitly supported JSON interrupt envelope, not models, clients, keys or corpus.
Review uses LangGraph `interrupt` and `Command(resume=...)`; the worker and network
request finish while the user reviews. Resume checks the context/configuration and
implementation fingerprints before any model call. Generation never retrieves again.

Application records are authoritative. Nodes skip already committed candidates,
ordering, context or answer if graph checkpoint persistence was interrupted. A server
restart marks active debug runs failed; explicit resume reconciles the saved stages.
Like benchmarks, a process death after a provider response but before durable answer
commit can cause a repeat provider call; exactly-once provider billing is not claimed.
Cancel is a durable separate signal checked between stages. In-flight CPU/provider
work finishes before cancellation takes effect. Provider retries use the existing
one-retry, 120-second ChatGroq configuration; explicit debug resumes are capped at
four execution attempts. Cancelled runs can be inspected or rerun as a new trace.
External tracing is disabled for debugger execution.

## Historical evidence and replay

**Results → Inspect answers → Open retrieval debugger** opens recorded retrieval
evidence and context/answer. Older results may lack candidates, scores, exact prompt,
usage or budget metadata. These remain unavailable. New benchmark generation records
the shared exact prompt messages. **Replay with these settings** creates a separate
new debug trace using the original immutable document snapshot when present; legacy
runs without snapshots read the currently stored document set. A replay never replaces
or describes the original execution, and uses the currently configured generator.

## Comparison, limits, retention and deployment

Comparison requires the same question and document content fingerprint. It displays
setting changes, retrieved chunk identities/ranks, context differences, side-by-side
answers, timing and usage with index cache conditions. Compatible chunking uses
source/page/token bounds/text hashes rather than Chroma's collection-local numeric
IDs. Different chunking disables exact correspondence. No model declares a winner.

Questions are limited to 2,000 characters; existing ingestion limits apply (10 files,
5 MB/file, 150,000 characters/file, 2,000 chunks). Candidates are capped at 40 and
final passages at eight. History is capped at 100 saved traces per workspace; delete
old traces explicitly. API trace/export responses are capped at 4 MB. Deletion removes
the trace, cancellation signal and its graph checkpoints but retains shared Chroma
indexes for other runs. Index caches have operator-managed retention under DATA_DIR;
no automatic shared-cache eviction is performed. Debug document snapshots remain
independent of later deletion of originating benchmark results.

No new services or dependency pins are required. Existing backend/dashboard images
include the feature after redeployment with the same persistent storage and auth.
Local changes do not imply hosted availability. Run `python -m scripts.debugger_smoke`
for one real local retrieval and, if GROQ_API_KEY is configured, one real Groq answer
using a synthetic document in isolated `data/debugger-smoke`. It never calls a judge.

LangGraph mechanism: [official interrupt reference](https://reference.langchain.com/python/langgraph/types/interrupt).
