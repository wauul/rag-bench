# Guided configuration optimization

Use **Optimize configuration** after uploading documents and reference questions and
saving a **LangChain + LangGraph** baseline. Planning makes no provider calls. Review
the exact configurations, split, objective, trial ceiling and HTTP-attempt ceiling,
then explicitly approve the plan. Historical configurations using the existing
engine remain supported for ordinary benchmarks; create a new LangGraph configuration
to optimize. The experiment never changes execution engines or provider models.

## Search and selection

Search changes only chunk size (32–240 WordPiece tokens), overlap (0–239, smaller
than size), supported embedding model, candidate_k (1–40), context_k (1–8, at most
candidate_k), and local reranking. Duplicate dimension values and unsupported values
are rejected. Invalid combinations are skipped and counted. Valid alternatives are
ordered by retrieval fingerprint, shuffled with a local seeded PRNG, then bounded
to `max_trials - 3`: one baseline tuning trial and two held-out checks are reserved.
The baseline is excluded from candidate alternatives. No LLM chooses settings.

Generation settings, system prompt, judge settings/examples, fixed judge embeddings,
metrics, package versions, model revisions and engine are frozen. Resume checks their
full provenance. Quota or compatibility failures never cause an automatic model switch.

Best quality uses the existing equal-weight mean of all four Ragas metrics. Every
required metric must be valid on every tuning question; errored or incomplete rows
exclude that configuration. The latency objective minimizes average retrieval,
reranking/passage-selection and generation time per question among configurations
meeting the chosen quality threshold. Indexing and judge time are separate and excluded
from serving latency. Retries accumulate: this is observed work, not a production SLA.
Baseline wins exact ties, then stable plan order breaks ties. If baseline is incomplete
no comparative winner is selected. A baseline win reports **No improvement found**.
If nothing meets the quality constraint the report says so explicitly.

## Splits and leakage

The plan freezes document text and reference questions before execution, with dataset
and plan SHA-256 fingerprints, question identities, split seed and selection policy.
Duplicate normalized question text is rejected. Six or more questions produce at
least two held-out questions (one third, rounded down) and a disjoint tuning set.
For fewer than six questions the user must acknowledge exploratory tuning; all
questions are tuning questions and there is no independent held-out check.

Selection is committed once, using tuning results only. Afterward only baseline and
the selected configuration are evaluated on held-out questions. If baseline wins,
one baseline held-out trial suffices. Disappointing held-out results remain visible
and cannot select another winner. The LangChain generation input allowlist accepts
only the question and retrieved documents. References enter Ragas evaluation only.
No global optimum, statistical significance or generalization claim follows from a
small single run, even when the held-out check looks better.

## Budgets and recovery

The enforced budget is **provider HTTP send attempts**, not graph nodes or logical
completions. Generation, every judge call and SDK retries share the gate in both
profiling HTTP clients. Each attempt is durably reserved before send, even if it
fails, returns 429, or crashes before its response is saved. The cap cannot be exceeded
by SDK retries. Tokens come from provider responses; unknown usage is not invented.
The plan's 18-attempts-per-question estimate is not a guarantee. No monetary price
is assumed. Optional token and wall-time limits are not implemented; HTTP attempts
are the enforceable bound. Local indexing/model work does not consume provider requests.

The experiment LangGraph stores compact identity/fingerprint/stage state and coordinates
tuning, selection, held-out evaluation and reporting around existing benchmark execution.
SQLite or configured PostgreSQL stores plans, reservations, stable trial references,
individual answers and scores. Application commits reconcile graph retries. Trial IDs
include experiment, plan, split and configuration. Reuse requires identical documents,
questions, configuration, models, prompts, metrics and pipeline provenance. Index reuse
is restricted to the same saved trial's fingerprint-checked index; new trials index
fresh. Reports distinguish fresh indexing, warm reuse, and trials containing both.
Reuse stays within the existing authenticated shared workspace; this is not multi-tenant.

The one-worker limit, execution lock and provider rate limiter apply. Concurrent starts
cannot duplicate trials. Cancellation sets a durable signal: new work stops while
an unavoidable in-flight response finishes, is counted and is saved. The API reports
**cancelling** during that interval. Restart marks interrupted experiments/benchmarks
failed and never automatically resumes provider-consuming work. Explicit resume
preserves usage and split; complete rows are reused and missing work is retried.
Completed reports freeze selection, including exclusions for failed candidates;
start a new experiment to retune after completed selection. Interrupted held-out
checks may resume without reopening tuning selection.

An exhausted request budget requires an explicit recorded increase with a reason
before resume. Trial ceilings cannot increase. Ordinary benchmark retries cannot bypass
experiment budgets, and experiment trial evidence cannot be deleted through ordinary
run deletion. JSON exports include provenance, split IDs, settings, coverage, failures,
measurements, usage and amendments. Save as configuration creates a new configuration
and leaves existing entries/defaults intact. Raw documents/reference answers are omitted
from experiment API responses/exports; trial evidence remains inspectable in Results.

## Deployment and audit

No dependency additions or external services are needed: pinned LangChain, LangGraph,
Chroma and existing storage adapters provide the foundation. Existing SQLite/PostgreSQL
objects storage accepts experiment/cancellation records without schema migrations.
Graph checkpoint tables are reused. Back up objects and graph checkpoints together
using the existing backup workflow. Chroma remains disposable. LangSmith stays disabled
unless `RAGBENCH_LANGSMITH=true` is explicitly configured. Deploy backend and dashboard
together using existing manifests. Local verification does not establish hosted availability.

The existing validator already enforces chunk and retrieval constraints. Benchmark
execution commits retrieval, answers and every Ragas metric independently; completeness
requires all four finite valid metrics. HTTP clients already count SDK attempts and
reported usage. Ordinary Results already provides comparisons, passage inspection and
performance exports. Immutable snapshots version inputs and pipeline implementation.
This feature reuses these mechanisms instead of introducing another evaluator.
