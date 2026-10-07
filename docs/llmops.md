# LLMOps, privacy and evaluation

## Four workflows, one optional destination

The shared layer in `backend.observability` wraps benchmark, investigator, optimization and debugger execution. Durable database records remain authoritative. Stable SHA256-derived trace IDs correlate workflow/record across restarts; execution attempt, revision, immutable fingerprint, configuration/question and origin/optimization IDs are allowlisted metadata. Existing measured phases cover indexing/model preparation, retrieval, reranking/context selection, generation and Ragas judging. Investigator graph nodes cover evidence, generation/validation/repair. Optimizer nodes link to original benchmark trials. Debugger nodes close execution before a durable review pause; explicit resume starts another closed execution segment. No human-wait span remains open.

This is custom instrumentation, without LangChain callbacks or automatic provider instrumentation. Provider attempts and returned numeric tokens are recorded once at HTTP transport boundaries; absence is unknown. Numeric Ragas scores attach only to original benchmark rows through stable idempotent score IDs; coverage/missing/error events accompany them. Investigation hypotheses and debugger/replay outputs never become benchmark scores. Saved work is authoritative and must not be interpreted as another provider call. Telemetry can be dropped and is not an accounting ledger.

## Explicit backend configuration

The owner confirmed project `cmuy5n8jc000mad0kjp3qkz0g` for metadata-only tracing. Browser access to its EU Cloud page was verified; backend credentials/ingestion are a separate activation check.

| Setting | Meaning |
| --- | --- |
| `RAGBENCH_LANGFUSE=false` | Default, no Langfuse export |
| `LANGFUSE_PROJECT_ID` | Must equal the confirmed project |
| `LANGFUSE_BASE_URL` | Confirmed EU project uses https://cloud.langfuse.com; US URL is only for a verified US project |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Backend-only project credentials; never dashboard/client variables |
| `RAGBENCH_DEBUGGER_TRACING=metadata` | Separate explicit opt-in; debugger remains disabled otherwise |
| `RAGBENCH_LANGFUSE_EXPERIMENTS=metadata` | Separate explicit saved-experiment export opt-in |
| `RAGBENCH_PROMPT_LABEL` | Optional explicit operator-managed remote prompt label; absent uses local prompts |
| `RAGBENCH_LANGSMITH=false` | Langfuse activation rejects simultaneous explicit LangSmith enablement |

There is **no content-export mode**. Documents, passages, questions, references, prompts, answers, diagnostic quotations, exceptions, headers and credentials are not accepted metadata. SDK masking and a final OTLP protobuf boundary remove input/output, scope keys, events and links. Actual SDK payload tests include sensitive fixtures and secret values. Project membership does not prove backend keys belong to that project: verify credential authentication and a synthetic metadata trace before enabling production.

The pinned SDK is Langfuse 4.17.0, with dedicated OTel provider, bounded span queue, 32-span batches, 128-score application queue, 2-second single-attempt network calls, zero export retries and a maximum 250ms caller wait for shutdown flushing. A full queue drops telemetry. SDK private-resource shutdown access is a tested pinned compatibility contract; SDK updates need the masking/outage tests. External failures cannot change results, Groq budgets or recovery. Crash/outage can lose tracing; preserve durable local records.

Dashboard links come from an authenticated backend endpoint and explicitly state ingestion is unverified until operator evidence exists. Links require Langfuse project membership. Provision keys only after reviewing intended scope; keep project restricted to the owner/approved collaborators, no public sharing. EU region was verified from the Cloud destination, but contractual residency/subprocessor assurances are not certified by this code. Set a reviewed retention/deletion policy in Langfuse if supported by the current plan; otherwise use approved periodic manual deletion. Free Hobby plan capabilities and configured retention must be inspected before claiming an enforced TTL. No local deletion propagates automatically to traces, scores, datasets, experiment items or backups.

## Prompts, models, datasets

Generation and investigator prompts have local versioned fallbacks. Exact generation text/version/fingerprint is saved in existing provenance; investigator resolved text is saved in its snapshot. Fixed Ragas judge templates/examples/settings and dependency versions are captured without changing Ragas semantics or fixed judge embeddings. Remote labels are resolved once at creation, saved and used on resume; an explicit remote label that cannot resolve fails instead of silently changing prompts. Remote prompts are bounded plain system text; reference answers remain excluded from generation inputs. Model IDs and pinned local revisions are recorded. Groq hosted model revisions are marked `provider-mutable-unavailable`, never claimed immutable.

`sample_data/dataset-manifest.json` identifies fictional development smoke data, not approved production/generalization evidence. The optimizer preserves seeded search, frozen development/held-out split, explicit approval, HTTP ceilings, stable trials, baseline tie behavior and frozen selection. Held-out results do not reopen tuning. No optimizer winner, investigator draft or debugger replay auto-deploys. Sanitized failures become regression cases only after a reviewer checks confidentiality, creates a synthetic case and records its origin/rubric.

```powershell
# Run inside the exact candidate image; explicit spend, fictional data, bounded HTTP budget.
python -m scripts.live_evaluation --approve-provider-spend --max-requests 48 --samples 2 --image ghcr.io/wauul/rag-bench-compact@sha256:DIGEST --output reports/live-evaluation.json
# Export already saved native optimizer experiment metadata; never execute trials.
python -m scripts.langfuse_export EXPERIMENT_ID --max-items 128
```

Native Langfuse experiment export creates fingerprint-only dataset slots and links them to original benchmark traces. It never uploads input text/expected answers or uses a Langfuse runner to bypass execution locks/budgets. Ingestion and cross-view score association need live credentials to verify. Existing scores can be re-exported idempotently; metadata events themselves can appear more than once across attempts.

## Human review and promotion

Rubric `ragbench-review-1`: verify exact artifact/dataset/prompt/model/evaluator fingerprints; independent sample/held-out coverage; factual support/citations and grounding; usefulness; missing scores/errors/quota; privacy; unexpected regressions; and measured baseline variability. Record reviewer, rubric version, decision and evidence revision. A diagnosis alone is not an approval.

Promotion report fields: revision, image, timezone-aware created_at, dataset_fingerprint, prompt_fingerprint, model, evaluator_fingerprint, sample_count, exploratory=false, complete=true, errors/quota_failures empty, all four scores and valid_counts, human_review. Reviewed policy fields: same four configuration fingerprints/model, reviewer, baseline_runs>=3, minimum_samples>=2, maximum_age_hours and minimum_scores for all four fixed metrics. Derive thresholds from repeated baseline measurements and evaluator variance; review meaningful deployment coverage, not just minimum schema counts. Fail missing, stale/future, quota/error, insufficient-sample or incomplete evidence. Smoke reports cannot authorize promotion.

Prompt/model promotion is a new reviewed snapshot and evaluation, followed by protected deployment; rollback selects a prior compatible prompt/model/release, without changing existing immutable work. Keep original environments for recovery when source fingerprints differ. [Official SDK](https://python.reference.langfuse.com/langfuse) and [advanced instrumentation](https://langfuse.com/docs/observability/sdk/advanced-features) were checked for the actual installed API.
