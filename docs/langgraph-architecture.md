# Two execution engines

`Configuration.engine` is `existing` (default) or `langgraph`. Historical JSON without
this field remains the existing engine. The same 2–4 configuration limit applies;
engine identity is preserved in snapshots, history, answer inspection, and CSV.
Selecting a framework is not a claim about answer quality.

## Responsibilities and boundary

FastAPI keeps its single-worker scheduler, cooperative cancellation, and explicit
retry endpoints. `pipeline.py` dispatches each configuration. The baseline keeps
its batch retrieval and imperative evaluation. `chains.py` provides LangChain
Documents, a BaseRetriever backed by existing Chroma/embedding functions, a
Runnable passage selector, formatting, ChatPromptTemplate, ChatGroq, and validation.
The generation chain accepts only a question and Documents. References enter only
Ragas evaluation. Passage text stays data inside the unchanged generation prompt.

`graph_pipeline.py` compiles a typed StateGraph for each configuration/question:

```text
validate → prepare_index → retrieve → rerank OR select → generate
    ↘ reuse candidates/passages/answer        ↓
      evaluate each missing metric → finalize
```

There are separate nodes for all four metrics, with conditional edges skipping
valid saved scores. A failed metric is attempted once per explicit benchmark
attempt; other metrics can still complete. Generation failure routes to finalization
with null scores. Maximum graph supersteps: 20. No agent, rewriting, answer revision,
tools, self-correction or parallel judge calls are introduced.

The graph boundary keeps state small and checkpoints frequent. Its typed state holds
only benchmark/configuration/question IDs, fingerprint, graph version and stage.
Clients, embedding models, passages, answers, references and credentials are absent.
Each configuration shares a persistent Chroma index. A completed fingerprint/count
manifest permits reuse; incomplete indexing re-upserts the same chunk IDs. Retrieval
candidates are committed before reranking, and final passages before generation.
Transient models are released per question to avoid retaining retriever/reranker
weights with judge embeddings. Unlike the baseline's batch retrieval, this can
repeat model loads; performance comparisons must include those costs.

## Storage and commit ownership

| Owner | Data | Location under DATA_DIR |
|---|---|---|
| Application Store | Inputs, immutable snapshots, candidate artifacts, answers, individual metric commits, history, profiles | `bench.sqlite3` |
| Chroma | Vector index, chunk metadata and ready manifest | `chroma/` |
| LangGraph SqliteSaver | Stage history and orchestration references | `langgraph.sqlite3`, including WAL/SHM while open |
| OS advisory lock | Single execution ownership across API and recovery processes | `execution.lock` |

The benchmark ID is the API's run ID. A separate LangGraph thread ID is
`work-` + SHA-256 of `[benchmark_id, configuration_id, question_index]`.
Checkpoints use a JSON-only serializer, native PostgreSQL storage (or local SQLite FULL synchronization), and LangGraph
`durability="sync"`. No pickle/object deserialization is used by this checkpointer.
The connection lives for one configuration and closes in `finally`.

New runs snapshot source documents, question/reference pairs, configurations,
prompt/settings, implementation versions, dependency versions and model revisions.
Uploaded source edits do not change this private snapshot. Execution and retry
validate its hash and the current implementation/settings before model work.
Unknown graph snapshots and incompatible graph checkpoints are rejected. Legacy
baseline retries retain their older, less complete provenance checks.

Local model/tokenizer revisions are pinned to official Hugging Face commit hashes.
Both inference engines use them; old ONNX artifacts need rebuilding when their
manifest lacks the matching revision. Groq model IDs/settings are recorded, but
provider-side weights behind a model ID cannot be verified or frozen by Ragbench.

## Recovery and interruption

Application results are authoritative. Each explicit retry re-enters graph validation
using the **same persistent thread**, rather than trusting a possibly stale cursor.
Completed expensive stages are routed around using durable application commits.
This intentionally retains checkpoint history while reconciling at the entry point;
it does not blindly invoke a saved cursor after a crash. Application commits
and graph commits are not a distributed transaction.

| Interruption boundary | Next explicit retry |
|---|---|
| Before provider call | Run missing stage |
| Response received, before application save | May repeat provider call; outcome was not durable |
| Application answer/score saved, before graph checkpoint | Reconcile and skip saved result |
| Graph says complete but application score is missing | Reconcile and run only missing metric |
| Answer missing/invalid but scores remain after a partial restore | Invalidate orphaned scores; regenerate and score the replacement answer |
| Between metrics | Retain completed metrics; score remaining metrics |
| Incomplete index | Idempotently upsert index batches; mark ready only at the end |
| Saved candidates, before reranking/final passage commit | Reuse candidates and finish selection |
| Cancellation during a provider call | Save returned answer/score, then stop at next boundary |
| Process restart | Mark active run failed; no provider calls until explicit retry |

There is **no exactly-once provider guarantee**. Response-to-save loss can consume
quota twice. An in-flight SDK retry or a metric's internal completion sequence is
not forcibly cancelled; cancellation is checked between stages and metric calls.
Native model loading/index operations are not preempted; indexing checks each batch.
The execution lock prevents duplicate work, including offline recovery, and releases
on process death. Operate one API process/worker per DATA_DIR; this is not a
multi-replica scheduler. Stop the API before using the local recovery command.

`DELETE /api/runs/{id}` requires an idle worker and removes associated graph threads,
candidate artifacts, input snapshot and Chroma collections before removing the run.
Shared uploaded documents/test sets/configurations remain for other runs. An
interrupted deletion can be repeated. No automatic retention interval was added.

## Provider, retry and tracing policy

Both engines use the same ChatGroq and GroqRagasLLM implementations, fixed MiniLM
evaluation embeddings and four Ragas algorithms. Strictness is 3; Groq receives
separate `n=1` requests. Generation temperature is 0 and max output is 2048 tokens.
The prompt's untrusted-passage and insufficient-evidence instructions are unchanged.

The SDK owns transport retries: `max_retries=1` means **at most two HTTP attempts per
logical completion**. Timeout is 120 seconds per HTTP attempt; SDK backoff honors a
Retry-After value up to 60 seconds. Logical completions are paced by the existing
rate limiter; SDK retries use their own backoff. Ragas outer attempt count is 1,
output-repair retries are 0, metric timeout is 240 seconds, and the graph has no
provider retry policy. Missing work is retried only on user request. A one-passage
question normally needs 8 logical calls: one answer, two faithfulness, three
relevancy, one precision and one recall; at most 16 HTTP attempts. More final
passages add one precision completion each. Failed/empty answers and invalid scores
remain errors/nulls. Relevancy permits [-1,1]; fraction metrics require [0,1].

Failures retain a retryable-provider versus validation/configuration classification.
This is diagnostic, not an automatic retry trigger. LangSmith is disabled inside
execution even if global LangChain tracing variables are set. Explicitly setting
`RAGBENCH_LANGSMITH=true` opts into external tracing and may transmit benchmark
content; it is neither required nor enabled by deployment templates.

## Fairness, performance and versions

Compare matching documents/questions, chunk parameters, embedding revisions,
candidate_k/context_k, reranking, generation/judge models, prompt examples and
inference precision. Model-provider nondeterminism means live answers need not be
byte-identical at temperature zero. Deterministic tests compare passage identities,
metadata, order and complete prompt content between engines.

Run provenance records dependency versions, pipeline/graph versions and relevant
settings; exports include this JSON. Version changes reject new-format retries.
Keep historical baseline exports usable; absent engine identity means `existing`.

Existing profiler phases remain distinct. Graph additions include index reuse,
passage selection and orchestration setup. Rows record cumulative graph wall time
and time outside graph nodes (framework/checkpoint overhead). Node time includes
stage persistence; overhead is not a pure framework CPU benchmark. Answer latency
still measures generation/scoring work; detailed spans separate provider and setup
time. Interrupted measurements are lower bounds. Model download-cache warmth is
unknown, not guessed; compare explicitly controlled cold/warm runs. A reused index
or persisted answer must not be advertised as faster fresh inference.

Dependencies are pinned in `pyproject.toml` and `uv.lock`, including the native
PostgreSQL saver and psycopg binary driver. The current set uses LangGraph 1.2.14,
checkpoint 4.2.0, PostgreSQL saver 3.1.2 and SQLite saver 3.1.1, with Ragas 0.3.9.
The Ragas embedding wrapper's deprecation remains; replacing it is separate work.

## Hosting and rollback

Checked-in `render.yaml` retains free ephemeral compute and requires an external
`DATABASE_URL`. PostgreSQL owns application records and graph checkpoints; the
local Chroma index is rebuilt from snapshots if lost. With no database URL, local
SQLite still requires a persistent volume. See [Neon storage](neon-storage.md) for
setup, migration, free-tier limits, verification and rollback details.

The paid disk example remains an optional, unapplied alternative. Never attach a
paid disk to implement this free hosting path. Select `existing` for new baseline
configurations without changing historical engine identities. Do not roll back to
a SQLite-only release while using a PostgreSQL database.
