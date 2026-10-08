---
title: RAG Bench API
emoji: ◈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
---

# RAG Bench

**Compare retrieval configurations. Inspect the evidence. Measure the trade-offs.**

RAG Bench is a private workspace for evaluating retrieval-augmented generation (RAG).
Upload documents and reference questions, compare two to four configurations, and
see how chunking, embeddings, reranking, and execution engines affect answer quality,
processing time, memory, and provider usage.

The Streamlit dashboard handles setup and exploration. FastAPI runs retrieval on
the CPU, calls Groq for generation and judging, and saves results for later review.

[Quick start](#quick-start) · [Benchmark workflow](#benchmark-workflow) ·
[Reading results](#reading-results) · [Documentation](#documentation)

## What you can do

- **Compare benchmarks:** question-level scores, answer inspection, source passages,
  quality/time charts, saved history, and CSV exports.
- **Debug retrieval:** inspect candidates, reranked order, final context, and exact
  prompts before choosing to generate an answer. Debug runs have separate history
  and never change benchmark scores.
- **Optimize configurations:** review a seeded search plan with a provider-attempt
  budget, compare against a baseline, and inspect tuning and held-out results.
- **Investigate failures:** get evidence-linked diagnostic hypotheses and editable
  experiment suggestions for saved answers, with JSON/Markdown reports.
- **Measure performance:** saved indexing, model-loading, retrieval, generation, and
  scoring timings; sampled backend RAM; HTTP attempts, retries, and reported tokens.
- **Keep work recoverable:** immutable run snapshots, checkpointed answers and scores,
  cooperative cancellation, and explicit retry of missing work.

## Quick start

On Windows or Linux, you need **Python 3.11**,
[uv](https://docs.astral.sh/uv/getting-started/installation/), a Groq API key,
and disk space for downloaded local models. The lockfile selects
CPU-only PyTorch; a GPU is not required.

### 1. Install

```sh
git clone https://github.com/wauul/rag-bench.git
cd rag-bench
uv python install 3.11
uv sync --locked --extra backend --extra cpu --extra dashboard
```

### 2. Configure access

Copy [.env.example](.env.example) to `.env`:

```powershell
Copy-Item .env.example .env
```

On Linux, use `cp .env.example .env` instead. Edit these values:

```dotenv
GROQ_API_KEY=your-groq-api-key
API_TOKEN=your-random-api-secret
DASHBOARD_PASSWORD=your-different-random-dashboard-password
```

Use different random secrets of at least **32 characters** for the token and password.
To generate one, run the command below twice and copy each value into `.env`:

```sh
uv run --no-sync python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep `.env` private; it is ignored by Git. Local password access works without Neon
Auth. Managed GitHub/email sign-in is optional and needs [additional setup](docs/account-login.md).

### 3. Start both applications

Run each command in a separate terminal, from the repository root:

```sh
# Terminal 1: API — keep one worker
uv run --no-sync python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers
```

```sh
# Terminal 2: dashboard
uv run --no-sync python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Open [localhost:8501](http://localhost:8501) and sign in with `DASHBOARD_PASSWORD`.
The API's [health endpoint](http://127.0.0.1:8000/health) checks basic availability;
it does not prove a benchmark can complete. Interactive API docs are disabled.

For a first run, open **Build a benchmark** and select **Run sample benchmark**.
It uses the fictional [Harbor handbook](sample_data/harbor-handbook.txt) and
[10 reference questions](sample_data/questions.json). This makes real Groq calls;
first use also downloads models and can take several minutes.

## Benchmark workflow

1. **Upload documents:** PDF, UTF-8 TXT, or Markdown. A set accepts 1–10 files,
   up to 5 MiB per file and 150,000 extracted characters in total. PDFs are limited
   to 100 pages each; scanned documents need OCR before upload.
2. **Add references:** write questions in the dashboard or upload JSON/CSV with
   1–20 question/reference pairs. References are used for evaluation, not generation.
3. **Choose 2–4 configurations:** start with the MiniLM/BGE presets or adjust the
   settings below. Change one setting at a time when you want to isolate its effect.
4. **Run and inspect:** review question-level scores, answers, retrieved evidence,
   and **Results → Performance**. Export results or reopen them from **History**.

Example question file:

```json
{
  "questions": [
    {
      "question": "When is Harbor Community Lab open on Saturdays?",
      "reference": "The lab opens on Saturdays from 10:00 to 16:00."
    }
  ]
}
```

A top-level JSON array also works. CSV uses `question,reference` headers;
`expected_answer` is accepted as an alias for `reference`.

| Setting | Choices / limits |
| --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5` |
| Chunk size | 32–240 WordPiece tokens; overlap must be smaller than chunk size |
| Retrieval candidates (`candidate_k`) | 1–40; at least the final passage count |
| Final passages (`context_k`) | 1–8 passages supplied to generation and evaluation |
| Reranking | Optional local `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Engine | Existing pipeline (default) or LangChain + LangGraph |

Reranking selects final passages from the candidate pool. Use a larger `candidate_k`
than `context_k` to let it consider alternatives. Guided optimization requires a
LangChain + LangGraph baseline; choosing that engine alone does not imply better answers.

## Reading results

The four Ragas metrics answer different questions:

| Metric | What it measures |
| --- | --- |
| Faithfulness | Whether answer claims are supported by the supplied context |
| Answer relevancy | How well the answer addresses the question |
| Context precision | Whether useful retrieved passages are ranked early |
| Context recall | How much of the reference answer is covered by retrieved context |

The equal-weight average is a convenience for **fully scored** comparisons.
Missing or failed scores remain unknown, and incomplete runs cannot establish an
overall winner. Answer relevancy uses cosine similarity and can be negative.
LLM judging and reference quality affect results; a small sample does not establish
general performance or statistical significance.

Performance profiles accumulate across execution attempts. RAM is sampled **backend
process RSS**, not isolated model memory or GPU memory. Saved row times can include
judging and retries; check phase timings before interpreting them as serving latency.
Provider attempts and reported tokens are usage measurements, not a dollar-cost estimate.
Older runs may lack measurements.

Download profile JSON/CSV from **Performance**, or fetch an existing run without
making provider calls:

```sh
uv run --no-sync python -m scripts.profile_run RUN_ID
```

Use **Cancel run** to stop subsequent stages. **Retry missing work** reuses saved
answers and scores when their snapshot is compatible. Restarts never automatically
resume spending; incompatible source/model/prompt changes may require the original
environment. Recovery cannot guarantee exactly-once provider billing.

## Configuration and storage

The full settings template is [.env.example](.env.example).

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Backend credential for generation, judging, and investigations |
| `GROQ_MODEL` | Hosted model for generation/judging; must be available to your account |
| `API_TOKEN` | Operator API credential; use the same value in the dashboard's server settings |
| `DASHBOARD_PASSWORD` | Operator dashboard access; regular users should use their own account/key |
| `BACKEND_URL` | Dashboard API URL; defaults to `http://127.0.0.1:8000` |
| `DATA_DIR` | Local data/cache directory; defaults to `./data` |
| `DATABASE_URL` | Optional authoritative PostgreSQL store; use a direct TLS URL for Neon |
| `GROQ_REQUEST_INTERVAL` | Provider pacing in seconds; defaults to `4` |

Without `DATABASE_URL`, SQLite stores inputs, snapshots, results, and checkpoints
under `DATA_DIR`; Chroma stores local vector indexes. With PostgreSQL configured,
the database holds durable records/checkpoints and local indexes are rebuildable
caches. Ephemeral hosting needs an external durable database to preserve history.
Keep **one API worker** in either mode.

Access is required by default. Personal keys and managed accounts isolate saved work;
PostgreSQL also enforces row-level security. Operator credentials access the owner's
workspace and should not be shared with regular users. Uploaded content, questions,
references, and answers may be sent to Groq; use non-sensitive benchmark data.

Default admission caps allow 500 provider HTTP attempts/day and 5,000/month, including
retries. A separate conservative token allowance and a kill switch also apply.
Inactive content expires after 30 days by default, with retained-work dependencies
protected. See [guardrails](docs/guardrails.md) for policy changes, user keys, deletion,
alerts, and exact limits. External tracing is disabled by default; optional Langfuse
export allows metadata only. See [LLMOps and privacy](docs/llmops.md).

## Development and deployment

Install the full engineering environment and run the same quality checks as CI:

```sh
uv sync --locked --extra backend --extra cpu --extra compact --extra dashboard --extra models
uv run --no-sync python -m scripts.checks
```

Checks cover the lockfile, generated requirements, installed dependencies, formatting,
lint, scoped typing, tests, and coverage. Ordinary tests use controlled provider
fixtures and make no paid calls. Real-model/PostgreSQL checks need explicit setup;
see [Contributing](CONTRIBUTING.md) and [CI/CD](docs/cicd.md).

For Docker, configure `.env` as above, then run `docker compose up --build`.
Compose binds both services to localhost and keeps SQLite/model caches in named
volumes. `Dockerfile` uses CPU PyTorch; `Dockerfile.render` uses prepared ONNX models.
Hosted deployment uses Render, Streamlit, and Neon, with a protected release process.
See the deployment guide before changing hosted services; pushing source does not
prove a release is deployed or its live model workload succeeds.

## Documentation

| Guide | Covers |
| --- | --- |
| [Retrieval debugger](docs/retrieval-debugger.md) | Trace inspection, review, replay, and comparisons |
| [Guided optimization](docs/guided-optimization.md) | Search plans, budgets, selection, and held-out checks |
| [Failure Investigator](docs/failure-investigator.md) | Diagnostic evidence, limitations, and experiment drafts |
| [Execution engines](docs/langgraph-architecture.md) | LangChain/LangGraph architecture and checkpoint recovery |
| [Neon storage](docs/neon-storage.md) | PostgreSQL setup, SQLite migration, and durability |
| [Account sign-in](docs/account-login.md) | Managed GitHub/email login and account keys |
| [Workspace guardrails](docs/guardrails.md) | Ownership, access, quotas, retention, and deletion |
| [LLMOps and privacy](docs/llmops.md) | Tracing boundaries, prompts, evaluation, and promotion |
| [CI/CD](docs/cicd.md) · [Release policy](docs/release-policy.md) | Engineering gates and artifact publication |
| [Deployment](docs/deployment.md) · [Operations](docs/operations.md) | Hosting, rollback, backups, and monitoring |
| [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) | Development conventions and vulnerability reporting |
| [Design](DESIGN.md) · [Changelog](CHANGELOG.md) | Interface principles and release history |
