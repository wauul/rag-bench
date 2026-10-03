---
title: RAG Bench API
emoji: ◈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
---

# ◈ RAG Bench

Compare 2–4 retrieval configurations on one document set and one reference test set. Inspect real generated answers, retrieved passages and four Ragas metrics in a separate Streamlit dashboard.

**Status (September 15, 2026):** The full local sample passed with 20 answers and all 80 finite Ragas scores; 11 automated tests passed. The hosted dashboard and API are deployed, and real hosted scoring, charts and passage inspection were checked. A complete error-free hosted run remains unverified after Groq quota errors. Further testing was stopped at the owner's request. No fabricated benchmark scores are shipped.

- [Live Streamlit dashboard](https://wauul-rag-bench-dashboardapp-peuvxw.streamlit.app/)
- [FastAPI documentation](https://rag-bench-api.onrender.com/docs)
- [GitHub repository](https://github.com/wauul/rag-bench)

The previously successful local run is `6b74daea575e4a09842f4953c4b05f07`. Its real JSON and CSV exports were saved under the original local ignored `data/` directory. MiniLM's equal-weight mean was 0.9504 and BGE with reranking was 0.9435 on that small sample; this is not a general model-quality claim. Those results predate the larger reranking candidate pool below.

### Current development changes

Configurations now separate **retrieval candidates** (`candidate_k`, 1–40) from **final passages** (`context_k`, 1–8). Reranking selects the final passages from the larger pool. The dashboard includes paginated **History**, **Retry missing work**, **Cancel run**, and separate processed/fully-scored progress counts. Generation and individual metric results are checkpointed to SQLite. The Dev Container installs both applications and starts both servers. These changes have automated coverage; a new live Groq benchmark and hosted deployment remain unverified.

The dashboard now has a responsive card layout, numbered navigation and a three-step readiness checklist. MiniLM and BGE presets make setup quicker; selected configurations can be edited or removed before evaluation. Results separate the score overview from answer inspection, with readable score cards and expandable evidence passages. History supports configuration-name/run-ID search and status filters on the current page.

New runs also record a durable performance profile. **Results → Performance** compares indexing, generation and scoring time, sampled backend RAM, Groq HTTP attempts and reported tokens. Phase, metric and model-loading details can be downloaded as CSV/JSON. Profiling accumulates across retries without repeating measurements for saved answers or scores.

## Architecture

```text
Streamlit dashboard → FastAPI → SQLite (documents, test sets, configurations, results)
                          └→ WordPiece windows → local Sentence Transformers
                             → persistent Chroma collection per run/configuration
                             → optional local cross-encoder → Groq answer
                             → Ragas + Groq judge + fixed local judge embeddings
```

Generation and judging both use `qwen/qwen3.8-27b` through Groq's free quota, with reasoning disabled to conserve tokens. The originally requested `llama-3.1-8b-instant` was [retired for free/developer accounts on August 16, 2026](https://console.groq.com/docs/deprecations). Set `GROQ_MODEL` to select an available Qwen or GPT-OSS model; Qwen uses `reasoning_effort=none` and GPT-OSS uses `low`. The tool does not silently switch models within a run. Retrieval compares `sentence-transformers/all-MiniLM-L6-v2` with `BAAI/bge-small-en-v1.5`. Optional reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. All inference except generation/judging runs on the backend CPU. Streamlit only calls the API.

## Local setup (Python 3.11+)

Use the two `requirements.txt` files. CPU PyTorch avoids unnecessary CUDA dependencies:

```powershell
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r backend/requirements.txt -r dashboard/requirements.txt
Copy-Item .env.example .env
```

Put your free-tier Groq API key into `.env` as `GROQ_API_KEY=...`. Never commit it. On macOS/Linux use `.venv/bin/python` and `cp` instead. `python -m venv` and `pip` are also supported.

Start two terminals from the repository root:

```powershell
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

```powershell
.venv/Scripts/python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Open [the dashboard](http://localhost:8501) and click **Run sample benchmark**. It uploads the fictional Harbor Community Lab handbook, stores ten reference questions, creates two configurations, and immediately starts a real evaluation. First use downloads all three open models to the Hugging Face cache (`~/.cache/huggingface`); allow several hundred MB of downloads and several GB for installed dependencies. CPU execution and Groq quota pacing can make a full run take many minutes.

Chroma persists vectors under `data/chroma`; metadata and result checkpoints persist in `data/bench.sqlite3`. Collection names include both run and configuration UUIDs. Model downloads are reused. Keep **one backend process/worker**; the executor serializes runs. A restart marks interrupted runs failed while preserving saved passages, answers and individual metric scores. Use **Retry missing work** to resume them.

## Add your data

1. **Setup:** upload 1–10 PDFs, TXT or Markdown files (UTF-8). Limits: 5 MB/file, 100 pages/PDF and 150,000 extracted characters/set. Scanned PDFs need OCR outside this tool.
2. Upload a CSV with `question,reference` columns (`expected_answer` is an alias), upload a JSON array of those objects, or enter pairs in the editable table. Accepts 1–20 questions. `sample_data/questions.json` is a complete example.
3. Add 2–4 configurations using the presets or the custom builder: chunk tokens, overlap, model, final passage count, candidate count and reranking. Edit or remove selected configurations as needed. Change one parameter at a time for controlled experiments.
4. Click **Continue to evaluation** after the checklist is complete, then **Run evaluation**. Progress polls automatically and distinguishes processed answers from fully scored answers. Reopen saved runs through **History**, or use a run ID. Cancel or retry missing work from **Evaluation** or **Results**.
5. Open **Results** for metric means, valid counts and bars/radar in **Score overview**. Use **Inspect answers** to compare reference answers, generated answers and supporting passages. CSV exports contain one row per configuration/question, all scores, contexts, errors and latency.

The sample compares multiple changes at once to demonstrate the interface; its winner cannot establish which individual setting caused an improvement.

Both demo configurations send two final passages per question to conserve free judge quota. MiniLM retrieves two candidates; BGE reranks twelve candidates and keeps two. Custom configurations support 1–40 candidates and 1–8 final passages, with candidates at least as large as the final count. Without reranking, the dashboard uses the final count as the candidate count.

## What the metrics mean

| Metric | Ragas implementation | Meaning |
|---|---|---|
| Faithfulness | `Faithfulness` | Judge extracts answer claims and checks support in retrieved text; score is the supported fraction. |
| Answer relevancy | `ResponseRelevancy` | Judge generates three questions from the answer; fixed MiniLM embeddings measure their cosine similarity to the original question, with a penalty for noncommittal answers. |
| Context precision | `LLMContextPrecisionWithReference` | Judge decides which retrieved chunks help answer the reference; average precision rewards relevant chunks near the top. |
| Context recall | `LLMContextRecall` | Judge extracts reference claims and checks how many the retrieved context supports. |

Higher is generally better. Relevancy is based on cosine similarity and is not a calibrated probability (it can theoretically be negative); the other metrics are fractions in [0,1]. The bar chart includes a small negative margin; use tables/CSV for the exact scores. An equal-weight arithmetic mean is shown as a convenience only when **every question and metric succeeds for every configuration**. Ties are named. No statistical significance claim is made.

We pin Ragas 0.3.9 and use its real single-turn metric API. To conserve free quota, prompts retain Ragas's instructions and JSON schemas but omit few-shot examples by default (`JUDGE_EXAMPLES=0`). Set this to 1–3 to include up to that many built-in examples per prompt. Prompt settings are fixed for a run and recorded in provenance; changing prompts can affect scores. The Groq adapter obtains relevancy's three completions through separate calls, handling Groq's `n=1` restriction and nested reasoning-token metadata. We keep relevancy embeddings fixed across configurations so changing the retriever does not also change the measurement. NaN, API failures and judge parsing errors appear as **missing**, never as fabricated zeros. Partial summaries include valid sample counts and cannot declare an overall winner.

## Chunking and reranking

Sliding windows use one shared MiniLM WordPiece tokenizer, with configurable overlap. Source substrings retain original case and page metadata. Windows stop at document/page boundaries. Sizes are limited to 32–240 tokens to fit MiniLM's 256-token window; overlap must be smaller than the size. Long questions may be truncated by embedding models, so keep questions concise.

BGE queries receive its retrieval instruction prefix. Both models output normalized vectors; Chroma uses cosine distance. Retrieve `candidate_k` chunks, optionally score/order them with a cross-encoder, then keep `context_k` passages for generation and evaluation. A larger candidate pool lets reranking select evidence that would have been omitted by a smaller vector-only top-k. It still cannot recover chunks outside that candidate pool. If a document produces fewer chunks than requested, use the available chunks.

Legacy API input and stored configurations using `top_k` remain supported: it becomes both the candidate count and the final count, preserving their previous behavior. New configuration responses and run snapshots use `candidate_k` and `context_k`. Supplying conflicting `top_k` and `context_k` is rejected.

## API

Interactive docs: [localhost:8000/docs](http://localhost:8000/docs).

| Endpoint | Input / output |
|---|---|
| `POST /api/documents` | Multipart `files` → document set ID |
| `POST /api/test-sets` | JSON `{name, questions: [{question, reference}]}` → test set |
| `POST /api/test-sets/upload` | Multipart `file` (CSV/JSON) → test set |
| `POST /api/configurations` | JSON configuration → stored configuration |
| `POST /api/demo` | Creates the sample data and two configurations |
| `POST /api/runs` | `{document_set_id, test_set_id, configuration_ids}` → run ID (202) |
| `GET /api/runs` | Paginated compact history; `limit` (1–100), `offset`; returns `{runs, next_offset}` |
| `POST /api/runs/{id}/retry` | Resume a partial, failed or cancelled run using saved work (202) |
| `POST /api/runs/{id}/cancel` | Request cancellation between operations, preserving results (202) |
| `GET /api/runs/{id}` | Status, progress, configuration snapshots, provenance, summary, per-question results |
| `GET /api/runs/{id}/export` | CSV, including partial rows if available |
| `GET /api/runs/{id}/profile` | Saved performance profile and summary; `profiling: null` for runs without measurements |
| `GET /api/runs/{id}/profile/export` | One CSV row per measured phase/attempt; returns 409 when no phases were recorded |
| `GET /health` | Lightweight readiness and key-presence check; does not call Groq |

Run states: `queued`, `running`, `completed`, `partial`, `failed`, `cancelled`. A busy worker or incompatible retry returns 409; a missing Groq key returns 503; invalid input returns 422. A cancellation request does not interrupt an in-flight model or judge call: its result is saved before the worker stops. Runs expose `retrieved`, `completed` (processed answers), `scored` (fully successful answers), `valid_scores`, and a live `cancel_requested` flag. The API never sends its Groq key to Streamlit. Set `API_TOKEN` to require a bearer token on all `/api` endpoints when publicly hosting. Give the dashboard the same token through secrets.

Startup validates `INFERENCE_BACKEND` (`sentence-transformers` or `onnx`), a nonempty `GROQ_MODEL`, finite `GROQ_REQUEST_INTERVAL` of at least 0.1 seconds, and integer `JUDGE_EXAMPLES` in 0–3. Invalid settings name the offending variable before a worker starts.

## Performance profiling

Profiling is automatic for new evaluations. It records separate elapsed times for chunking, retriever loading, document embedding/vector-index writes, query retrieval, reranker loading/ranking, judge embedding loading/setup, answer generation and each Ragas metric attempt. Configuration/question/metric identifiers associate measurements with their work. Generation and metric timings include quota pacing, retries and checkpoint overhead; `http_seconds` isolates time inside the HTTP client. Worker time also includes setup, cleanup and work between phases. Metric tables show attempt counts, total, mean, median and 95th percentile latency, including failures and retries.

Memory is the backend process's resident set size (RSS), measured in **MiB** every 100 ms and at phase boundaries. Each phase records RSS before/after, the change and sampled peak. Model-loading rows name the model and show its process-memory impact. These values include libraries, indexes, Python and allocator effects; they do not isolate model weights. Sampling can miss shorter peaks. Attempt metadata records OS, Python and CPU count so comparisons can account for host differences. First-use downloads can affect model-loading time; compare runs on the same hardware with the same cache conditions.

Groq accounting wraps both synchronous and asynchronous HTTP clients, counting every SDK HTTP attempt, 429 response, other HTTP failure and transport error. Usage includes the readiness check, generation and individual judge completions. Prompt, completion, total, reasoning and cached-token counters use provider-reported values; reasoning/cached tokens are subsets, not additions to the total. A missing redundant total is computed only when both prompt and completion counts are reported. Successful responses without complete usage and requests without a saved outcome are disclosed separately. Unknown usage is never treated as a free request or a fabricated zero. This reports the run's consumption, not account-wide remaining quota or billing cost. No prompts, answer text, credentials or HTTP headers are copied into the profile.

Profiles live in the run's SQLite record and checkpoint phase boundaries and HTTP attempts/outcomes. Retries append an attempt, retaining earlier measurements and costs. A process restart marks open phases/attempts **interrupted** and retains their last saved values as lower bounds; requests that never returned and peaks after the last checkpoint cannot be reconstructed. Historical runs without profiling show it as unavailable. Retrying an older run measures only new work and marks coverage as partial.

The existing results CSV adds per-answer generation/scoring/per-metric seconds and known Groq requests/tokens. The separate performance CSV retains indexing, memory and all phase attempts; unreported token fields are blank. Download a saved profile without making new Groq calls:

```powershell
.venv/Scripts/python.exe -m scripts.profile_run RUN_ID
```

Set `BACKEND_URL` and `API_TOKEN` for the target backend. Reports are saved under ignored `data/profile-RUN_ID.json` and `.csv`.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m scripts.check_retrieval
# Requires a configured key and a running backend:
.venv/Scripts/python.exe -m scripts.run_demo
```

The current automated suite has 44 tests covering ingestion validation, malformed/blank/encrypted files, auth, missing keys, settings, non-finite metric errors, partial averages, checkpointed retry/cancellation, restart recovery, history pagination, dashboard actions and CSV formula escaping. Dashboard coverage includes preset creation, configuration editing/removal, guided navigation, validation, history filters and performance rendering. Profiling tests use real Groq SDK calls against controlled HTTP transports to check retry counts, async judge completions, nested token metadata, missing/zero usage and row attribution; memory/clock fixtures test sampled peaks and cumulative attempts. A real persistent Chroma test uses deterministic tiny vectors to verify candidate selection and recorded retrieval/indexing/model-loading phases without downloading models. The separate retrieval check downloads both real embedding models, exercises ten questions per model, checks token windows, and runs a real cross-encoder. These checks do **not** replace the full Groq/Ragas demo. `run_demo` requires 20 complete rows, nonzero mean scores and a working CSV endpoint, and saves results locally under ignored `data/`.

The redesigned dashboard was also checked in a local headless browser at desktop and 390px mobile widths. Uploads, presets, editing, navigation, history, charts and passage inspection were exercised without calling Groq. Clearly labeled UI fixtures were used only in an ignored, isolated preview store to check results rendering; they are not shipped as benchmark scores.

To verify a demo already started through the dashboard, set `BACKEND_URL` to that backend and run `python -m scripts.run_demo RUN_ID`. It checks all 80 finite scores and saves JSON plus CSV.

After Groq quota recovers, use **Retry missing work** or `POST /api/runs/{id}/retry`. This keeps successful answers/scores, uses the original stored passages, fills missing retrieval rows, and retains previous row attempts in `retry_history`. It accepts partial, failed and cancelled runs, including restart interruptions, while rejecting changed model/backend/judge settings. Complete runs cannot be retried. A retry needs the original document set only when retrieval rows are missing. It cannot restore data lost by an ephemeral host.

For offline **local** recovery, stop the local API and run `python -m scripts.retry_failed RUN_ID`, then restart the API. The command uses the same execution/checkpoint code as API retry. Both paths preserve existing configuration snapshots; previously saved `top_k` runs keep their smaller candidate pools.

### Dev Container / Codespaces

Opening the Dev Container installs CPU Torch plus the backend and dashboard requirements. On container startup it launches one FastAPI worker on port 8000 and Streamlit on port 8501; both ports are forwarded. Logs are under ignored `.tools/backend.log` and `.tools/dashboard.log`. Configure `GROQ_API_KEY` in `.env` or the container environment before starting a benchmark. The dashboard and ingestion/history can open without a Groq key. After changing backend settings, restart the backend/container.

## Free hosting and current deployment constraint

**Render free and Railway free currently offer only 512 MB / 0.5 GB RAM.** The standard Sentence Transformers check measured approximately **749–824 MB RSS on Windows**, so `Dockerfile.render` provides a compact runtime using int8 ONNX exports of the same three models. It runs on CPU without importing PyTorch. `Dockerfile` retains the full Sentence Transformers runtime for local/larger hosts. `render.yaml` selects the compact image and the Free plan.

The compact runtime measured approximately **320 MB RSS** with MiniLM, Chroma and all Ragas metrics loaded in the local check; deployed peak memory still needs verification. Original model tokenizers and pooling are preserved. Build-time ONNX weights come from each model's official Hugging Face repository; BGE is dynamically quantized during the build. Agreement checks across the ten sample questions and reference answers gave minimum cosine agreement **0.991 for MiniLM** and **0.986 for BGE** against the float32 Sentence Transformers implementation. The reranker's best passage was unchanged in the checked question. This is approximate inference, not bit-identical output; run provenance records `inference_backend` and `precision`.

To reproduce the compact model checks locally:

```powershell
uv pip install --python .venv/Scripts/python.exe onnx==1.19.0
.venv/Scripts/python.exe -m scripts.prepare_onnx
.venv/Scripts/python.exe -m scripts.check_onnx
# Select compact inference before starting FastAPI:
$env:INFERENCE_BACKEND = "onnx"
```

Render free also has an ephemeral filesystem: SQLite data, Chroma collections and downloaded models disappear on restart/redeploy. It sleeps after inactivity. Railway's free usage allowance is limited. References: [Render free limits](https://render.com/docs/free), [Render compute plans](https://render.com/docs/compute-plans), [Railway pricing plans](https://docs.railway.com/pricing/plans).

Hugging Face was considered as an alternative because its documentation lists free CPU Basic hardware. However, the actual new-account Space creation UI on September 15, 2026 restricted Docker and Gradio to a **paid** plan. No subscription was created. Render's compact runtime is the free deployment path being verified. The dashboard remains on Streamlit Community Cloud.

### Hugging Face Spaces (only for accounts already eligible for free Docker compute)

1. Sign in to Hugging Face and create a public **Docker / Blank** Space named `rag-bench-api`, choosing **CPU Basic / Free**. If a free option is unavailable or payment is requested, stop.
2. Upload this repository's `Dockerfile`, `README.md`, `backend/` and `sample_data/` to the Space. The README metadata sets Docker's app port to 8000. Alternatively authenticate with `hf auth login` using a write-scoped token and run `python -m scripts.deploy_hf YOUR_USERNAME/rag-bench-api` to create/upload the Space.
3. Space Settings → Variables and secrets → add `GROQ_API_KEY` and a randomly generated `API_TOKEN` as **secrets**. No credentials belong in the repository.
4. Wait for the Docker build; open the Space's direct `.hf.space` URL and append `/health` to check readiness. Use this direct host as Streamlit's `BACKEND_URL`.
5. Set the same API token in Streamlit secrets. Verify the complete demo. Free Spaces may sleep and lose stored data on restart; export results.

### Render Free (compact runtime)

1. Connect GitHub at [Render](https://dashboard.render.com/).
2. New → Blueprint → select this repository. Review `render.yaml`: the plan must remain **Free**. Alternatively use New Web Service → Public Git Repository → `https://github.com/wauul/rag-bench`, choose Docker, set Dockerfile path to `Dockerfile.render`, and select Free.
3. Add `GROQ_API_KEY` and `API_TOKEN` as environment secrets. Blueprints generate the API token automatically; manual setup needs a random token. Keep both secret.
4. Deploy, check `/health`, then run the actual demo and check for memory failures. A green health check alone does not prove the ML workload works.
5. Copy the backend URL and API token into Streamlit secrets.

### Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) and connect GitHub if needed.
2. Create app → deploy an existing GitHub app. Repository: `wauul/rag-bench`; branch: `main`; main file: `dashboard/app.py`.
3. Advanced settings → Python **3.11** → secrets:

```toml
BACKEND_URL = "https://YOUR-BACKEND-HOST"
API_TOKEN = "SAME-TOKEN-AS-BACKEND"
```

4. Deploy. Streamlit installs `dashboard/requirements.txt`, keeping ML packages off the frontend.
5. Click **Run sample benchmark**, wait for all 20 answers and 80 finite metric scores, inspect passages, and download CSV. Deployment is only end-to-end verified after this succeeds.

## Limitations and operational notes

- Small English embedding models and compact judge prompts prioritize free CPU/quota use. LLM judges can be inconsistent or wrong; same-model judging can introduce correlated bias. Reference quality matters. No confidence intervals or human validation are implied.
- Groq generation and judging both consume free API quota; this tool makes multiple requests per answer. Default pacing is one request every four seconds, sequential scoring, with bounded SDK retries. Daily quotas may still run out. Adjust `GROQ_REQUEST_INTERVAL` conservatively.
- Local sequential model loading limits simultaneous weight memory, but Python/PyTorch may retain allocations. Provision adequate RAM.
- Intended as a small shared internal tool, not a multi-tenant service. No per-user data isolation, distributed queue or automatic retention policy. Cancellation is cooperative and resumption is explicitly requested. Run IDs and history are shared workspace identifiers.
- Public dashboard visitors share the backend's free quota. Use Streamlit access restrictions for private use. Uploaded text, questions, references and generated answers are sent to Groq for generation/evaluation; use appropriate non-sensitive benchmark data.
- Stored data persists locally until removed; long-lived instances need manual retention/cleanup when idle. Free ephemeral hosting is unsuitable for durable records. Export results promptly.
- PDF text extraction does not provide OCR or sophisticated table reconstruction. Chunking is token-window-based, not semantic segmentation.
