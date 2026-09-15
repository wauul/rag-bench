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

**Status:** Implementation and local retrieval verification are available. A successful full Groq/Ragas evaluation and live backend verification still require a Groq key and suitable hosting. No precomputed or fabricated scores are shipped.

## Architecture

```text
Streamlit dashboard → FastAPI → SQLite (documents, test sets, configurations, results)
                          └→ WordPiece windows → local Sentence Transformers
                             → persistent Chroma collection per run/configuration
                             → optional local cross-encoder → Groq answer
                             → Ragas + Groq judge + fixed local judge embeddings
```

Generation and judging both use `llama-3.1-8b-instant` through Groq. Retrieval compares `sentence-transformers/all-MiniLM-L6-v2` with `BAAI/bge-small-en-v1.5`. Optional reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. All inference except generation/judging runs on the backend CPU. Streamlit only calls the API.

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

Open [the dashboard](http://localhost:8501) and click **Try it now**. It uploads the fictional Harbor Community Lab handbook, stores ten reference questions, creates two configurations, and immediately starts a real evaluation. First use downloads all three open models to the Hugging Face cache (`~/.cache/huggingface`); allow several hundred MB of downloads and several GB for installed dependencies. CPU execution and Groq quota pacing can make a full run take many minutes.

Chroma persists vectors under `data/chroma`; metadata and complete result rows persist in `data/bench.sqlite3`. Collection names include both run and configuration UUIDs. Model downloads are reused. Keep **one backend process/worker**; the executor serializes runs. A restart marks interrupted runs failed while preserving completed rows.

## Add your data

1. **Upload / Setup:** upload 1–10 PDFs, TXT or Markdown files (UTF-8). Limits: 5 MB/file, 100 pages/PDF and 150,000 extracted characters/set. Scanned PDFs need OCR outside this tool.
2. Upload a CSV with `question,reference` columns (`expected_answer` is an alias), upload a JSON array of those objects, or enter pairs in the editable table. Accepts 1–20 questions. `sample_data/questions.json` is a complete example.
3. Add 2–4 configurations: chunk tokens, overlap, model, top-k and reranking. Change one parameter at a time for controlled experiments.
4. Open **Run**, start evaluation, and retain the run ID. Progress polls automatically. Use the run ID to reopen results after a browser session ends.
5. Open **Results** for metric means, valid counts, bars/radar, reference answers and passages side by side. CSV exports contain one row per configuration/question, all scores, contexts, errors and latency.

The sample compares multiple changes at once to demonstrate the interface; its winner cannot establish which individual setting caused an improvement.

## What the metrics mean

| Metric | Ragas implementation | Meaning |
|---|---|---|
| Faithfulness | `Faithfulness` | Judge extracts answer claims and checks support in retrieved text; score is the supported fraction. |
| Answer relevancy | `ResponseRelevancy` | Judge generates three questions from the answer; fixed MiniLM embeddings measure their cosine similarity to the original question, with a penalty for noncommittal answers. |
| Context precision | `LLMContextPrecisionWithReference` | Judge decides which retrieved chunks help answer the reference; average precision rewards relevant chunks near the top. |
| Context recall | `LLMContextRecall` | Judge extracts reference claims and checks how many the retrieved context supports. |

Higher is generally better. Relevancy is based on cosine similarity and is not a calibrated probability (it can theoretically be negative); the other metrics are fractions in [0,1]. The bar chart includes a small negative margin; use tables/CSV for the exact scores. An equal-weight arithmetic mean is shown as a convenience only when **every question and metric succeeds for every configuration**. Ties are named. No statistical significance claim is made.

We pin Ragas 0.3.9 and use its real single-turn metric API. We keep relevancy embeddings fixed across configurations so changing the retriever does not also change the measurement. NaN, API failures and judge parsing errors appear as **missing**, never as fabricated zeros. Partial summaries include valid sample counts and cannot declare an overall winner.

## Chunking and reranking

Sliding windows use one shared MiniLM WordPiece tokenizer, with configurable overlap. Source substrings retain original case and page metadata. Windows stop at document/page boundaries. Sizes are limited to 32–240 tokens to fit MiniLM's 256-token window; overlap must be smaller than the size. Long questions may be truncated by embedding models, so keep questions concise.

BGE queries receive its retrieval instruction prefix. Both models output normalized vectors; Chroma uses cosine distance. Reranking scores the **same top-k candidates**, then reorders them. A cross-encoder sees query and passage together, which can improve ranking precision over comparing independent embeddings. It cannot recover chunks outside this candidate pool; it does not change pool recall directly.

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
| `GET /api/runs/{id}` | Status, progress, configuration snapshots, provenance, summary, per-question results |
| `GET /api/runs/{id}/export` | CSV, including partial rows if available |
| `GET /health` | Lightweight readiness and key-presence check; does not call Groq |

Run states: `queued`, `running`, `completed`, `partial`, `failed`. A busy worker returns 409; a missing Groq key returns 503; invalid input returns 422. The API never sends its Groq key to Streamlit. Set `API_TOKEN` to require a bearer token on all `/api` endpoints when publicly hosting. Give the dashboard the same token through secrets.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m scripts.check_retrieval
# Requires a configured key and a running backend:
.venv/Scripts/python.exe -m scripts.run_demo
```

Unit/API tests cover ingestion validation, malformed/blank/encrypted files, auth, missing keys, non-finite metric errors, partial averages, restart behavior and CSV formula escaping. The retrieval check downloads both real embedding models, exercises ten questions per model in persistent Chroma, checks token windows, and runs a real cross-encoder. These checks do **not** replace the full Groq/Ragas demo. `run_demo` requires 20 complete rows, nonzero mean scores and a working CSV endpoint, and saves results locally under ignored `data/`.

## Free hosting and current deployment constraint

`Dockerfile` and `render.yaml` are included for a single FastAPI service on Render. **Render free and Railway free currently offer only 512 MB / 0.5 GB RAM.** The real local retrieval check measured approximately **749–824 MB RSS on Windows**, before a full evaluation. Linux memory may differ, but these free plans are not a verified fit. Do not upgrade to a paid instance unless you intend to pay.

Render free also has an ephemeral filesystem: SQLite data, Chroma collections and downloaded models disappear on restart/redeploy. It sleeps after inactivity. Railway's free usage allowance is limited. References: [Render free limits](https://render.com/docs/free), [Render compute plans](https://render.com/docs/compute-plans), [Railway pricing plans](https://docs.railway.com/pricing/plans).

The approved backend target is a free Hugging Face Docker Space, subject to account availability; its [documented CPU Basic hardware](https://huggingface.co/docs/hub/spaces-overview) has 16 GB RAM. It still has ephemeral disk without paid storage. The dashboard remains on Streamlit Community Cloud.

### Hugging Face Spaces (approved free backend target)

1. Sign in to Hugging Face and create a public **Docker / Blank** Space named `rag-bench-api`, choosing **CPU Basic / Free**. If a free option is unavailable or payment is requested, stop.
2. Upload this repository's `Dockerfile`, `README.md`, `backend/` and `sample_data/` to the Space. The README metadata sets Docker's app port to 8000. Alternatively authenticate with `hf auth login` using a write-scoped token and run `python -m scripts.deploy_hf YOUR_USERNAME/rag-bench-api` to create/upload the Space.
3. Space Settings → Variables and secrets → add `GROQ_API_KEY` and a randomly generated `API_TOKEN` as **secrets**. No credentials belong in the repository.
4. Wait for the Docker build; open the Space's direct `.hf.space` URL and append `/health` to check readiness. Use this direct host as Streamlit's `BACKEND_URL`.
5. Set the same API token in Streamlit secrets. Verify the complete demo. Free Spaces may sleep and lose stored data on restart; export results.

### Render (only if available resources can support the ML process)

1. Connect GitHub at [Render](https://dashboard.render.com/).
2. New → Blueprint → select this repository. Review `render.yaml`: the plan must remain **Free**.
3. Add `GROQ_API_KEY`; Render generates `API_TOKEN`. Keep both secret.
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
5. Click **Try it now**, wait for all 20 answers and 80 finite metric scores, inspect passages, and download CSV. Deployment is only end-to-end verified after this succeeds.

## Limitations and operational notes

- Small English embedding models and a small generation/judge model prioritize free CPU/quota use. LLM judges can be inconsistent or wrong; same-model judging can introduce correlated bias. Reference quality matters. No confidence intervals or human validation are implied.
- Groq generation and judging both consume free API quota; this tool makes multiple requests per answer. Default pacing is one request every four seconds, sequential scoring, with bounded SDK retries. Daily quotas may still run out. Adjust `GROQ_REQUEST_INTERVAL` conservatively.
- Local sequential model loading limits simultaneous weight memory, but Python/PyTorch may retain allocations. Provision adequate RAM.
- Intended as a small shared internal tool, not a multi-tenant service. No per-user data isolation, cancellation, distributed queue, automatic retention policy or resume after backend restart. Run IDs are shared workspace identifiers.
- Public dashboard visitors share the backend's free quota. Use Streamlit access restrictions for private use. Uploaded text, questions, references and generated answers are sent to Groq for generation/evaluation; use appropriate non-sensitive benchmark data.
- Stored data persists locally until removed; long-lived instances need manual retention/cleanup when idle. Free ephemeral hosting is unsuitable for durable records. Export results promptly.
- PDF text extraction does not provide OCR or sophisticated table reconstruction. Chunking is token-window-based, not semantic segmentation.
