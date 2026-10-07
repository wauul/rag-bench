# Integration verification — 7 October 2026

This record distinguishes deterministic tests, real local model execution, rendered
browser checks and external-provider/deployment evidence. Local implementation
keeps `existing` as the default engine.

## Automated checks

| Check | Command / environment | Result |
|---|---|---|
| Original baseline | `.venv/Scripts/python.exe -m pytest -q` before edits | 44 passed |
| New clean environment | `uv venv .tools/verify-venv --python .venv/Scripts/python.exe`, then `uv pip install --python .tools/verify-venv/Scripts/python.exe -r backend/requirements.txt -r dashboard/requirements.txt` | Installed successfully, Python 3.11.16 |
| Dependency consistency | `uv pip check --python .tools/verify-venv/Scripts/python.exe` and the development environment | No conflicts |
| Compact clean environment | Same installation/check using `backend/requirements-compact.txt` | No conflicts; Ragas, graph, Groq adapter and ONNX modules import without Torch |
| Linux resolution | `uv pip compile backend/requirements-compact.txt --python-platform x86_64-manylinux_2_28 --python-version 3.11` | Resolves successfully |
| Full updated suite | `.tools/verify-venv/Scripts/python.exe -m pytest -q` | **81 passed, 2 intentionally skipped**, 5 upstream warnings |
| Deployment image | `docker build -f Dockerfile.render -t ragbench-langgraph:verify .` | Build, pinned ONNX preparation, imports and `pip check` passed |
| Container recreation and persistent volume | `.tools/container-restart.ps1` using two separate disposable containers | Passed: no calls on startup; explicit retry preserved the saved answer and first metric |
| Real compact models + graph suite | Linux image, `RAGBENCH_TEST_REAL_MODELS=1`, `python -m pytest -q tests/test_real_retrieval.py tests/test_graph.py` | **38 passed**, including both real-model cases and fresh-process recovery |
| Formatting | `git diff --check` | Passed |

Linux tests mount `tests/` and `scripts/` read-only at `/app/tests` and `/app/scripts`;
the backend under test comes from the image, not a source bind mount. The two
real-model tests intentionally skip in the ordinary suite unless explicitly enabled.
The container runs them using the image's pinned MiniLM/BGE models and cross-encoder.

`test_graph.py` uses real StateGraph, SQLite checkpoints and Chroma with deterministic
small vectors for ordinary tests. Process-death cases call `os._exit(73)`, then
recover from another interpreter. They cover initial checkpoint persistence,
indexing, saved retrieval, before generation, response-before-save, saved answer
before graph advance, and between metrics. The response-before-save case correctly
records a repeated fixture generation; saved answers/metrics are not repeated.

Other coverage includes legacy defaults, mixed engine dispatch, metadata/order/prompt
parity, reference exclusion from generation, cancellation/retry, partial scores,
checkpoint/application disagreement, stale inputs/graph versions, OS lock contention
from another process, empty answers, invalid scores, cleanup and API exports.
Tests also reconcile stale error bookkeeping without recomputing saved valid values.
Scores orphaned from a missing/invalid answer are invalidated before regenerating it.

The controlled-HTTP integration uses actual ChatGroq, GroqRagasLLM and all four Ragas
metrics, including their prompts, parsers and score algorithms. Two one-passage
answers (one per engine) completed in **16 HTTP requests**. A separate test verifies
that 429, 500 and timeout failures stop after **two SDK HTTP attempts**. These are
deterministic fixtures, not real Groq responses or measured answer quality.

## Browser workflow

An isolated local backend at port 8016 used test-only embedding/provider/metric
fixtures and `.tools/browser-data`; Streamlit ran at port 8516. Real StateGraph,
Chroma, SQLite, API scheduling and dashboard actions remained in the path.

The Codex browser exercised configuration editing and engine selection, launch,
persisted progress, cancellation, retry, answer inspection, history and a full CSV
download. The completed mixed-engine fixture had 20 answers and 80 valid fixture
scores; a full refresh could reopen it through history. Both engine labels rendered
in cards and history. The downloaded CSV was read back and verified to contain 20
rows, both engine values and version provenance. Browser console error list: empty.

Answers explicitly stated “UI fixture — generated text, not a real Groq response.”
The equal fixture scores carry no quality/comparison conclusion. Screenshot evidence
is saved locally at `.tools/langgraph-browser-answers.png`; the downloaded test CSV
is `rag-bench-6c8e0f5409c7456c9bf7dbcfee5e0947.csv` in the local Downloads directory.
This check was desktop browser verification, not a new mobile/browser-matrix audit.

## Evidence boundaries and deployment

No `GROQ_API_KEY` was available in the local environment or normal dotenv loading.
**No live Groq request was made.** A real provider smoke test remains unverified;
the controlled-HTTP budget above is not a claim about production quota or quality.

Real local-model verification compares retrieved passages, metadata, ordering and
prompt text for both embedding models with reranking. No answer-quality improvement,
provider determinism, production performance advantage or hosted RAM guarantee is
inferred. The graph can add repeated model loading and checkpoint overhead; warm
index/retry reuse must be separated from cold work in comparisons.

The current Render Free blueprint is unchanged in service tier and explicitly marked
ephemeral. Its filesystem durability was checked against [Render's documentation](https://render.com/docs/disks),
not by changing the live service. The paid disk file is only a reviewable proposal;
the local Compose volume uses existing hardware. No deployment, push, paid service
or infrastructure purchase was performed. Hosted restart/redeploy survival remains
unverified. Local Docker image success is reported separately from hosting success.

Known warnings are upstream Starlette/AnyIO and Ragas embedding-wrapper deprecations,
plus LangGraph's default-serializer deprecation emitted by package imports. Ragbench's
checkpoint serializer is explicitly JSON-only. The full Sentence Transformers
Dockerfile has not been rebuilt in this verification; its Windows dependencies were
installed/tested, while the actual Render compact image was built and exercised.

## Final outcomes

The final clean Windows suite passed **81 tests**, with the two real-model tests
intentionally skipped there. The final Linux image passed **38 graph/local-model tests**.
A separate Docker test killed a container after the
first metric commit and started a new container against the same named volume.
API startup marked the interrupted run failed without model calls. Explicit retry
completed it with exactly one recorded fixture generation and one call per metric
across both containers. The temporary volume was removed after verification.

The final Docker build hit a local unpacking-cache error after successful build
steps. It was recovered without pruning shared Docker state using:

```powershell
docker buildx build -f Dockerfile.render -t ragbench-langgraph:verify --output type=docker,dest=.tools/ragbench-verify.tar .
docker load --input .tools/ragbench-verify.tar
```

Both commands succeeded. Final image manifest:
`sha256:bc9fbdff670fb9d920ac2de4e8e766716dc344acb9f1366f2beb10303f177d80`.
The container-recreation check used this image.

Git changes are local and uncommitted on `main`; the starting HEAD was `e663d41`
and matched the locally known `origin/main`. No commit or push was performed.

Concurrent workspace edits appeared near the end of verification, including
`pyproject.toml`, `uv.lock`, `.python-version` and operational validation in
`backend/settings.py`. They were preserved, not authored or incorporated into the
integration image by this task. The test counts and image digest above describe
the tested integration snapshot, not a certification of subsequent concurrent edits.
