import hmac
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from backend.ingestion import MAX_BYTES, MAX_TEXT, extract_document, parse_test_set
from backend.models import Configuration, RunRequest, TestSet
from backend.storage import Store

store = Store()
executor = ThreadPoolExecutor(max_workers=1)
run_lock = threading.Lock()
sample_dir = Path(__file__).resolve().parents[1] / "sample_data"


@asynccontextmanager
async def lifespan(app):
    # A killed process cannot resume its in-memory worker. Keep partial rows, mark honestly.
    for run in store.list("run"):
        if run["status"] in {"queued", "running"}:
            run.update(status="failed", stage="Server restarted; start a new run", error="Interrupted by server restart")
            store.save("run", run, run["id"])
    yield


def authorize(authorization: str | None = Header(default=None)):
    token = os.getenv("API_TOKEN", "")
    if token and not hmac.compare_digest(authorization or "", "Bearer " + token):
        raise HTTPException(401, "Invalid API token")


app = FastAPI(title="RAG Bench", version="1.0.0", lifespan=lifespan)
auth = [Depends(authorize)]


def fetch(kind, object_id):
    try:
        return store.get(kind, object_id)
    except KeyError:
        raise HTTPException(404, f"Unknown {kind} ID")


@app.get("/health")
def health():
    return {"status": "ok", "generation_ready": bool(os.getenv("GROQ_API_KEY")),
            "authentication_required": bool(os.getenv("API_TOKEN")),
            "revision": os.getenv("RENDER_GIT_COMMIT", "local")}


@app.post("/api/documents", dependencies=auth, status_code=201)
async def documents(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 10:
        raise HTTPException(422, "Upload between 1 and 10 documents")
    docs = []
    try:
        for file in files:
            docs.extend(extract_document(file.filename or "document.txt", await file.read(MAX_BYTES + 1)))
        if sum(len(d["text"]) for d in docs) > MAX_TEXT:
            raise ValueError("Document set exceeds 150,000 characters")
    except Exception as exc:
        raise HTTPException(422, str(exc) if isinstance(exc, ValueError) else "Document extraction failed")
    saved = store.save("documents", {"documents": docs})
    return {"id": saved["id"], "pages": len(docs), "characters": sum(len(d["text"]) for d in docs)}


@app.post("/api/test-sets", dependencies=auth, status_code=201)
def test_sets(body: TestSet):
    return store.save("test_set", body.model_dump())


@app.post("/api/test-sets/upload", dependencies=auth, status_code=201)
async def upload_test_set(file: UploadFile = File(...)):
    try:
        body = parse_test_set(file.filename or "questions.csv", await file.read(MAX_BYTES + 1))
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(422, str(exc))
    return store.save("test_set", body.model_dump())


@app.post("/api/configurations", dependencies=auth, status_code=201)
def configurations(body: Configuration):
    return store.save("configuration", body.model_dump())


@app.post("/api/demo", dependencies=auth, status_code=201)
def demo():
    doc = store.save("documents", {"documents": extract_document("harbor-handbook.txt", (sample_dir / "harbor-handbook.txt").read_bytes())})
    test = test_sets(TestSet(**json.loads((sample_dir / "questions.json").read_text(encoding="utf-8"))))
    # Two candidates keep the ten-question demo within modest free judge quotas.
    configs = [configurations(Configuration(name="MiniLM · 96 tokens", chunk_size=96, overlap=16, top_k=2)),
               configurations(Configuration(name="BGE · 192 + rerank", embedding_model="BAAI/bge-small-en-v1.5", rerank=True, top_k=2))]
    return {"document_set_id": doc["id"], "test_set_id": test["id"],
            "configuration_ids": [c["id"] for c in configs], "configurations": configs,
            "questions": test["questions"]}


def worker(run_id):
    try:
        from backend.pipeline import execute_run
        execute_run(store, run_id)
    except Exception as exc:
        run = store.get("run", run_id)
        run.update(status="failed", error=type(exc).__name__ + ": worker could not start", stage="Worker failed")
        store.save("run", run, run_id)
    finally:
        run_lock.release()


@app.post("/api/runs", dependencies=auth, status_code=202)
def start_run(body: RunRequest):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(503, "Set GROQ_API_KEY on the backend before running evaluations")
    fetch("documents", body.document_set_id)
    test = fetch("test_set", body.test_set_id)
    configs = [fetch("configuration", cid) for cid in body.configuration_ids]
    if len({c["name"].strip().casefold() for c in configs}) != len(configs):
        raise HTTPException(422, "Configuration names must be distinct for comparison charts")
    if not run_lock.acquire(blocking=False):
        raise HTTPException(409, "An evaluation is already running. Wait for it to finish.")
    try:
        from backend.pipeline import now
        run = store.save("run", {**body.model_dump(), "configurations": configs, "questions": test["questions"],
            "status": "queued", "stage": "Queued", "created_at": now(), "completed": 0,
            "total": len(configs) * len(test["questions"]), "rows": [], "summary": [],
            "provenance": {"generator": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"), "judge": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
                "ragas": "0.3.9", "relevancy_embeddings": "sentence-transformers/all-MiniLM-L6-v2",
                "inference_backend": os.getenv("INFERENCE_BACKEND", "sentence-transformers"),
                "precision": "dynamic-int8" if os.getenv("INFERENCE_BACKEND") == "onnx" else "float32",
                "generator_temperature": 0, "judge_temperature": "Ragas default per metric",
                "judge_prompt_examples": max(0, int(os.getenv("JUDGE_EXAMPLES", "0"))),
                "reasoning_effort": "none" if os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").startswith("qwen/") else "low",
                "relevancy_strictness": 3}})
        executor.submit(worker, run["id"])
    except Exception:
        run_lock.release()
        raise
    return {"id": run["id"], "status": run["status"]}


@app.get("/api/runs/{run_id}", dependencies=auth)
def get_run(run_id: str):
    return fetch("run", run_id)


def spreadsheet_safe(value):
    # Prevent uploaded answers/questions becoming formulas when a CSV is opened in Excel.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


@app.get("/api/runs/{run_id}/export", dependencies=auth)
def export(run_id: str):
    run = fetch("run", run_id)
    if not run["rows"]:
        raise HTTPException(409, "No result rows available yet")
    rows = [{**{k: v for k, v in row.items() if k not in {"scores", "contexts", "errors"}},
             **row["scores"], "contexts": json.dumps(row["contexts"], ensure_ascii=False),
             "errors": json.dumps(row["errors"])} for row in run["rows"]]
    frame = pd.DataFrame(rows).map(spreadsheet_safe)
    return StreamingResponse(io.StringIO(frame.to_csv(index=False)), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="rag-bench-{run["id"]}.csv"'})
