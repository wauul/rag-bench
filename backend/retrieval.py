"""Local token windows, embeddings and candidate selection."""

import gc
import os
from functools import lru_cache

from backend.models import MODEL_REVISIONS, MODELS, Configuration
from backend.profiling import measure


@lru_cache(maxsize=1)
def tokenizer():
    if os.getenv("INFERENCE_BACKEND") == "onnx":
        from backend.onnx_inference import ChunkTokenizer

        return ChunkTokenizer()
    from transformers import AutoTokenizer

    # Use one fixed tokenizer for both configurations: chunk boundaries do not drift by model.
    return AutoTokenizer.from_pretrained(MODELS[0], revision=MODEL_REVISIONS[MODELS[0]])


def chunk_documents(documents, config):
    tok = tokenizer()
    chunks = []
    for doc in documents:
        # Sliding WordPiece windows retain overlap and exact source substrings, including case.
        # Each source/page is chunked independently so citations never cross document boundaries.
        offsets = tok(
            doc["text"], add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )["offset_mapping"]
        for start in range(0, len(offsets), config.chunk_size - config.overlap):
            end = min(start + config.chunk_size, len(offsets))
            if end <= start:
                break
            a, b = offsets[start][0], offsets[end - 1][1]
            chunks.append(
                {
                    "text": doc["text"][a:b],
                    "source": doc["source"],
                    "page": doc["page"],
                    "token_start": start,
                    "token_end": end,
                }
            )
            if end == len(offsets):
                break
    if not chunks:
        raise ValueError("Documents produced no text chunks")
    if len(chunks) > 2000:
        raise ValueError("Too many chunks; reduce document size or overlap")
    return chunks


def load_embedder(name):
    if os.getenv("INFERENCE_BACKEND") == "onnx":
        from backend.onnx_inference import OnnxModel

        return OnnxModel(name)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu", revision=MODEL_REVISIONS[name])


def embed(model, texts, query=False, name=""):
    if query and name == MODELS[1]:
        texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
    return model.encode(
        texts, normalize_embeddings=True, batch_size=8, show_progress_bar=False
    ).tolist()


def load_reranker():
    if os.getenv("INFERENCE_BACKEND") == "onnx":
        from backend.onnx_inference import OnnxModel

        return OnnxModel("cross-encoder/ms-marco-MiniLM-L-6-v2")
    from sentence_transformers import CrossEncoder

    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return CrossEncoder(name, device="cpu", revision=MODEL_REVISIONS[name])


def select_contexts(contexts, config, question, reranker=None):
    if config.rerank:
        values = reranker.predict([(question, c["text"]) for c in contexts], batch_size=8)
        for context, value in zip(contexts, values):
            context["rerank_score"] = float(value)
        contexts.sort(key=lambda c: c["rerank_score"], reverse=True)
    return contexts[: config.context_k]


def retrieve_questions(store, run_id, config_data, questions, indexes, checkpoint, stage):
    """Rebuild incomplete indexes idempotently; saved passages are never retrieved again."""
    import chromadb
    from chromadb.config import Settings

    config = Configuration(**config_data)
    checkpoint()
    stage(f"Indexing {config.name}")
    from backend.provenance import run_documents

    documents = run_documents(store, store.get("run", run_id))
    labels = {"configuration_id": config_data["id"]}
    with measure("chunking", **labels):
        chunks = chunk_documents(documents, config)
    with measure("embedding_model_load", model=config.embedding_model, **labels):
        model = load_embedder(config.embedding_model)
    try:
        checkpoint()
        with measure("indexing", chunks=len(chunks), **labels):
            client = chromadb.PersistentClient(
                path=str(store.root / "chroma"), settings=Settings(anonymized_telemetry=False)
            )
            collection = client.get_or_create_collection(
                f"run-{run_id}-cfg-{config_data['id']}", metadata={"hnsw:space": "cosine"}
            )
            for start in range(0, len(chunks), 64):
                checkpoint()
                batch = chunks[start : start + 64]
                collection.upsert(
                    ids=[str(i) for i in range(start, start + len(batch))],
                    documents=[c["text"] for c in batch],
                    embeddings=embed(model, [c["text"] for c in batch]),
                    metadatas=[{k: v for k, v in c.items() if k != "text"} for c in batch],
                )
        retrieved = {}
        stage(f"Retrieving {config.name}")
        for index in indexes:
            checkpoint()
            with measure("retrieval", question_index=index, **labels):
                query = embed(
                    model, [questions[index]["question"]], query=True, name=config.embedding_model
                )
                result = collection.query(
                    query_embeddings=query, n_results=min(config.candidate_k, len(chunks))
                )
                retrieved[index] = [
                    {**meta, "text": text, "distance": float(distance), "chunk_id": cid}
                    for meta, text, distance, cid in zip(
                        result["metadatas"][0],
                        result["documents"][0],
                        result["distances"][0],
                        result["ids"][0],
                    )
                ]
    finally:
        del model
        gc.collect()
    reranker = None
    try:
        if config.rerank:
            checkpoint()
            stage(f"Reranking {config.name}")
            with measure("reranker_load", model="cross-encoder/ms-marco-MiniLM-L-6-v2", **labels):
                reranker = load_reranker()
        for index, contexts in retrieved.items():
            checkpoint()
            if config.rerank:
                with measure("reranking", question_index=index, **labels):
                    retrieved[index] = select_contexts(
                        contexts, config, questions[index]["question"], reranker
                    )
            else:
                retrieved[index] = select_contexts(
                    contexts, config, questions[index]["question"], reranker
                )
        return retrieved
    finally:
        del reranker
        gc.collect()
