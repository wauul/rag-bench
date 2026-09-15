"""Download and exercise both real embedding models, Chroma persistence and reranking (no key needed)."""
import gc
import json
import os
from pathlib import Path
import tempfile
import psutil
import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
from backend.models import Configuration, MODELS
from backend.pipeline import chunk_documents, embed, load_embedder
from backend.ingestion import extract_document


def main():
    root = Path(__file__).resolve().parents[1]
    docs = extract_document("harbor-handbook.txt", (root / "sample_data/harbor-handbook.txt").read_bytes())
    questions = json.loads((root / "sample_data/questions.json").read_text())["questions"]
    report = []
    for index, name in enumerate(MODELS):
        config = Configuration(name=name, embedding_model=name, chunk_size=96 if index == 0 else 192, overlap=16)
        chunks = chunk_documents(docs, config)
        model = load_embedder(name)
        path = Path(tempfile.mkdtemp(prefix="rag-bench-test-"))
        client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        collection = client.create_collection("retrieval-test", metadata={"hnsw:space": "cosine"})
        collection.add(ids=[str(i) for i in range(len(chunks))], documents=[c["text"] for c in chunks],
                       embeddings=embed(model, [c["text"] for c in chunks]))
        hits = []
        for q in questions:
            result = collection.query(query_embeddings=embed(model, [q["question"]], query=True, name=name), n_results=3)
            hits.append({"question": q["question"], "contexts": result["documents"][0]})
        # Verify the native tokenizer's embedding window does not silently truncate our chunks.
        lengths = [len(model.tokenizer(c["text"])["input_ids"]) for c in chunks]
        assert max(lengths) <= model.max_seq_length
        assert (path / "chroma.sqlite3").exists()
        report.append({"model": name, "chunks": len(chunks), "max_tokens": max(lengths),
                       "process_rss_mb": round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2), "retrievals": hits})
        print(f"OK: {name}: {len(chunks)} chunks, 10 questions, RSS {report[-1]['process_rss_mb']} MB", flush=True)
        del model
        gc.collect()
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    hit = report[-1]["retrievals"][0]
    scores = reranker.predict([(hit["question"], text) for text in hit["contexts"]])
    assert len(scores) == 3
    print("OK: real cross-encoder scores:", scores.tolist(), flush=True)
    out = root / "data/retrieval-check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
