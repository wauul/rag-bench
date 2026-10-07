"""Use real persistent Chroma with tiny vectors to check candidate-pool selection."""

import numpy as np

from backend.models import Configuration
from backend.profiling import RunProfiler
from backend.retrieval import retrieve_questions
from backend.storage import Store


def test_reranking_selects_evidence_outside_original_top_k(tmp_path, monkeypatch):
    import backend.retrieval as retrieval

    store = Store(tmp_path)
    docs = store.save("documents", {"documents": []})
    run = store.save("run", {"document_set_id": docs["id"]})
    vectors = {
        "Which passage?": [1, 0],
        "nearest": [1, 0],
        "second": [0.8, 0.6],
        "best evidence": [0.6, 0.8],
        "distant": [0, 1],
    }
    chunks = [
        {"text": text, "source": "test.txt", "page": 1, "token_start": 0, "token_end": 2}
        for text in list(vectors)[1:]
    ]

    class Embedder:
        def encode(self, texts, **kwargs):
            return np.array([vectors[t] for t in texts])

    class Reranker:
        def predict(self, pairs, **kwargs):
            assert len(pairs) == 3
            return [10 if text == "best evidence" else 0 for _, text in pairs]

    monkeypatch.setattr(retrieval, "chunk_documents", lambda *_: chunks)
    monkeypatch.setattr(retrieval, "load_embedder", lambda _: Embedder())
    monkeypatch.setattr(retrieval, "load_reranker", Reranker)
    stages = []
    config = {
        "id": "cfg",
        **Configuration(name="Rerank", rerank=True, candidate_k=3, context_k=1).model_dump(),
    }
    questions = [{"question": "Which passage?"}]
    measured = {"rows": [], "attempt": 1}
    with RunProfiler(measured) as profiler:
        contexts = retrieve_questions(
            store, run["id"], config, questions, [0], lambda: None, stages.append
        )[0]
        profiler.status = "completed"
    entries = measured["profiling"]["attempts"][0]["spans"]
    assert [e["stage"] for e in entries] == [
        "chunking",
        "embedding_model_load",
        "indexing",
        "retrieval",
        "reranker_load",
        "reranking",
    ]
    assert all(e["configuration_id"] == "cfg" and e["seconds"] >= 0 for e in entries)
    assert next(e for e in entries if e["stage"] == "indexing")["chunks"] == 4
    assert measured["profiling"]["summary"]["rss_peak_mb"] > 0
    assert len(contexts) == 1 and contexts[0]["text"] == "best evidence"
    assert contexts[0]["rerank_score"] == 10
    # The same run/config collection can be rebuilt after an indexing interruption.
    again = retrieve_questions(
        store, run["id"], config, questions, [0], lambda: None, stages.append
    )[0]
    assert again == contexts
    config.update(rerank=False)
    assert (
        retrieve_questions(store, run["id"], config, questions, [0], lambda: None, stages.append)[
            0
        ][0]["text"]
        == "nearest"
    )
