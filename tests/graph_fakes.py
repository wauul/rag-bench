"""Deterministic local provider/model boundaries; never used by the shipped app."""

import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.models import METRICS, Configuration
from backend.provenance import snapshot_run
from backend.settings import load_settings
from backend.storage import Store

PASSAGES = ["Nearest text", "Second text", "Best evidence"]


def create_run(store, engines=("existing", "langgraph"), rerank=True, questions=1):
    docs = store.save(
        "documents", {"documents": [{"text": " ".join(PASSAGES), "source": "test.txt", "page": 2}]}
    )
    configs = [
        {
            "id": str(i),
            **Configuration(
                name=f"Engine {i}", engine=e, rerank=rerank, candidate_k=3, context_k=1
            ).model_dump(),
        }
        for i, e in enumerate(engines)
    ]
    run = store.save(
        "run",
        {
            "document_set_id": docs["id"],
            "questions": [
                {"question": f"Question {i}?", "reference": "SECRET_REFERENCE_EVAL_ONLY"}
                for i in range(questions)
            ],
            "configurations": configs,
            "status": "queued",
            "stage": "Queued",
            "created_at": "2026-10-07",
            "rows": [],
            "summary": [],
            "total": len(configs) * questions,
            "attempt": 1,
            "provenance": load_settings().provenance(),
        },
    )
    snapshot_run(store, run)
    return store.get("run", run["id"])


def install(monkeypatch, log_path=None, event=None, fail_metric=None, crash=None):
    from backend import pipeline, retrieval

    calls, prompts = Counter(), []

    def record(name):
        calls[name] += 1
        if log_path:
            with Path(log_path).open("a", encoding="utf-8") as file:
                file.write(name + "\n")

    class Embedder:
        def encode(self, texts, **kwargs):
            return np.array(
                [
                    [1.0, 0.0]
                    if t.startswith("Question")
                    else {
                        PASSAGES[0]: [1.0, 0.0],
                        PASSAGES[1]: [0.8, 0.6],
                        PASSAGES[2]: [0.6, 0.8],
                    }[t]
                    for t in texts
                ]
            )

    class Reranker:
        def predict(self, pairs, **kwargs):
            record("rerank")
            return [10.0 if text == PASSAGES[2] else 0.0 for _, text in pairs]

    class LLM:
        def invoke(self, value):
            messages = value.to_messages() if hasattr(value, "to_messages") else value
            content = [(m.type, m.content) if hasattr(m, "content") else m for m in messages]
            assert "SECRET_REFERENCE" not in str(content)
            prompts.append(content)
            record("generation")
            return SimpleNamespace(content="A saved answer")

    class Metric:
        def __init__(self, name):
            self.name = name

        async def single_turn_ascore(self, sample, **kwargs):
            record(self.name)
            assert sample.reference == "SECRET_REFERENCE_EVAL_ONLY"
            if self.name == fail_metric:
                raise TimeoutError("injected")
            if event and self.name == METRICS[0]:
                event.set()
            return 0.75

    monkeypatch.setattr(
        retrieval,
        "chunk_documents",
        lambda *_: [
            {"text": p, "source": "test.txt", "page": 2, "token_start": i, "token_end": i + 2}
            for i, p in enumerate(PASSAGES)
        ],
    )
    monkeypatch.setattr(retrieval, "load_embedder", lambda _: Embedder())
    monkeypatch.setattr(retrieval, "load_reranker", Reranker)
    monkeypatch.setattr(pipeline, "load_embedder", lambda _: Embedder())
    monkeypatch.setattr(pipeline, "make_llm", LLM)
    monkeypatch.setattr(pipeline, "make_metrics", lambda *_: {m: Metric(m) for m in METRICS})
    if crash:
        if crash == "initial_checkpoint":
            from langgraph.checkpoint.sqlite import SqliteSaver

            put = SqliteSaver.put

            def interrupted_put(self, *args, **kwargs):
                put(self, *args, **kwargs)
                os._exit(73)

            monkeypatch.setattr(SqliteSaver, "put", interrupted_put)
        if crash == "indexing":
            from chromadb.api.models.Collection import Collection

            upsert = Collection.upsert

            def interrupted_upsert(self, *args, **kwargs):
                upsert(self, *args, **kwargs)
                os._exit(73)

            monkeypatch.setattr(Collection, "upsert", interrupted_upsert)
        original = Store.save

        def save(self, kind, payload, object_id=None):
            rows = payload.get("rows", [])
            target = next((r for r in rows if r.get("engine") == "langgraph"), None)
            trigger = (
                (
                    crash == "before_call"
                    and "· Generating answer" in payload.get("stage", "")
                    and target
                    and not target["answer"]
                )
                or (
                    crash in {"before_answer_save", "after_answer_save"}
                    and target
                    and target["answer"]
                )
                or (
                    crash == "between_metrics"
                    and target
                    and target["scores"][METRICS[0]] is not None
                )
                or (crash == "retrieval" and kind == "graph_artifact")
            )
            if trigger and crash in {"before_call", "before_answer_save"}:
                os._exit(73)
            result = original(self, kind, payload, object_id)
            if trigger:
                os._exit(73)
            return result

        monkeypatch.setattr(Store, "save", save)
    return calls, prompts
