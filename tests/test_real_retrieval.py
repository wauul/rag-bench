"""Opt-in integration: real pinned local models, Chroma and graph; fake provider only.

Run inside the compact image with RAGBENCH_TEST_REAL_MODELS=1. Scores are test
fixtures in pytest's temporary store and have no answer-quality interpretation.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import pipeline
from backend.models import METRICS, MODELS, Configuration
from backend.provenance import snapshot_run
from backend.settings import load_settings
from backend.storage import Store


@pytest.mark.skipif(
    os.getenv("RAGBENCH_TEST_REAL_MODELS") != "1",
    reason="Opt in after provisioning pinned local models",
)
@pytest.mark.parametrize("model", MODELS)
def test_real_embedding_reranking_and_graph_prompt_parity(tmp_path, monkeypatch, model):
    store = Store(tmp_path)
    root = Path(__file__).resolve().parents[1]
    docs = store.save(
        "documents",
        {
            "documents": [
                {
                    "text": (root / "sample_data/harbor-handbook.txt").read_text(),
                    "source": "harbor-handbook.txt",
                    "page": 1,
                }
            ]
        },
    )
    question = json.loads((root / "sample_data/questions.json").read_text())["questions"][0]
    configurations = [
        {
            "id": str(i),
            **Configuration(
                name=engine,
                engine=engine,
                embedding_model=model,
                rerank=True,
                candidate_k=12,
                context_k=2,
            ).model_dump(),
        }
        for i, engine in enumerate(("existing", "langgraph"))
    ]
    run = store.save(
        "run",
        {
            "document_set_id": docs["id"],
            "configurations": configurations,
            "questions": [question],
            "provenance": load_settings().provenance(),
            "rows": [],
            "status": "queued",
            "summary": [],
            "total": 2,
        },
    )
    snapshot_run(store, run)
    prompts = []

    class LLM:
        def invoke(self, value):
            messages = value.to_messages() if hasattr(value, "to_messages") else value
            prompts.append([(m.type, m.content) if hasattr(m, "content") else m for m in messages])
            return SimpleNamespace(content="Fixed provider fixture, not a measured answer")

    class Metric:
        async def single_turn_ascore(self, *args, **kwargs):
            return 0.5

    monkeypatch.setattr(pipeline, "make_llm", LLM)
    monkeypatch.setattr(pipeline, "make_metrics", lambda *_: {m: Metric() for m in METRICS})
    result = pipeline.execute_run(store, run["id"])
    assert result["status"] == "completed", result.get("error")
    assert result["rows"][0]["contexts"] == result["rows"][1]["contexts"]
    assert len(result["rows"][0]["contexts"]) == 2
    assert all("rerank_score" in c for c in result["rows"][0]["contexts"])
    assert prompts[0] == prompts[1]
