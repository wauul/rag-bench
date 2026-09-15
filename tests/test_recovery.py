from copy import deepcopy
import pytest
from backend.models import METRICS
from backend.storage import Store
from scripts.retry_failed import recover


def test_recovery_preserves_successful_scores_and_failure_history(tmp_path, monkeypatch):
    import backend.pipeline as pipeline
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    monkeypatch.setenv("JUDGE_EXAMPLES", "0")
    store = Store(tmp_path)
    good = {"configuration_id": "cfg", "question_index": 0, "scores": {m: 0.75 for m in METRICS},
            "errors": {}, "answer": "Original answer", "latency_seconds": 1}
    bad = deepcopy(good)
    bad.update(question_index=1, errors={"faithfulness": "RateLimitError"})
    bad["scores"]["faithfulness"] = None
    original_bad = deepcopy(bad)
    run = store.save("run", {"status": "partial", "total": 2, "rows": [good, bad],
        "questions": [{}, {}], "configurations": [{"id": "cfg", "name": "Config"}],
        "provenance": {"generator": "openai/gpt-oss-120b", "judge": "openai/gpt-oss-120b",
                       "inference_backend": "sentence-transformers", "judge_prompt_examples": 0}})
    monkeypatch.setattr(pipeline, "make_llm", lambda: object())
    monkeypatch.setattr(pipeline, "load_embedder", lambda _: object())
    monkeypatch.setattr(pipeline, "make_metrics", lambda *_: dict.fromkeys(METRICS))
    calls = []

    async def score(metrics, row):
        calls.append((list(metrics), row["question_index"]))
        return {"faithfulness": 0.5}, {}

    monkeypatch.setattr(pipeline, "score_row", score)
    recover(store, run["id"])
    saved = store.get("run", run["id"])
    assert calls == [(["faithfulness"], 1)]
    assert saved["rows"][0] == good
    assert saved["rows"][1]["scores"]["answer_relevancy"] == 0.75
    assert saved["retry_history"][0]["previous_row"] == original_bad
    assert saved["status"] == "completed"

    saved["status"] = "partial"
    store.save("run", saved, run["id"])
    monkeypatch.setenv("GROQ_MODEL", "different-model")
    with pytest.raises(ValueError, match="original model"):
        recover(store, run["id"])
