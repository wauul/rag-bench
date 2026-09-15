"""Compare real compact embeddings/reranking against Sentence Transformers; no scoring mocks."""
import gc
import json
from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder
from backend.models import MODELS
from backend.pipeline import load_embedder, embed
from backend.onnx_inference import OnnxModel, RERANKER


def main():
    questions = json.loads(Path("sample_data/questions.json").read_text())["questions"]
    texts = [q["question"] for q in questions] + [q["reference"] for q in questions]
    for name in MODELS:
        torch_model = load_embedder(name)
        expected = np.asarray(embed(torch_model, texts))
        del torch_model
        gc.collect()
        compact = OnnxModel(name)
        actual = compact.encode(texts)
        similarities = (expected * actual).sum(axis=1)
        assert float(similarities.min()) > 0.97, similarities
        print(name, "minimum cosine agreement:", float(similarities.min()), flush=True)
        del compact
        gc.collect()
    pairs = [(questions[0]["question"], q["reference"]) for q in questions]
    baseline = CrossEncoder(RERANKER, device="cpu").predict(pairs)
    optimized = OnnxModel(RERANKER).predict(pairs)
    assert int(np.argmax(baseline)) == int(np.argmax(optimized)) == 0
    print("Reranker top passage preserved; max logit change:", float(np.max(np.abs(baseline - optimized))), flush=True)


if __name__ == "__main__":
    main()
