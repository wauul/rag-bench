"""Compact CPU deployment of the same transformer models; no PyTorch import at runtime.

MiniLM/BGE use their original tokenizers and pooling (mean/CLS respectively).
Weights are dynamically int8 quantized at image build time. Quantization is disclosed in
run provenance; scripts/check_onnx.py compares against Sentence Transformers directly.
"""
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from backend.models import MODELS

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def model_dir(name):
    return Path(os.getenv("ONNX_MODEL_DIR", "data/onnx")) / name.replace("/", "--")


class ChunkTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer.from_file(str(model_dir(MODELS[0]) / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def __call__(self, text, **kwargs):
        return {"offset_mapping": self.tokenizer.encode(text, add_special_tokens=False).offsets}


class OnnxModel:
    def __init__(self, name):
        self.name = name
        directory = model_dir(name)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256 if name == MODELS[0] else 512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(directory / "model.int8.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def forward(self, inputs):
        tokens = self.tokenizer.encode_batch(inputs)
        values = {"input_ids": np.array([t.ids for t in tokens], dtype=np.int64),
                  "attention_mask": np.array([t.attention_mask for t in tokens], dtype=np.int64),
                  "token_type_ids": np.array([t.type_ids for t in tokens], dtype=np.int64)}
        output = self.session.run(None, {k: v for k, v in values.items() if k in self.inputs})[0]
        return output, values["attention_mask"]

    def encode(self, texts, **kwargs):
        embeddings = []
        # Small batches contain activation memory as well as model-weight memory.
        for start in range(0, len(texts), 2):
            output, mask = self.forward(texts[start:start + 2])
            if self.name == MODELS[1]:
                pooled = output[:, 0, :]
            else:
                expanded = mask[..., None]
                pooled = (output * expanded).sum(axis=1) / np.maximum(expanded.sum(axis=1), 1)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            embeddings.extend(pooled)
        return np.asarray(embeddings, dtype=np.float32)

    def predict(self, pairs, **kwargs):
        return np.asarray([float(self.forward([pair])[0].reshape(-1)[0]) for pair in pairs])
