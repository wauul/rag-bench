"""Download official ONNX weights and quantize them for the compact Render runtime."""
import json
import shutil
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from backend.models import MODELS
from backend.onnx_inference import model_dir, RERANKER


def main():
    for name in MODELS + [RERANKER]:
        directory = model_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "model.int8.onnx"
        tokenizer_path = hf_hub_download(name, "tokenizer.json")
        shutil.copyfile(tokenizer_path, directory / "tokenizer.json")
        if not target.exists():
            if name == MODELS[1]:
                source = hf_hub_download(name, "onnx/model.onnx")
                quantize_dynamic(source, str(target), weight_type=QuantType.QUInt8,
                                 op_types_to_quantize=["MatMul", "Gemm"])
            else:
                source = hf_hub_download(name, "onnx/model_quint8_avx2.onnx")
                shutil.copyfile(source, target)
        (directory / "provenance.json").write_text(json.dumps({"model": name, "runtime": "onnxruntime",
            "precision": "dynamic-int8", "tokenizer_revision": tokenizer_path.replace('\\', '/').split('/snapshots/')[-1].split('/')[0]}))
        print(f"Prepared {name}: {target.stat().st_size / 1024 ** 2:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
