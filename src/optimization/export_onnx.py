"""
Phase 4 — export the promoted model to ONNX, then apply dynamic
quantization.

Loads whatever is currently aliased "production" in the MLflow Model
Registry (Phase 3), so this always optimizes the model that actually
passed the promotion gate — not just whatever's newest on disk.

Production note: for a real HF DistilBERT/FinBERT model this whole
script is one command — `optimum-cli export onnx --model <dir> <out>`
— which is the path to use once train_transformer.py (Phase 2
production path) has produced a real fine-tune. The hand-written
export below exists because the sandbox-demo model
(src/training/model.py) is a plain PyTorch nn.Module, not an
optimum-wrapped HF model, so it needs a plain torch.onnx.export call
instead of the optimum CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.registry.promote import MODEL_NAME, PRODUCTION_ALIAS
from src.training.model import SmallTransformerClassifier


def load_production_artifacts(out_dir: Path) -> Path:
    """Download the current production model version's artifacts
    (model.pt, vocab.json, config.json) to out_dir."""
    uri = f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"
    local_path = mlflow.artifacts.download_artifacts(uri, dst_path=str(out_dir))
    return Path(local_path)


def build_model_from_artifacts(artifact_dir: Path) -> tuple[torch.nn.Module, dict, int]:
    config = json.loads((artifact_dir / "config.json").read_text())
    vocab = json.loads((artifact_dir / "vocab.json").read_text())
    model = SmallTransformerClassifier(
        vocab_size=len(vocab),
        num_classes=len(config["label2id"]),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        max_len=config["max_len"],
    )
    state_dict = torch.load(artifact_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, len(vocab)


def export_to_onnx(model: torch.nn.Module, max_len: int, out_path: Path):
    dummy_input = torch.randint(0, 100, (1, max_len), dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(out_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,  # legacy TorchScript-based exporter — plays much more
        # reliably with onnxruntime's shape-inference-dependent quantization
        # tooling than torch's newer dynamo-based exporter (see README notes
        # / commit history: the dynamo path produced a graph that tripped
        # onnxruntime.quantization's shape inference).
    )


def quantize(onnx_fp32_path: Path, onnx_int8_path: Path):
    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QUInt8,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="models/onnx")
    p.add_argument("--artifact-cache", default="models/production_cache")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = load_production_artifacts(Path(args.artifact_cache))
    # download_artifacts on a directory URI nests under the original relpath (e.g. "model/")
    candidates = list(artifact_dir.rglob("model.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model.pt found under {artifact_dir}")
    artifact_dir = candidates[0].parent

    model, config, vocab_size = build_model_from_artifacts(artifact_dir)

    fp32_path = out_dir / "model_fp32.onnx"
    int8_path = out_dir / "model_int8.onnx"
    export_to_onnx(model, config["max_len"], fp32_path)
    quantize(fp32_path, int8_path)

    # also copy vocab/config alongside so serving (Phase 6) is self-contained
    (out_dir / "vocab.json").write_text((artifact_dir / "vocab.json").read_text())
    (out_dir / "config.json").write_text((artifact_dir / "config.json").read_text())

    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    int8_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"FP32 ONNX size: {fp32_mb:.2f} MB")
    print(f"INT8 ONNX size: {int8_mb:.2f} MB  ({fp32_mb / int8_mb:.2f}x smaller)")
    print(f"Wrote {fp32_path}, {int8_path}, vocab.json, config.json to {out_dir}")


if __name__ == "__main__":
    main()
