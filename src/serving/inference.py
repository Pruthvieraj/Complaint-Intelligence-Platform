"""Shared inference logic for the FastAPI service (Phase 6) and the
Gradio demo (src/demo/gradio_app.py) — one code path, so "the model
behind the API" and "the model behind the demo" can never silently
drift apart."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.training.vocab import encode


class ComplaintClassifier:
    def __init__(self, onnx_dir: str | Path = "models/onnx", prefer_quantized: bool = True):
        onnx_dir = Path(onnx_dir)
        model_path = onnx_dir / ("model_int8.onnx" if prefer_quantized else "model_fp32.onnx")
        if not model_path.exists():
            model_path = onnx_dir / "model_fp32.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No ONNX model found under {onnx_dir}. Run "
                "`python -m src.optimization.export_onnx` first (Phase 4)."
            )
        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.vocab = json.loads((onnx_dir / "vocab.json").read_text())
        self.config = json.loads((onnx_dir / "config.json").read_text())
        self.id2label = {int(v): k for k, v in self.config["label2id"].items()}
        self.max_len = self.config["max_len"]

    def predict(self, text: str, top_k: int = 3) -> dict:
        t0 = time.perf_counter()
        ids = np.array([encode(text, self.vocab, self.max_len)], dtype=np.int64)
        logits = self.session.run(None, {"input_ids": ids})[0][0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        order = np.argsort(-probs)[:top_k]
        top = [{"label": self.id2label[int(i)], "probability": round(float(probs[i]), 4)} for i in order]

        return {
            "predicted_category": self.id2label[int(order[0])],
            "confidence": round(float(probs[order[0]]), 4),
            "latency_ms": round(latency_ms, 3),
            "top_k": top,
        }
