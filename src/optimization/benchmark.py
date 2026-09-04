"""
Phase 4 — rigorous, reproducible latency benchmark harness.

Built BEFORE optimizing (per the project scope's "hardest part #2"),
against a fixed hardware target, fixed batch size, N>=50 repetitions
with warmup runs discarded, reporting p50/p95/p99 — not a single
wall-clock number. Run identically against the unoptimized PyTorch
model, the FP32 ONNX export, and the INT8-quantized ONNX export so the
before/after comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.optimization.export_onnx import build_model_from_artifacts


def _percentiles(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "throughput_rps": round(1000.0 / float(np.percentile(arr, 50)), 2),
    }


def benchmark_callable(run_once, n_reps: int, n_warmup: int) -> dict:
    for _ in range(n_warmup):
        run_once()
    latencies = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        run_once()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return _percentiles(latencies)


def benchmark_pytorch(model, input_ids: torch.Tensor, n_reps: int, n_warmup: int) -> dict:
    model.eval()

    def run_once():
        with torch.no_grad():
            model(input_ids)

    return benchmark_callable(run_once, n_reps, n_warmup)


def benchmark_onnx(session: ort.InferenceSession, input_ids: np.ndarray, n_reps: int, n_warmup: int) -> dict:
    def run_once():
        session.run(None, {"input_ids": input_ids})

    return benchmark_callable(run_once, n_reps, n_warmup)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-dir", default="models/onnx")
    p.add_argument("--artifact-cache", default="models/production_cache")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-reps", type=int, default=100)
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--report", default="reports/benchmark_report.json")
    args = p.parse_args()

    assert args.n_reps >= 50, "benchmark harness requires N>=50 repetitions per the project scope"

    onnx_dir = Path(args.onnx_dir)
    config = json.loads((onnx_dir / "config.json").read_text())
    max_len = config["max_len"]

    # Find the cached unoptimized PyTorch artifacts (export_onnx.py already
    # downloaded them from the registry's production alias).
    candidates = list(Path(args.artifact_cache).rglob("model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No cached model.pt under {args.artifact_cache} — run export_onnx.py first."
        )
    artifact_dir = candidates[0].parent
    pt_model, _, _ = build_model_from_artifacts(artifact_dir)

    input_ids_torch = torch.randint(0, 100, (args.batch_size, max_len), dtype=torch.long)
    input_ids_np = input_ids_torch.numpy()

    print(f"Benchmarking on: {platform.processor() or platform.machine()}, "
          f"batch_size={args.batch_size}, N={args.n_reps}, warmup={args.n_warmup} (discarded)")

    results = {}

    print("Benchmarking unoptimized PyTorch (eager) model...")
    results["pytorch_fp32"] = benchmark_pytorch(pt_model, input_ids_torch, args.n_reps, args.n_warmup)
    results["pytorch_fp32"]["model_size_mb"] = round(
        sum(p.numel() * p.element_size() for p in pt_model.parameters()) / (1024 * 1024), 3
    )

    print("Benchmarking ONNX FP32...")
    sess_fp32 = ort.InferenceSession(str(onnx_dir / "model_fp32.onnx"), providers=["CPUExecutionProvider"])
    results["onnx_fp32"] = benchmark_onnx(sess_fp32, input_ids_np, args.n_reps, args.n_warmup)
    results["onnx_fp32"]["model_size_mb"] = round((onnx_dir / "model_fp32.onnx").stat().st_size / (1024 * 1024), 3)

    print("Benchmarking ONNX INT8 (dynamic quantization)...")
    sess_int8 = ort.InferenceSession(str(onnx_dir / "model_int8.onnx"), providers=["CPUExecutionProvider"])
    results["onnx_int8"] = benchmark_onnx(sess_int8, input_ids_np, args.n_reps, args.n_warmup)
    results["onnx_int8"]["model_size_mb"] = round((onnx_dir / "model_int8.onnx").stat().st_size / (1024 * 1024), 3)

    speedup = results["pytorch_fp32"]["p95_ms"] / results["onnx_int8"]["p95_ms"]
    size_reduction = results["pytorch_fp32"]["model_size_mb"] / results["onnx_int8"]["model_size_mb"]

    report = {
        "hardware": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "batch_size": args.batch_size,
        "n_reps": args.n_reps,
        "n_warmup_discarded": args.n_warmup,
        "results": results,
        "summary": {
            "p95_speedup_pytorch_to_onnx_int8": round(speedup, 2),
            "model_size_reduction_x": round(size_reduction, 2),
        },
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))

    print("\n=== Benchmark summary (p50 / p95 / p99 ms, batch_size=%d, N=%d) ===" % (args.batch_size, args.n_reps))
    for name, r in results.items():
        print(f"  {name:14s} p50={r['p50_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  p99={r['p99_ms']:.2f}ms  "
              f"size={r['model_size_mb']:.2f}MB")
    print(f"\np95 latency speedup (PyTorch -> ONNX INT8): {speedup:.2f}x")
    print(f"Model size reduction: {size_reduction:.2f}x")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
