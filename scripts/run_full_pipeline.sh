#!/usr/bin/env bash
# End-to-end pipeline runner — Phases 0 through 6, in order.
# Each step's artifact is what the next step consumes (see README
# architecture diagram). Safe to re-run from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Phase 0: data acquisition + preprocessing + EDA + baseline ==="
python -m src.data.acquire --n 60000 --out data/raw/complaints.csv
python -m src.data.preprocess --raw data/raw/complaints.csv --out-dir data/processed
python -m src.data.eda
python -m src.baseline.train_tfidf

echo "=== Phase 2: fine-tuning with experiment tracking ==="
# Sandbox-runnable path (no GPU / no HF Hub access required):
python -m src.training.train_sandbox_transformer
# Production path (requires GPU + internet — run on Colab instead, see
# notebooks/02_colab_finetune.ipynb):
#   python -m src.training.train_transformer --base-model distilbert-base-uncased

echo "=== Phase 3: model registry + promotion gate ==="
# See README for how the specific --run-id values below were chosen
# (worst variant bootstraps 'production', best variant is then promoted,
# a third weak candidate is used to demonstrate a documented reject).
echo "Run src/registry/promote.py manually with run_ids from reports/finetune_runs_summary.json"

echo "=== Phase 4: ONNX export + quantization + benchmark ==="
python -m src.optimization.export_onnx
python -m src.optimization.benchmark

echo "=== Phase 5: tests + CI regression gate ==="
pytest tests/ -v
python -m src.registry.ci_check

echo "=== Phase 6: serving + monitoring ==="
echo "Start the API with:  uvicorn src.serving.app:app --reload"
echo "Or via Docker:       docker compose up --build"
echo "Demo UI:              python -m src.demo.gradio_app"
echo "Drift report:          python -m src.monitoring.drift_report"

echo "Pipeline complete."
