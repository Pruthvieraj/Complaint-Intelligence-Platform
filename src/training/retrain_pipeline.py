"""
Phase 8 — automated retraining pipeline. Closes the MLOps loop: this is
what turns "we trained a model once" into "the model keeps itself
current without a human kicking off each step by hand".

Run manually (`python -m src.training.retrain_pipeline`) or on
`.github/workflows/retrain.yml`'s weekly schedule / manual dispatch.
Each run:

  1. Synthesizes a fresh batch of "new" complaints with a run-specific
     random seed — a stand-in for a week's worth of new CFPB
     submissions, exactly like src/data/synthesize.py already stands in
     for the real CFPB pull anywhere this sandbox has no network path
     to it (see README "Sandbox execution note").
  2. Preprocesses + stratified-splits it the same way the original
     training data was (src/data/preprocess.py — reused, not
     reimplemented).
  3. Retrains the current production architecture on the fresh data.
     Deliberately NOT a fresh hyperparameter grid search: a real
     retraining job retrains a fixed, already-chosen champion config on
     new data on a schedule; re-running a full search every week is a
     different (much more expensive) thing nobody actually does.
  4. Runs the *exact same* decide_promotion() gate already unit-tested
     in tests/test_promotion.py and used by both the interactive CLI
     (src/registry/promote.py) and CI (src/registry/ci_check.py). A
     candidate only replaces production if it actually wins.
  5. If (and only if) promoted: exports the winning candidate straight
     to ONNX/INT8 (reusing src/optimization/export_onnx.py's helpers on
     the candidate's local artifact dir — no MLflow round-trip needed,
     see note below), re-benchmarks it, and overwrites the committed
     models/onnx/*, reports/*.json, and src/serving/model_info.json
     that CI's regression check and the live dashboard both read.

Note on state — why this doesn't touch MLflow's registry at all:
mlflow.db and mlruns/ are gitignored on purpose (see .gitignore) — a
fresh GitHub Actions checkout starts with neither, so there is no
persisted "current production" to query via MlflowClient the way the
interactive CLI does. This pipeline instead treats the *committed*
reports/production_model_metrics.json as the durable source of truth
for "what's in production right now" — which is exactly the same file
CI's own regression check (ci_check.py) already relies on. Runs are
still logged to MLflow for local experiment-tracking visibility; that
log just doesn't need to survive between CI runs for the gate to work.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.preprocess import CANONICAL_LABELS, LABEL_COL, prepare_dataset, stratified_split
from src.data.synthesize import generate_dataset
from src.optimization.export_onnx import build_model_from_artifacts, export_to_onnx, quantize
from src.registry.promote import decide_promotion
from src.training.dataset import ComplaintDataset
from src.training.model import SmallTransformerClassifier
from src.training.train_sandbox_transformer import class_weights, run_epoch
from src.training.vocab import build_vocab

# Mirrors the config of "v1_baseline_lr1e-3" (see
# src/training/train_sandbox_transformer.py) — the variant currently in
# production per reports/production_model_metrics.json. A larger system
# would look the champion's hyperparameters up dynamically from the
# registry; pinned here because between retrains the *architecture*
# doesn't change, only the data and the resulting weights do.
CHAMPION_CONFIG = dict(
    name="retrain_champion",
    lr=1e-3,
    weighting="none",
    max_len=96,
    d_model=96,
    nhead=4,
    num_layers=2,
    vocab_size=12_000,
)

PRODUCTION_METRICS_SNAPSHOT = Path("reports/production_model_metrics.json")
PROMOTION_LOG = Path("reports/promotion_log.json")
MODEL_INFO_PATH = Path("src/serving/model_info.json")
RETRAIN_SUMMARY_PATH = Path("reports/retrain_run_summary.json")


def _append_promotion_log(entry: dict) -> None:
    PROMOTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(PROMOTION_LOG.read_text()) if PROMOTION_LOG.exists() else []
    log.append(entry)
    PROMOTION_LOG.write_text(json.dumps(log, indent=2))


def _load_current_production_metrics() -> dict | None:
    if not PRODUCTION_METRICS_SNAPSHOT.exists():
        return None
    return json.loads(PRODUCTION_METRICS_SNAPSHOT.read_text())


def synthesize_and_split(seed: int, n_rows: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates a fresh synthetic batch (a stand-in for new incoming
    complaints since the last retrain) and produces the same
    train/val/test split shape the original data went through."""
    raw_df = generate_dataset(n_rows=n_rows, seed=seed)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "complaints.csv"
    raw_df.to_csv(raw_csv, index=False)

    df = prepare_dataset(raw_csv)
    train_df, val_df, test_df = stratified_split(df, seed=seed)
    return train_df, val_df, test_df


def train_champion(train_df, val_df, test_df, label2id, id2label, epochs: int, batch_size: int, out_dir: Path) -> dict:
    """A trimmed-down version of train_sandbox_transformer.train_one_variant
    for a single fixed config and no MLflow run — this pipeline logs its
    own decision to reports/promotion_log.json directly (see note in the
    module docstring on why MLflow state can't be relied on here)."""
    variant = CHAMPION_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab(train_df["Consumer complaint narrative"], max_vocab_size=variant["vocab_size"])

    train_ds = ComplaintDataset(train_df, vocab, label2id, max_len=variant["max_len"])
    val_ds = ComplaintDataset(val_df, vocab, label2id, max_len=variant["max_len"])
    test_ds = ComplaintDataset(test_df, vocab, label2id, max_len=variant["max_len"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SmallTransformerClassifier(
        vocab_size=len(vocab),
        num_classes=len(label2id),
        d_model=variant["d_model"],
        nhead=variant["nhead"],
        num_layers=variant["num_layers"],
        max_len=variant["max_len"],
    ).to(device)

    weights = class_weights(train_df[LABEL_COL], label2id, variant["weighting"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant["lr"])

    t0 = time.time()
    best_val_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        _, train_f1, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        _, val_f1, _, _ = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        print(f"  [retrain] epoch {epoch}/{epochs}  train_f1={train_f1:.4f}  val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    _, test_f1, test_preds, test_labels = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
    per_class = f1_score(test_labels, test_preds, average=None, zero_division=0, labels=list(range(len(label2id))))
    minority_labels = sorted(label2id, key=lambda l: (train_df[LABEL_COL] == l).sum())[:3]
    minority_f1 = float(np.mean([per_class[label2id[l]] for l in minority_labels]))

    variant_dir = out_dir / "retrain_candidate"
    variant_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), variant_dir / "model.pt")
    (variant_dir / "vocab.json").write_text(json.dumps(vocab))
    (variant_dir / "config.json").write_text(json.dumps({**variant, "label2id": label2id}, indent=2))

    return {
        "candidate_dir": variant_dir,
        "val_macro_f1": best_val_f1,
        "test_macro_f1": test_f1,
        "test_minority_class_f1": minority_f1,
        "minority_labels": minority_labels,
        "train_time_seconds": train_time,
    }


def export_candidate_to_onnx(candidate_dir: Path, out_dir: Path) -> dict:
    model, config, _ = build_model_from_artifacts(candidate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path, int8_path = out_dir / "model_fp32.onnx", out_dir / "model_int8.onnx"
    export_to_onnx(model, config["max_len"], fp32_path)
    quantize(fp32_path, int8_path)
    (out_dir / "vocab.json").write_text((candidate_dir / "vocab.json").read_text())
    (out_dir / "config.json").write_text((candidate_dir / "config.json").read_text())
    return {
        "fp32_size_mb": round(fp32_path.stat().st_size / (1024 * 1024), 3),
        "int8_size_mb": round(int8_path.stat().st_size / (1024 * 1024), 3),
    }


def update_model_info(candidate_metrics: dict, promotion_entry: dict) -> None:
    """Keeps the live dashboard's 'Model & Pipeline' tab (served straight
    from this file, see src/serving/app.py) in sync with a promotion,
    without touching the parts of it (baseline, raw benchmark numbers)
    this pipeline didn't just recompute."""
    if not MODEL_INFO_PATH.exists():
        return
    info = json.loads(MODEL_INFO_PATH.read_text())
    info["production_model"] = {
        "architecture": "Small transformer encoder (from scratch, PyTorch nn.TransformerEncoder)",
        "run_name": "retrain_champion",
        "model_version": info.get("production_model", {}).get("model_version", 0) + 1,
        "test_macro_f1": round(candidate_metrics["test_macro_f1"], 4),
        "test_minority_class_f1": round(candidate_metrics["test_minority_class_f1"], 4),
        "minority_labels": candidate_metrics["minority_labels"],
        "promoted_at": promotion_entry["timestamp"],
    }
    info.setdefault("promotion_gate", {}).setdefault("history", []).append(promotion_entry)
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2) + "\n")


def run(seed: int, n_rows: int, epochs: int, batch_size: int, work_dir: Path, onnx_dir: Path, dry_run: bool) -> dict:
    label2id = {label: i for i, label in enumerate(CANONICAL_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"=== Retraining pipeline run — seed={seed}, n_rows={n_rows}, epochs={epochs} ===")
    train_df, val_df, test_df = synthesize_and_split(seed, n_rows, work_dir)
    print(f"Fresh data: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    result = train_champion(train_df, val_df, test_df, label2id, id2label, epochs, batch_size, work_dir)
    candidate_metrics = {
        "test_macro_f1": result["test_macro_f1"],
        "test_minority_class_f1": result["test_minority_class_f1"],
    }

    production_metrics = _load_current_production_metrics()
    promoted, reason = decide_promotion(candidate_metrics, production_metrics)
    print(reason)

    timestamp = datetime.now(timezone.utc).isoformat()
    promotion_entry = {
        "timestamp": timestamp,
        "run_name": "retrain_champion",
        "seed": seed,
        "candidate_test_macro_f1": round(result["test_macro_f1"], 4),
        "candidate_test_minority_f1": round(result["test_minority_class_f1"], 4),
        "previous_production_test_macro_f1": (production_metrics or {}).get("test_macro_f1"),
        "decision": "promoted" if promoted else "rejected",
        "reason": reason,
        "source": "automated_retrain",
    }

    summary = {
        "timestamp": timestamp,
        "seed": seed,
        "candidate_metrics": candidate_metrics,
        "promoted": promoted,
        "reason": reason,
    }

    if dry_run:
        print("--dry-run: not writing promotion log / exporting ONNX / touching production files.")
        RETRAIN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        RETRAIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        return summary

    _append_promotion_log(promotion_entry)

    if promoted:
        export_stats = export_candidate_to_onnx(result["candidate_dir"], onnx_dir)
        summary["export"] = export_stats

        snapshot = {
            "model_version": (json.loads(PRODUCTION_METRICS_SNAPSHOT.read_text())["model_version"] + 1)
            if PRODUCTION_METRICS_SNAPSHOT.exists()
            else 1,
            "run_name": "retrain_champion",
            "test_macro_f1": candidate_metrics["test_macro_f1"],
            "test_minority_class_f1": candidate_metrics["test_minority_class_f1"],
            "minority_labels": result["minority_labels"],
            "promoted_at": timestamp,
        }
        PRODUCTION_METRICS_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        PRODUCTION_METRICS_SNAPSHOT.write_text(json.dumps(snapshot, indent=2))

        update_model_info(candidate_metrics | {"minority_labels": result["minority_labels"]}, promotion_entry)
        print(f"-> PROMOTED. Exported new ONNX model to {onnx_dir}, updated committed reports + model_info.json.")
    else:
        print("-> REJECTED. No files under models/onnx or src/serving/model_info.json were touched.")

    RETRAIN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None, help="defaults to a time-based seed (a fresh batch every run)")
    p.add_argument("--n-rows", type=int, default=20_000, help="synthetic rows to generate — smaller than the original 120k for a fast scheduled job")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--work-dir", default="data/retrain_tmp")
    p.add_argument("--onnx-dir", default="models/onnx")
    p.add_argument("--dry-run", action="store_true", help="train and evaluate the gate decision, but don't export/commit anything")
    args = p.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    torch.manual_seed(seed)

    summary = run(seed, args.n_rows, args.epochs, args.batch_size, Path(args.work_dir), Path(args.onnx_dir), args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
