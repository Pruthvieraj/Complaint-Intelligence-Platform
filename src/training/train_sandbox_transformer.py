"""
Phase 2 (sandbox execution path) — fine-tuning with experiment tracking.

Every run below is a real training run, tracked as a real, comparable
MLflow experiment, exactly per the Phase-2 deliverable. What's
*substituted* for sandbox reasons is the model: this trains a small
Transformer encoder from scratch (src/training/model.py) instead of
fine-tuning a pretrained DistilBERT, because this sandbox has no
network path to huggingface.co (see README "Sandbox execution note").

For the real submission, run src/training/train_transformer.py instead
(same MLflow tracking server, same variant-grid structure, real
DistilBERT/FinBERT weights) on Colab or any machine with GPU + internet.

Both scripts write to the same MLflow experiment name so registry/
promotion (Phase 3) work identically regardless of which one produced
the candidate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.preprocess import CANONICAL_LABELS, LABEL_COL
from src.training.dataset import ComplaintDataset
from src.training.model import SmallTransformerClassifier
from src.training.vocab import build_vocab

EXPERIMENT_NAME = "complaint-classification-finetune"


def class_weights(train_labels: pd.Series, label2id: dict, strategy: str) -> torch.Tensor:
    counts = train_labels.value_counts()
    n_classes = len(label2id)
    weights = torch.ones(n_classes)
    if strategy == "none":
        return weights
    for label, idx in label2id.items():
        n = counts.get(label, 1)
        if strategy == "inverse_freq":
            weights[idx] = 1.0 / n
        elif strategy == "inverse_sqrt_freq":
            weights[idx] = 1.0 / np.sqrt(n)
    weights = weights / weights.sum() * n_classes  # normalize to mean 1
    return weights


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), macro_f1, all_preds, all_labels


def train_one_variant(variant: dict, train_df, val_df, test_df, label2id, id2label, device, epochs, batch_size, out_dir):
    vocab = build_vocab(train_df["Consumer complaint narrative"], max_vocab_size=variant.get("vocab_size", 12_000))

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

    with mlflow.start_run(run_name=variant["name"]):
        mlflow.log_params({**variant, "epochs": epochs, "batch_size": batch_size, "vocab_actual_size": len(vocab)})

        t0 = time.time()
        best_val_f1 = -1.0
        best_state = None
        for epoch in range(1, epochs + 1):
            train_loss, train_f1, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
            val_loss, val_f1, val_preds, val_labels = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
            mlflow.log_metrics(
                {"train_loss": train_loss, "train_macro_f1": train_f1, "val_loss": val_loss, "val_macro_f1": val_f1},
                step=epoch,
            )
            print(f"  [{variant['name']}] epoch {epoch}/{epochs}  train_f1={train_f1:.4f}  val_f1={val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        train_time = time.time() - t0
        model.load_state_dict(best_state)

        test_loss, test_f1, test_preds, test_labels = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
        per_class = f1_score(test_labels, test_preds, average=None, zero_division=0, labels=list(range(len(label2id))))
        minority_labels = sorted(label2id, key=lambda l: (train_df[LABEL_COL] == l).sum())[:3]
        minority_f1 = float(np.mean([per_class[label2id[l]] for l in minority_labels]))

        mlflow.log_metrics(
            {
                "best_val_macro_f1": best_val_f1,
                "test_macro_f1": test_f1,
                "test_minority_class_f1": minority_f1,
                "train_time_seconds": train_time,
            }
        )
        mlflow.set_tags({"model_type": "small-transformer-from-scratch", "sandbox_demo": "true"})

        variant_dir = out_dir / variant["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), variant_dir / "model.pt")
        with open(variant_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(variant_dir / "config.json", "w") as f:
            json.dump(
                {**variant, "label2id": label2id, "id2label": {v: k for k, v in id2label.items()} if False else id2label},
                f,
                indent=2,
            )
        mlflow.log_artifacts(str(variant_dir), artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        return {
            "name": variant["name"],
            "run_id": run_id,
            "val_macro_f1": best_val_f1,
            "test_macro_f1": test_f1,
            "test_minority_class_f1": minority_f1,
            "train_time_seconds": train_time,
            "artifact_dir": str(variant_dir),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train.csv")
    p.add_argument("--val", default="data/processed/val.csv")
    p.add_argument("--test", default="data/processed/test.csv")
    p.add_argument("--train-subsample", type=int, default=7000, help="CPU sandbox speed cap")
    p.add_argument("--val-subsample", type=int, default=1500)
    p.add_argument("--test-subsample", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default="models/candidates")
    p.add_argument("--summary", default="reports/finetune_runs_summary.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)

    if args.train_subsample and len(train_df) > args.train_subsample:
        train_df = train_df.groupby(LABEL_COL, group_keys=False).apply(
            lambda g: g.sample(frac=min(1.0, args.train_subsample / len(train_df)), random_state=args.seed)
        ).reset_index(drop=True)
    if args.val_subsample and len(val_df) > args.val_subsample:
        val_df = val_df.sample(n=args.val_subsample, random_state=args.seed).reset_index(drop=True)
    if args.test_subsample and len(test_df) > args.test_subsample:
        test_df = test_df.sample(n=args.test_subsample, random_state=args.seed).reset_index(drop=True)

    print(f"Sandbox training scale: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"(subsampled from the full processed split for CPU-only tractability)")

    label2id = {label: i for i, label in enumerate(CANONICAL_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    variants = [
        dict(name="v1_baseline_lr1e-3", lr=1e-3, weighting="none", max_len=96, d_model=96, nhead=4, num_layers=2),
        dict(name="v2_weighted_inv_freq", lr=1e-3, weighting="inverse_freq", max_len=96, d_model=96, nhead=4, num_layers=2),
        dict(name="v3_weighted_lower_lr", lr=3e-4, weighting="inverse_freq", max_len=96, d_model=96, nhead=4, num_layers=2),
        dict(name="v4_weighted_short_seq", lr=1e-3, weighting="inverse_freq", max_len=64, d_model=96, nhead=4, num_layers=2),
        dict(name="v5_weighted_bigger_model", lr=1e-3, weighting="inverse_freq", max_len=96, d_model=128, nhead=8, num_layers=3),
        dict(name="v6_weighted_inv_sqrt_freq", lr=1e-3, weighting="inverse_sqrt_freq", max_len=96, d_model=96, nhead=4, num_layers=2),
    ]

    mlflow.set_experiment(EXPERIMENT_NAME)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for variant in variants:
        print(f"\n=== Training variant: {variant['name']} ===")
        result = train_one_variant(variant, train_df, val_df, test_df, label2id, id2label, device, args.epochs, args.batch_size, out_dir)
        results.append(result)

    results_sorted = sorted(results, key=lambda r: r["val_macro_f1"], reverse=True)
    best = results_sorted[0]

    print("\n=== Variant comparison (sorted by val macro-F1) ===")
    for r in results_sorted:
        print(f"  {r['name']:28s} val_f1={r['val_macro_f1']:.4f}  test_f1={r['test_macro_f1']:.4f}  "
              f"minority_f1={r['test_minority_class_f1']:.4f}  time={r['train_time_seconds']:.1f}s")
    print(f"\nBest variant: {best['name']} (run_id={best['run_id']})")

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump({"results": results_sorted, "best": best, "label2id": label2id}, f, indent=2)
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
