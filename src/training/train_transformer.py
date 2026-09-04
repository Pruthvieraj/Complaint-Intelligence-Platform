"""
Phase 2 (production path) — fine-tune a real pretrained transformer
(DistilBERT-base or FinBERT-base) with the HuggingFace Trainer API,
weighted-loss class imbalance handling, and MLflow experiment tracking.

Run this on a machine with internet access to huggingface.co and,
ideally, a GPU (a free Colab T4 is exactly enough — see
notebooks/02_colab_finetune.ipynb for a ready-to-run wrapper). It was
written and reviewed in a sandbox that could not reach the HF Hub
(only PyPI/GitHub were reachable), so it has not been executed there —
src/training/train_sandbox_transformer.py is the script that was
actually run end-to-end, using a from-scratch model, to prove out the
tracking/registry/CI/serving machinery this script plugs into
identically. See README "Sandbox execution note".

Usage:
    python -m src.training.train_transformer --base-model distilbert-base-uncased
    python -m src.training.train_transformer --base-model ProsusAI/finbert
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.preprocess import CANONICAL_LABELS, LABEL_COL, NARRATIVE_COL

EXPERIMENT_NAME = "complaint-classification-finetune"


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to the CE loss so the
    fine-tune doesn't collapse toward the majority class (Credit
    reporting is ~35-40% of the data — see reports/eda_summary.md)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: pd.Series, label2id: dict, strategy: str) -> torch.Tensor | None:
    if strategy == "none":
        return None
    counts = labels.value_counts()
    weights = torch.ones(len(label2id))
    for label, idx in label2id.items():
        n = counts.get(label, 1)
        weights[idx] = 1.0 / n if strategy == "inverse_freq" else 1.0 / np.sqrt(n)
    return weights / weights.sum() * len(label2id)


def build_metrics_fn():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    return compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train.csv")
    p.add_argument("--val", default="data/processed/val.csv")
    p.add_argument("--test", default="data/processed/test.csv")
    p.add_argument("--base-model", default="distilbert-base-uncased",
                    help="or ProsusAI/finbert for a finance-pretrained starting point")
    p.add_argument("--out-dir", default="models/candidates/hf_finetune")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--weighting", default="inverse_freq", choices=["none", "inverse_freq", "inverse_sqrt_freq"])
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    label2id = {label: i for i, label in enumerate(CANONICAL_LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def to_hf_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[[NARRATIVE_COL, LABEL_COL]].rename(columns={NARRATIVE_COL: "text"}))
        ds = ds.map(lambda x: {"label": label2id[x[LABEL_COL]]})
        ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=args.max_length), batched=True)
        return ds

    train_ds, val_ds, test_ds = to_hf_dataset(train_df), to_hf_dataset(val_df), to_hf_dataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    class_weights = compute_class_weights(train_df[LABEL_COL], label2id, args.weighting)

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = args.run_name or f"hf_{args.base_model.split('/')[-1]}_{args.weighting}_lr{args.lr}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "base_model": args.base_model,
                "weighting": args.weighting,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "model_type": "hf-pretrained-finetune",
            }
        )

        training_args = TrainingArguments(
            output_dir=args.out_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            logging_steps=50,
            report_to=[],  # we log to MLflow ourselves for full control
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            compute_metrics=build_metrics_fn(),
            class_weights=class_weights,
        )
        trainer.train()

        val_metrics = trainer.evaluate(val_ds)
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        for k, v in {**val_metrics, **test_metrics}.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        trainer.save_model(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        with open(Path(args.out_dir) / "label_map.json", "w") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
        mlflow.log_artifacts(args.out_dir, artifact_path="model")

        print(f"Val macro-F1:  {val_metrics.get('eval_macro_f1'):.4f}")
        print(f"Test macro-F1: {test_metrics.get('test_macro_f1'):.4f}")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
