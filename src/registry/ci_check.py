"""
Phase 5 — CI regression check.

Free GitHub Actions runners don't have the compute or time budget for a
full fine-tuning run on every push (Section 6.2), and they don't have
network access to a self-hosted MLflow instance either. So this does
NOT retrain and does NOT talk to MLflow. It re-evaluates the exact
committed production artifact (models/onnx/model_int8.onnx — the same
file the API and demo serve) against a small, fixed, committed eval
subset (data/ci_eval_subset.csv), and checks the result against the
metrics recorded at promotion time (reports/production_model_metrics.json,
also committed). That's the whole point of a CI gate: catch the case
where a code change (a tokenization tweak, a preprocessing bug, a
dependency bump) silently breaks the deployed model, even with zero new
training.

A small REGRESSION_TOLERANCE is applied because re-evaluating on a
~300-row fixed subset carries more sampling noise than the full test
set the promotion decision itself was based on — the goal here is
"catch a real regression", not "catch subset-sampling variance".

Exit code 0 = pass, 1 = fail (blocks the PR in ci.yml).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from src.data.preprocess import LABEL_COL, NARRATIVE_COL
from src.serving.inference import ComplaintClassifier

REGRESSION_TOLERANCE = 0.05  # allowed drop vs. recorded production metrics, subset-noise budget


def reevaluate(classifier: ComplaintClassifier, eval_df: pd.DataFrame) -> dict:
    preds = [classifier.predict(t)["predicted_category"] for t in eval_df[NARRATIVE_COL]]
    labels = eval_df[LABEL_COL].tolist()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"macro_f1": macro_f1, "preds": preds, "labels": labels}


def minority_f1(labels, preds, minority_labels) -> float:
    per_class = f1_score(
        labels, preds, average=None, zero_division=0, labels=sorted(set(labels) | set(preds))
    )
    class_order = sorted(set(labels) | set(preds))
    idx = {c: i for i, c in enumerate(class_order)}
    present = [m for m in minority_labels if m in idx]
    if not present:
        return 0.0
    return float(sum(per_class[idx[m]] for m in present) / len(present))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-csv", default="data/ci_eval_subset.csv")
    p.add_argument("--onnx-dir", default="models/onnx")
    p.add_argument("--reference", default="reports/production_model_metrics.json")
    p.add_argument("--report", default="reports/ci_check_result.json")
    args = p.parse_args()

    eval_path, ref_path = Path(args.eval_csv), Path(args.reference)
    if not eval_path.exists() or not ref_path.exists():
        print(f"SKIP: missing {eval_path} or {ref_path} — nothing promoted yet to check.")
        sys.exit(0)

    eval_df = pd.read_csv(eval_path)
    reference = json.loads(ref_path.read_text())

    classifier = ComplaintClassifier(onnx_dir=args.onnx_dir)
    result = reevaluate(classifier, eval_df)
    cand_minority_f1 = minority_f1(result["labels"], result["preds"], reference.get("minority_labels", []))

    ref_macro_f1 = reference["test_macro_f1"]
    ref_minority_f1 = reference.get("test_minority_class_f1", 0.0)

    macro_ok = result["macro_f1"] >= ref_macro_f1 - REGRESSION_TOLERANCE
    minority_ok = cand_minority_f1 >= ref_minority_f1 - REGRESSION_TOLERANCE

    passed = macro_ok and minority_ok
    report = {
        "eval_subset_rows": len(eval_df),
        "reevaluated_macro_f1": round(result["macro_f1"], 4),
        "reference_macro_f1": round(ref_macro_f1, 4),
        "reevaluated_minority_f1": round(cand_minority_f1, 4),
        "reference_minority_f1": round(ref_minority_f1, 4),
        "tolerance": REGRESSION_TOLERANCE,
        "macro_f1_check": "PASS" if macro_ok else "FAIL",
        "minority_f1_check": "PASS" if minority_ok else "FAIL",
        "overall": "PASS" if passed else "FAIL",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    if not passed:
        print("\nCI CHECK FAILED: production model re-evaluation regressed beyond tolerance.")
        sys.exit(1)
    print("\nCI CHECK PASSED.")


if __name__ == "__main__":
    main()
