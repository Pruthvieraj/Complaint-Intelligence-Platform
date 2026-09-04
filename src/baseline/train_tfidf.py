"""
Phase 0 — TF-IDF + Logistic Regression baseline.

Not throwaway code: this is the floor number every later claim (fine-
tuned transformer, promotion gate, benchmark) gets compared against.
Trained with class_weight="balanced" so the baseline itself isn't
trivially gamed by predicting the majority class — the model has to
beat a baseline that's already imbalance-aware.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from src.data.preprocess import LABEL_COL, NARRATIVE_COL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train.csv")
    p.add_argument("--val", default="data/processed/val.csv")
    p.add_argument("--test", default="data/processed/test.csv")
    p.add_argument("--out-dir", default="models/baseline")
    p.add_argument("--report", default="reports/baseline_report.json")
    args = p.parse_args()

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    test = pd.read_csv(args.test)

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train[NARRATIVE_COL])
    X_val = vectorizer.transform(val[NARRATIVE_COL])
    X_test = vectorizer.transform(test[NARRATIVE_COL])

    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
    clf.fit(X_train, train[LABEL_COL])
    train_time = time.time() - t0

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_macro_f1 = f1_score(val[LABEL_COL], val_pred, average="macro")
    test_macro_f1 = f1_score(test[LABEL_COL], test_pred, average="macro")
    test_report = classification_report(test[LABEL_COL], test_pred, output_dict=True, zero_division=0)

    print(f"Trained in {train_time:.1f}s")
    print(f"Val macro-F1:  {val_macro_f1:.4f}")
    print(f"Test macro-F1: {test_macro_f1:.4f}")
    print(classification_report(test[LABEL_COL], test_pred, zero_division=0))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(clf, out_dir / "logreg.joblib")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(
            {
                "model": "TF-IDF (1-2 gram, 50k vocab) + LogisticRegression(class_weight=balanced)",
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                "train_time_seconds": round(train_time, 2),
                "val_macro_f1": round(val_macro_f1, 4),
                "test_macro_f1": round(test_macro_f1, 4),
                "per_class_f1": {
                    k: round(v["f1-score"], 4)
                    for k, v in test_report.items()
                    if k not in ("accuracy", "macro avg", "weighted avg")
                },
            },
            f,
            indent=2,
        )
    print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    main()
