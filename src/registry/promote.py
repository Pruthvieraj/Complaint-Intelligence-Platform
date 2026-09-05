"""
Phase 3 — Model registry + automated promotion gate.

The promotion criterion (the actual "decision with evidence attached"
the project scope asks for):

    A candidate is promoted to 'production' only if its macro-F1 on the
    held-out test set exceeds the current production version's macro-F1
    by >= PROMOTION_MARGIN, AND its minority-class F1 does not regress
    versus the current production version.

`decide_promotion()` is the pure function both this CLI and the CI
pipeline (Phase 5) call — same code path either way, which is exactly
what makes the CI gate trustworthy instead of a rubber stamp.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow import MlflowClient

MODEL_NAME = "complaint-classifier"
PRODUCTION_ALIAS = "production"
PROMOTION_MARGIN = 0.01  # candidate macro-F1 must exceed production by >= 1 point
MINORITY_F1_TOLERANCE = 0.0  # candidate minority-class F1 must not regress at all
PROMOTION_LOG = Path("reports/promotion_log.json")
PRODUCTION_METRICS_SNAPSHOT = Path("reports/production_model_metrics.json")


def get_minority_labels(train_csv: str = "data/processed/train.csv", n: int = 3) -> list[str]:
    """The n least-frequent canonical labels in the training split — used
    both to select which classes count as 'minority' when logging metrics
    and, later, by CI's regression check (ci_check.py)."""
    import pandas as pd

    from src.data.preprocess import LABEL_COL

    counts = pd.read_csv(train_csv)[LABEL_COL].value_counts()
    return counts.sort_values().head(n).index.tolist()


def get_run_metrics(run_id: str) -> dict:
    import mlflow

    run = mlflow.get_run(run_id)
    return dict(run.data.metrics)


def ensure_registered_model(client: "MlflowClient"):
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        client.create_registered_model(
            MODEL_NAME, description="CFPB complaint -> product/issue category classifier"
        )


def register_candidate(client: "MlflowClient", run_id: str) -> str:
    ensure_registered_model(client)
    model_uri = f"runs:/{run_id}/model"
    mv = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=run_id)
    return mv.version


def get_production(client: "MlflowClient"):
    """Returns (version, metrics) for the current production alias, or
    (None, None) if nothing has been promoted yet."""
    from mlflow.exceptions import MlflowException

    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except MlflowException:
        return None, None
    return mv.version, get_run_metrics(mv.run_id)


def decide_promotion(candidate_metrics: dict, production_metrics: dict | None) -> tuple[bool, str]:
    """Pure decision function — no MLflow I/O — so it's trivially unit
    testable (tests/test_promotion.py) and callable identically from CI."""
    cand_f1 = candidate_metrics.get("test_macro_f1")
    if cand_f1 is None:
        return False, "candidate run has no test_macro_f1 metric logged"

    if production_metrics is None:
        return True, (
            f"No existing production version to compare against — bootstrap promotion "
            f"of candidate (test_macro_f1={cand_f1:.4f})."
        )

    prod_f1 = production_metrics.get("test_macro_f1", 0.0)
    cand_min = candidate_metrics.get("test_minority_class_f1", 0.0)
    prod_min = production_metrics.get("test_minority_class_f1", 0.0)

    if cand_f1 < prod_f1 + PROMOTION_MARGIN:
        return False, (
            f"REJECTED: candidate macro-F1 {cand_f1:.4f} does not exceed production "
            f"macro-F1 {prod_f1:.4f} by the required margin of {PROMOTION_MARGIN} "
            f"(needed >= {prod_f1 + PROMOTION_MARGIN:.4f})."
        )

    if cand_min < prod_min - MINORITY_F1_TOLERANCE:
        return False, (
            f"REJECTED: candidate minority-class F1 {cand_min:.4f} regresses versus "
            f"production minority-class F1 {prod_min:.4f} (tolerance {MINORITY_F1_TOLERANCE})."
        )

    return True, (
        f"PROMOTED: candidate macro-F1 {cand_f1:.4f} beats production {prod_f1:.4f} "
        f"by {cand_f1 - prod_f1:.4f} (>= margin {PROMOTION_MARGIN}); minority-class F1 "
        f"{cand_min:.4f} does not regress vs production {prod_min:.4f}."
    )


def _append_log(entry: dict):
    PROMOTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(PROMOTION_LOG.read_text()) if PROMOTION_LOG.exists() else []
    log.append(entry)
    PROMOTION_LOG.write_text(json.dumps(log, indent=2))


def register_and_check(run_id: str, run_name: str | None = None, dry_run: bool = False) -> bool:
    from mlflow import MlflowClient

    client = MlflowClient()
    version = register_candidate(client, run_id)
    candidate_metrics = get_run_metrics(run_id)
    prod_version, prod_metrics = get_production(client)

    promoted, reason = decide_promotion(candidate_metrics, prod_metrics)

    print(f"Candidate: run_id={run_id} -> registered as {MODEL_NAME} v{version} ({run_name or ''})")
    print(f"Current production: v{prod_version}" if prod_version else "Current production: none")
    print(reason)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_name": run_name,
        "candidate_version": version,
        "candidate_test_macro_f1": candidate_metrics.get("test_macro_f1"),
        "candidate_test_minority_f1": candidate_metrics.get("test_minority_class_f1"),
        "previous_production_version": prod_version,
        "previous_production_test_macro_f1": (prod_metrics or {}).get("test_macro_f1"),
        "decision": "promoted" if promoted else "rejected",
        "reason": reason,
    }
    if not dry_run:
        _append_log(entry)
        if promoted:
            client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)
            print(f"-> Set alias '{PRODUCTION_ALIAS}' to v{version}.")

            snapshot = {
                "model_version": version,
                "run_id": run_id,
                "run_name": run_name,
                "test_macro_f1": candidate_metrics.get("test_macro_f1"),
                "test_minority_class_f1": candidate_metrics.get("test_minority_class_f1"),
                "minority_labels": get_minority_labels(),
                "promoted_at": entry["timestamp"],
            }
            PRODUCTION_METRICS_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            PRODUCTION_METRICS_SNAPSHOT.write_text(json.dumps(snapshot, indent=2))
            print(f"-> Wrote production metrics snapshot to {PRODUCTION_METRICS_SNAPSHOT}")
    return promoted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--dry-run", action="store_true", help="evaluate the decision but don't register/promote")
    args = p.parse_args()
    promoted = register_and_check(args.run_id, args.run_name, args.dry_run)
    raise SystemExit(0 if promoted else 1)


if __name__ == "__main__":
    main()
