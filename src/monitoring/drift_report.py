"""
Phase 6 — scheduled drift monitoring with Evidently AI.

Per Section 6.3 of the project scope, this is a *scheduled* job (run
after a demo session, or on a cron), not a live streaming pipeline —
free-tier infra has no business pretending otherwise. It compares a
rolling window of live prediction request logs (logs/requests.jsonl,
written by src/serving/app.py) against the training-time reference
distribution, and produces an HTML drift report plus a compact JSON
summary for the README/API to link to.

Usage:
    python -m src.monitoring.drift_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def build_reference(train_csv: str, sample_size: int = 2000, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    ref = pd.DataFrame(
        {
            "text_length_chars": df["Consumer complaint narrative"].str.len(),
            "text_length_words": df["Consumer complaint narrative"].str.split().str.len(),
            "predicted_category": df["label"],
        }
    )
    return ref


def load_current(log_path: Path, window: int) -> pd.DataFrame | None:
    if not log_path.exists():
        return None
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    df = pd.DataFrame(rows).tail(window)
    return df[["text_length_chars", "text_length_words", "predicted_category"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train.csv")
    p.add_argument("--log-path", default="logs/requests.jsonl")
    p.add_argument("--window", type=int, default=500, help="how many recent requests to evaluate")
    p.add_argument("--out-html", default="reports/drift_report.html")
    p.add_argument("--out-json", default="reports/drift_summary.json")
    args = p.parse_args()

    reference = build_reference(args.train)
    current = load_current(Path(args.log_path), args.window)

    if current is None or len(current) < 20:
        summary = {
            "status": "insufficient_data",
            "message": (
                f"Fewer than 20 live requests logged at {args.log_path} — "
                "drift can't be assessed yet. Send some traffic to the API "
                "(or run src/demo/gradio_app.py) and rerun this script."
            ),
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(summary["message"])
        return

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    result.save_html(args.out_html)

    result_dict = result.dict()
    summary = {
        "status": "ok",
        "n_reference_rows": len(reference),
        "n_current_rows": len(current),
        "report_html": args.out_html,
    }
    try:
        # Evidently's dict schema — pull the top-level drift verdict if present.
        drift_metrics = result_dict.get("metrics", [])
        summary["raw_metrics_count"] = len(drift_metrics)
    except Exception:
        pass

    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {args.out_html} and {args.out_json}")


if __name__ == "__main__":
    main()
