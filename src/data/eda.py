"""
Phase 0 — EDA: class balance, text length distribution, label-noise
discussion. Produces reports/eda_summary.md + two figures, all derived
from the actual committed train split (not hand-picked numbers).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocess import LABEL_COL, NARRATIVE_COL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train.csv")
    p.add_argument("--out-dir", default="reports")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train)
    counts = df[LABEL_COL].value_counts().sort_values(ascending=True)
    imbalance_ratio = counts.max() / counts.min()

    df["_wordcount"] = df[NARRATIVE_COL].str.split().str.len()

    # --- Figure 1: class balance ---
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="barh", ax=ax, color="#2b5876")
    ax.set_xlabel("Training examples")
    ax.set_title(f"Class balance (train split, n={len(df)}) — {imbalance_ratio:.1f}x imbalance ratio")
    fig.tight_layout()
    fig.savefig(fig_dir / "class_balance.png", dpi=130)
    plt.close(fig)

    # --- Figure 2: narrative length distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    df["_wordcount"].clip(upper=200).hist(bins=40, ax=ax, color="#c17817")
    ax.set_xlabel("Words per complaint narrative (clipped at 200)")
    ax.set_ylabel("Count")
    ax.set_title("Narrative length distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "text_length.png", dpi=130)
    plt.close(fig)

    md = f"""# EDA Summary — Phase 0

Generated from `data/processed/train.csv` (n={len(df)}).

## Class balance

| Category | Train count | Share |
|---|---:|---:|
"""
    for label, n in counts.sort_values(ascending=False).items():
        md += f"| {label} | {n} | {n / len(df):.1%} |\n"

    md += f"""
Imbalance ratio (largest / smallest class): **{imbalance_ratio:.1f}x**.

This is the collapsed 8-category label set (see `src/data/preprocess.py::LABEL_MAP`
for the full rationale). The raw CFPB `Product` field has 20+ overlapping
historical variants from taxonomy renames across years — those were
merged down to 8 categories, never across substantively different
products, so that every class still has enough examples to be learnable
(smallest class here is {counts.min()} rows, ~{counts.min() / len(df):.1%} of
train) while the total label space stays tractable for a portfolio
timeline. Credit reporting complaints dominate the real CFPB dataset
too — this mirrors that skew rather than flattening it, which is why
Phase 2 uses class-weighted loss instead of a naive fine-tune.

## Narrative length

- Median words/complaint: {int(df['_wordcount'].median())}
- 5th/95th percentile: {int(df['_wordcount'].quantile(0.05))} / {int(df['_wordcount'].quantile(0.95))}

## Label noise

Complaints go through `src/data/preprocess.py::collapse_labels`, which
maps the raw multi-year `Product` taxonomy onto the 8 canonical
categories. A portion of rows carry a raw `Product` value that's
inconsistent with the complaint content (CFPB's taxonomy has genuinely
been applied inconsistently across years — see project scope Section
4.1). This is not cleaned away before training: the point of the
class-weighted loss in Phase 2 and the macro-F1-driven promotion
criterion in Phase 3 is to be robust to exactly this kind of real-world
label noise, not to train against an artificially clean set.

![Class balance](figures/class_balance.png)
![Text length](figures/text_length.png)
"""
    (out_dir / "eda_summary.md").write_text(md)
    print(f"Wrote {out_dir / 'eda_summary.md'} and 2 figures to {fig_dir}")
    print(f"Imbalance ratio: {imbalance_ratio:.1f}x")


if __name__ == "__main__":
    main()
