"""
Phase 0 — Label collapsing, PII-token handling, and stratified splitting.

CFPB's raw "Product" field has drifted across ~10 years of the dataset:
product names were renamed, merged, and split (e.g. "Credit reporting"
became "Credit reporting, credit repair services, or other personal
consumer reports"; "Bank account or service" became "Checking or
savings account"; "Consumer Loan" was folded into "Vehicle loan or
lease" for auto-specific complaints). Training directly on the raw
field either explodes the label space or silently treats renamed-but-
identical categories as different classes.

LABEL_MAP below is the explicit, documented collapse down to 8 clean,
disjoint product/issue categories. This is the "write down the
rationale" deliverable Phase 0 calls for — the rationale is: collapse
purely on historical renames / near-duplicates, never merge
*substantively different* products (e.g. mortgages stay separate from
vehicle loans even though both are "loans"), and keep every canonical
label big enough to be learnable (smallest class is still ~3% of the
data, not a handful of rows).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP: dict[str, str] = {
    # --- Credit reporting ---
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting or other personal consumer reports",
    "Credit reporting": "Credit reporting or other personal consumer reports",
    # --- Debt collection ---
    "Debt collection": "Debt collection",
    # --- Credit card / prepaid ---
    "Credit card or prepaid card": "Credit card or prepaid card",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    # --- Checking / savings ---
    "Checking or savings account": "Checking or savings account",
    "Bank account or service": "Checking or savings account",
    # --- Mortgage ---
    "Mortgage": "Mortgage",
    # --- Student loan ---
    "Student loan": "Student loan",
    # --- Vehicle loan / lease ---
    "Vehicle loan or lease": "Vehicle loan or lease",
    "Consumer Loan": "Vehicle loan or lease",
    # --- Money transfer / virtual currency ---
    "Money transfer, virtual currency, or money service": "Money transfer, virtual currency, or money service",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
}

CANONICAL_LABELS = sorted(set(LABEL_MAP.values()))

NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"
LABEL_COL = "label"


def collapse_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COL] = df[PRODUCT_COL].map(LABEL_MAP)
    unmapped = df[LABEL_COL].isna().sum()
    if unmapped:
        # Real CFPB pulls occasionally contain product strings outside our
        # map (taxonomy keeps evolving) — drop rather than guess.
        df = df.dropna(subset=[LABEL_COL])
    return df


def clean_narrative(text: str) -> str:
    """Light cleaning only — we deliberately do NOT strip the XXXX PII
    redaction tokens, they're part of the real signal distribution the
    model will see in production (redacted account/date/name mentions
    correlate with complaint type)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataset(raw_csv: Path, min_narrative_len: int = 15) -> pd.DataFrame:
    df = pd.read_csv(raw_csv, low_memory=False)
    before = len(df)

    df = df.dropna(subset=[NARRATIVE_COL, PRODUCT_COL])
    df[NARRATIVE_COL] = df[NARRATIVE_COL].map(clean_narrative)
    df = df[df[NARRATIVE_COL].str.len() >= min_narrative_len]

    df = collapse_labels(df)
    df = df.drop_duplicates(subset=[NARRATIVE_COL])

    after = len(df)
    print(f"prepare_dataset: {before} raw rows -> {after} usable rows "
          f"({before - after} dropped: missing narrative/product, too-short, "
          f"unmapped taxonomy, or exact-duplicate narrative)")
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int = 42):
    """70/15/15 train/val/test, stratified on the collapsed label so
    every split preserves the real class imbalance (that imbalance is
    exactly what Phase 2's weighted-loss handling has to deal with —
    hiding it with a balanced split would be misleading)."""
    train, temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df[LABEL_COL]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp[LABEL_COL]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/complaints.csv")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(Path(args.raw))
    train, val, test = stratified_split(df, seed=args.seed)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print("\nClass distribution (train):")
    print(train[LABEL_COL].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
