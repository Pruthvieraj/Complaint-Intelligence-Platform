"""Unit tests for the data pipeline (Phase 0 / Phase 5 CI check). Fast,
no network, no large files — synthesizes a tiny in-memory sample."""
import pandas as pd

from src.data.preprocess import (
    LABEL_COL,
    NARRATIVE_COL,
    PRODUCT_COL,
    clean_narrative,
    collapse_labels,
    prepare_dataset,
    stratified_split,
)
from src.data.synthesize import CANONICAL_CATEGORIES, RAW_VARIANTS, generate_dataset


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(n_rows=500, seed=1)
    assert len(df) == 500
    for col in ["Complaint ID", "Date received", "Product", "Sub-product", "Issue",
                "Consumer complaint narrative", "Company", "State"]:
        assert col in df.columns


def test_every_raw_product_variant_maps_to_a_canonical_label():
    for canon, variants in RAW_VARIANTS.items():
        for raw in variants:
            df = pd.DataFrame({PRODUCT_COL: [raw]})
            mapped = collapse_labels(df)
            assert mapped[LABEL_COL].iloc[0] == canon


def test_collapse_labels_drops_unmapped_products():
    df = pd.DataFrame({PRODUCT_COL: ["Some Future Unknown Product"]})
    out = collapse_labels(df)
    assert len(out) == 0


def test_clean_narrative_collapses_whitespace_and_handles_non_string():
    assert clean_narrative("hello   \n\n world  ") == "hello world"
    assert clean_narrative(None) == ""
    assert clean_narrative(float("nan")) == ""


def test_stratified_split_preserves_class_proportions_roughly():
    df = generate_dataset(n_rows=4000, seed=3)
    df[LABEL_COL] = df[PRODUCT_COL].map(
        {raw: canon for canon, variants in RAW_VARIANTS.items() for raw in variants}
    )
    df = df.dropna(subset=[LABEL_COL])
    train, val, test = stratified_split(df, seed=3)

    assert len(train) + len(val) + len(test) == len(df)
    full_props = df[LABEL_COL].value_counts(normalize=True)
    train_props = train[LABEL_COL].value_counts(normalize=True)
    for label in CANONICAL_CATEGORIES:
        if label in full_props.index and label in train_props.index:
            assert abs(full_props[label] - train_props[label]) < 0.05


def test_prepare_dataset_end_to_end(tmp_path):
    df = generate_dataset(n_rows=2000, seed=7)
    raw_csv = tmp_path / "raw.csv"
    df.to_csv(raw_csv, index=False)

    prepared = prepare_dataset(raw_csv)
    assert LABEL_COL in prepared.columns
    assert prepared[NARRATIVE_COL].str.len().min() >= 15
    assert prepared[NARRATIVE_COL].duplicated().sum() == 0
