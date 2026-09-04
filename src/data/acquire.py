"""
Phase 0 — Data acquisition.

Production path: pull the real CFPB Consumer Complaint Database via the
public Socrata API (no key required, no cost). This is the "real
regulatory data" the project scope calls for.

Sandbox fallback: the environment this repo was originally built in has
network egress restricted to PyPI/GitHub only (no access to
files.consumerfinance.gov or data.consumerfinance.gov). Rather than fake
a clean run, this script tries the real endpoint first, and only falls
back to a schema-accurate synthetic generator (src/data/synthesize.py)
when the real endpoint is unreachable — logging loudly when that happens
so nobody mistakes sandbox output for the real dataset.

Usage:
    python -m src.data.acquire --n 120000 --out data/raw/complaints.csv
"""
import argparse
import io
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("acquire")

# CFPB Socrata dataset id for the Consumer Complaint Database.
CFPB_SOCRATA_BASE = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
CFPB_SOCRATA_CSV = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"


def try_real_download(n_rows: int, timeout: int = 15) -> pd.DataFrame | None:
    """Attempt to pull real CFPB data. Returns None (does not raise) on
    any network failure so callers can fall back cleanly."""
    try:
        log.info("Attempting real CFPB API pull from %s ...", CFPB_SOCRATA_BASE)
        resp = requests.get(
            CFPB_SOCRATA_BASE,
            params={"size": n_rows, "format": "csv"},
            timeout=timeout,
        )
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
        log.info("Real CFPB pull succeeded: %d rows.", len(df))
        return df
    except Exception as exc:  # noqa: BLE001 — deliberately broad, this is a fallback boundary
        log.warning("Real CFPB endpoint unreachable (%s). Falling back to synthetic generator.", exc)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=120_000, help="target row count")
    parser.add_argument("--out", type=Path, default=Path("data/raw/complaints.csv"))
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="skip the real-API attempt entirely (useful in CI / known-offline envs)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = None
    if not args.force_synthetic:
        df = try_real_download(args.n)

    if df is None:
        from src.data.synthesize import generate_dataset

        log.warning(
            "=" * 78
            + "\nSYNTHETIC DATA IN USE — this is NOT the real CFPB Consumer Complaint"
            " Database.\nSee README 'Sandbox execution note' for why, and rerun this"
            " script with\nnetwork access to www.consumerfinance.gov to get the real"
            " thing.\n"
            + "=" * 78
        )
        df = generate_dataset(n_rows=args.n, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("Wrote %d rows to %s", len(df), args.out)


if __name__ == "__main__":
    sys.exit(main())
