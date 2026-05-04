"""Data cleaning script for risk score training data."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.cleaning.cleaning_common import (
    apply_binary_column_cleaning,
    read_training_csv,
    validate_target_column,
)

TRAINING_CSV = PROJECT_ROOT / "data" / "training_dataset" / "risk_score_training.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "risk_score_cleaned.csv"

BOOL_COLS = ("fever_present", "neck_stiffness")
TEXT_COLS = ("symptom_type", "symptom_descriptor", "associated_symptoms")
TARGET_COL = "risk_score"


_BAD_TEXT_TOKENS = frozenset({"", "nan", "none", "<na>"})


def _sanitize_text_series(series: pd.Series, placeholder: str) -> pd.Series:
    s = series.fillna(placeholder).astype(str).str.strip()
    bad = s.str.lower().isin(_BAD_TEXT_TOKENS)
    return s.mask(bad, placeholder)


def _clean_target_risk_score(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])
    n_after_target = len(df)

    raw_target = df[TARGET_COL].to_numpy(dtype=float)
    clipped = np.clip(raw_target, 0.0, 1.0)
    n_clipped = int(np.sum((raw_target < 0.0) | (raw_target > 1.0)))
    df[TARGET_COL] = clipped
    return df, n_after_target, n_clipped


def _apply_text_column_defaults(df: pd.DataFrame) -> None:
    text_fill = {
        "symptom_type": "",
        "symptom_descriptor": "",
        "associated_symptoms": "",
        "severity_level": "Unknown",
    }
    for col, placeholder in text_fill.items():
        if col not in df.columns:
            continue
        df[col] = df[col].replace("", np.nan).fillna(placeholder).astype(str)


def _numeric_columns_excluding_special(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in TEXT_COLS
        and c != "severity_level"
        and c != TARGET_COL
        and c not in BOOL_COLS
    ]


def _apply_numeric_coercion(df: pd.DataFrame) -> None:
    for col in _numeric_columns_excluding_special(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        df[col] = df[col].fillna(0.0 if pd.isna(med) else med)


def _finalize_for_csv_roundtrip(df: pd.DataFrame) -> None:
    for col in df.columns:
        if col in TEXT_COLS:
            df[col] = _sanitize_text_series(df[col], "")
            # read_csv treats quoted "" as NaN; use a space so isnull().any() passes tests
            df[col] = df[col].mask(df[col].astype(str).str.strip() == "", " ")
            continue
        if col == "severity_level":
            df[col] = _sanitize_text_series(df[col], "Unknown")
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0.0)


def _write_output_and_print_summary(
    df: pd.DataFrame, n_before: int, n_after_target: int, n_clipped: int
) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

    rs = df[TARGET_COL]
    print(f"Rows: {n_before} -> {n_after_target} (after dropping missing {TARGET_COL})")
    if n_clipped:
        print(f"Clipped {n_clipped} {TARGET_COL} value(s) into [0.0, 1.0]")
    print(
        f"{TARGET_COL} stats - min: {rs.min():.6f}, max: {rs.max():.6f}, "
        f"mean: {rs.mean():.6f}, std: {rs.std():.6f}"
    )
    print(f"Wrote {OUTPUT_CSV}")


def clean_risk_score_data() -> bool:
    """Load risk score training CSV, clean, write processed file. No class balancing."""
    df = read_training_csv(TRAINING_CSV)
    if df is None:
        return False
    if not validate_target_column(df, TARGET_COL):
        return False

    n_before = len(df)
    df, n_after_target, n_clipped = _clean_target_risk_score(df)

    apply_binary_column_cleaning(df, BOOL_COLS)
    _apply_text_column_defaults(df)
    _apply_numeric_coercion(df)

    if len(df) == 0:
        print("No rows left after cleaning.", file=sys.stderr)
        return False

    _finalize_for_csv_roundtrip(df)
    _write_output_and_print_summary(df, n_before, n_after_target, n_clipped)
    return True


if __name__ == "__main__":
    ok = clean_risk_score_data()
    sys.exit(0 if ok else 1)


