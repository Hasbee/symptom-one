"""Data cleaning script for severity training data."""

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

TRAINING_CSV = PROJECT_ROOT / "data" / "training_dataset" / "severity_training.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "severity_cleaned.csv"

BOOL_COLS = ("fever_present", "neck_stiffness")
TEXT_COLS = ("symptom_type", "symptom_descriptor", "associated_symptoms")
TARGET_COL = "severity_level"
VALID_SEVERITY = frozenset({"Low", "Moderate", "High", "Critical"})
SEVERITY_LOWER_MAP = {
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
    "critical": "Critical",
}
RNG_SEED = 42


def _normalize_severity(value: object) -> object:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s in VALID_SEVERITY:
        return s
    mapped = SEVERITY_LOWER_MAP.get(s.lower())
    return mapped if mapped is not None else np.nan


def _downsample_per_class(df: pd.DataFrame, target_col: str, n_per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for label in sorted(df[target_col].unique()):
        sub = df[df[target_col] == label]
        if len(sub) <= n_per_class:
            parts.append(sub)
            continue
        idx = rng.choice(len(sub), size=n_per_class, replace=False)
        parts.append(sub.iloc[idx])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)


def _prepare_severity_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    n_raw = len(df)
    out = df.copy()
    out[TARGET_COL] = out[TARGET_COL].map(_normalize_severity)
    out = out.dropna(subset=[TARGET_COL])
    return out, n_raw, len(out)


def _report_and_get_min_class_count(df: pd.DataFrame) -> int | None:
    class_counts_before = df[TARGET_COL].value_counts().sort_index()
    print("Class distribution (after label filter, before balance):")
    print(class_counts_before.to_string())
    n_min = int(class_counts_before.min())
    if n_min == 0:
        print("Empty severity class after cleaning.", file=sys.stderr)
        return None
    return n_min


def _apply_text_column_defaults(df: pd.DataFrame) -> None:
    text_fill = {
        "symptom_type": "",
        "symptom_descriptor": "",
        "associated_symptoms": "",
    }
    for col, placeholder in text_fill.items():
        if col not in df.columns:
            continue
        df[col] = df[col].replace("", np.nan).fillna(placeholder).astype(str)


def _numeric_columns_excluding_target_and_text(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in TEXT_COLS
        and c != TARGET_COL
        and c not in BOOL_COLS
    ]


def _apply_numeric_coercion(df: pd.DataFrame) -> None:
    for col in _numeric_columns_excluding_target_and_text(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col == "risk_score":
            df[col] = df[col].clip(lower=0.0, upper=1.0)
        med = df[col].median()
        df[col] = df[col].fillna(0.0 if pd.isna(med) else med)


def _mask_empty_text_columns(df: pd.DataFrame) -> None:
    for col in TEXT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].mask(df[col].astype(str).str.strip() == "", " ")


def _write_balanced_csv_and_print_summary(
    df_bal: pd.DataFrame, n_raw: int, n_after_label: int, n_min: int
) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)

    n_balanced = len(df_bal)
    class_counts_after = df_bal[TARGET_COL].value_counts().sort_index()
    print(
        f"Rows: {n_raw} raw -> {n_after_label} valid label -> "
        f"{n_balanced} balanced (per-class n={n_min})"
    )
    print("Class distribution (after balance):")
    print(class_counts_after.to_string())
    print(f"Wrote {OUTPUT_CSV}")


def clean_severity_data() -> bool:
    """Load severity training CSV, clean, balance classes, write processed file."""
    df = read_training_csv(TRAINING_CSV)
    if df is None:
        return False
    if not validate_target_column(df, TARGET_COL):
        return False

    df, n_raw, n_after_label = _prepare_severity_labels(df)
    if n_after_label == 0:
        print("No rows with valid severity_level.", file=sys.stderr)
        return False

    n_min = _report_and_get_min_class_count(df)
    if n_min is None:
        return False

    apply_binary_column_cleaning(df, BOOL_COLS)
    _apply_text_column_defaults(df)
    _apply_numeric_coercion(df)

    rng = np.random.default_rng(RNG_SEED)
    df_bal = _downsample_per_class(df, TARGET_COL, n_min, rng)

    _mask_empty_text_columns(df_bal)
    _write_balanced_csv_and_print_summary(df_bal, n_raw, n_after_label, n_min)
    return True


if __name__ == "__main__":
    ok = clean_severity_data()
    sys.exit(0 if ok else 1)
