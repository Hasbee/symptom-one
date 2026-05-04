"""Shared helpers for ML training data cleaning scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def coerce_binary(value: object) -> float:
    if pd.isna(value) or value is None:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(int(value != 0))
    s = str(value).strip().lower()
    if s in ("true", "yes", "1", "y"):
        return 1.0
    if s in ("false", "no", "0", "n", ""):
        return 0.0
    try:
        v = float(s)
        return float(int(v != 0))
    except ValueError:
        return np.nan


def read_training_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"Missing input file: {path}", file=sys.stderr)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Failed to read CSV: {exc}", file=sys.stderr)
        return None


def validate_target_column(df: pd.DataFrame, target_col: str) -> bool:
    if target_col not in df.columns:
        print(f"Missing column {target_col!r}", file=sys.stderr)
        return False
    return True


def apply_binary_column_cleaning(df: pd.DataFrame, bool_cols: tuple[str, ...]) -> None:
    for col in bool_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].map(coerce_binary)
        med = df[col].median()
        fill = 0.0 if pd.isna(med) else float(med)
        df[col] = df[col].fillna(fill)
