"""
src/data_loader.py — Load, validate, and profile an energy consumption dataset.

This module is responsible for reading CSV files (uploaded or from disk),
checking data quality, and returning a clean, profiled DataFrame.

Key design decisions
--------------------
* Timestamp parsing uses pandas' flexible parser; we do NOT assume a fixed
  date format, which keeps the loader compatible with most real-world exports.
* Column selection is kept separate from loading so the UI can let the user
  pick columns after seeing what the file contains.
* Validation checks are returned as structured messages rather than raised
  exceptions so the Streamlit UI can display them gracefully.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    """Summary statistics and quality report for a loaded dataset."""

    filename: str
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    date_range_start: Optional[pd.Timestamp]
    date_range_end: Optional[pd.Timestamp]
    missing_counts: dict[str, int]
    duplicate_rows: int
    inferred_freq: Optional[str]          # e.g. "1h", "15min"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def total_missing(self) -> int:
        return sum(self.missing_counts.values())

    @property
    def duration_days(self) -> Optional[float]:
        if self.date_range_start and self.date_range_end:
            return (self.date_range_end - self.date_range_start).total_seconds() / 86400
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_parse_timestamps(series: pd.Series) -> pd.Series:
    """Attempt to parse a string column as datetime using pandas' flexible parser."""
    return pd.to_datetime(series, errors="coerce")


def _infer_sampling_frequency(dt_series: pd.Series) -> Optional[str]:
    """
    Estimate the most common time-step in a datetime series.

    Returns a pandas frequency string (e.g. "1min", "15min", "1h", "1D")
    or None if it cannot be determined.
    """
    if len(dt_series) < 3:
        return None

    # Compute consecutive differences, ignore NaT
    deltas = dt_series.dropna().sort_values().diff().dropna()
    if deltas.empty:
        return None

    # Use the median to resist outliers (e.g. daylight-saving gaps)
    median_delta = deltas.median()
    total_seconds = median_delta.total_seconds()

    if total_seconds <= 0:
        return None

    freq_map = [
        (60,     "1min"),
        (300,    "5min"),
        (900,    "15min"),
        (1800,   "30min"),
        (3600,   "1h"),
        (86400,  "1D"),
        (604800, "1W"),
    ]
    # Return the closest standard frequency
    for seconds, label in freq_map:
        if abs(total_seconds - seconds) / seconds < 0.10:  # within 10 %
            return label

    # Generic fallback
    minutes = round(total_seconds / 60)
    return f"{minutes}min"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_from_path(filepath: str) -> tuple[pd.DataFrame, DataProfile]:
    """
    Load a CSV dataset from a file path on disk.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
        Raw DataFrame exactly as read from disk (no column selection yet).
    profile : DataProfile
        Data quality summary.
    """
    if not os.path.isfile(filepath):
        profile = DataProfile(
            filename=os.path.basename(filepath),
            n_rows=0, n_cols=0, columns=[], dtypes={},
            date_range_start=None, date_range_end=None,
            missing_counts={}, duplicate_rows=0, inferred_freq=None,
            errors=[f"File not found: {filepath}"],
        )
        return pd.DataFrame(), profile

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as exc:
        profile = DataProfile(
            filename=os.path.basename(filepath),
            n_rows=0, n_cols=0, columns=[], dtypes={},
            date_range_start=None, date_range_end=None,
            missing_counts={}, duplicate_rows=0, inferred_freq=None,
            errors=[f"Failed to read CSV: {exc}"],
        )
        return pd.DataFrame(), profile

    return _profile_dataframe(df, os.path.basename(filepath))


def load_from_upload(uploaded_file) -> tuple[pd.DataFrame, DataProfile]:
    """
    Load a CSV dataset from a Streamlit UploadedFile object.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        File object returned by st.file_uploader.

    Returns
    -------
    Same as load_from_path.
    """
    try:
        content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:
        profile = DataProfile(
            filename=getattr(uploaded_file, "name", "uploaded_file"),
            n_rows=0, n_cols=0, columns=[], dtypes={},
            date_range_start=None, date_range_end=None,
            missing_counts={}, duplicate_rows=0, inferred_freq=None,
            errors=[f"Failed to read uploaded file: {exc}"],
        )
        return pd.DataFrame(), profile

    return _profile_dataframe(df, getattr(uploaded_file, "name", "uploaded_file"))


def select_columns(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply column selection: parse the timestamp column and return a
    clean working DataFrame with the target column validated.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_from_path / load_from_upload.
    timestamp_col : str
        Column to use as the datetime index.
    target_col : str
        Column to predict (must be numeric).

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame with a DatetimeIndex and validated target column.
    errors : list[str]
        List of blocking errors (empty if all is well).
    """
    errors: list[str] = []

    if timestamp_col not in df.columns:
        errors.append(f"Timestamp column '{timestamp_col}' not found in dataset.")
        return df, errors

    if target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found in dataset.")
        return df, errors

    # Parse timestamp
    dt_series = _try_parse_timestamps(df[timestamp_col])
    n_failed = dt_series.isna().sum()
    if n_failed == len(df):
        errors.append(
            f"Could not parse any timestamps in column '{timestamp_col}'. "
            "Check the column name and date format."
        )
        return df, errors
    if n_failed > 0:
        errors.append(
            f"{n_failed} timestamps in '{timestamp_col}' could not be parsed "
            "and will be dropped."
        )

    # Parse target
    target_series = pd.to_numeric(df[target_col], errors="coerce")
    n_non_numeric = target_series.isna().sum() - df[target_col].isna().sum()
    if n_non_numeric > 0:
        errors.append(
            f"{n_non_numeric} non-numeric values found in '{target_col}' "
            "and will be treated as missing."
        )

    df_out = df.copy()
    df_out[timestamp_col] = dt_series
    df_out[target_col] = target_series
    df_out = df_out.dropna(subset=[timestamp_col])
    df_out = df_out.set_index(timestamp_col).sort_index()

    return df_out, errors


# ---------------------------------------------------------------------------
# Internal profiling logic
# ---------------------------------------------------------------------------

def _profile_dataframe(df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, DataProfile]:
    """Build a DataProfile for a freshly loaded DataFrame."""
    warnings: list[str] = []
    errors: list[str] = []

    if df.empty:
        errors.append("The CSV file is empty.")
        return df, DataProfile(
            filename=filename, n_rows=0, n_cols=0, columns=[], dtypes={},
            date_range_start=None, date_range_end=None,
            missing_counts={}, duplicate_rows=0, inferred_freq=None,
            errors=errors,
        )

    # Detect a plausible timestamp column for the profile
    dt_col = _guess_timestamp_column(df)
    dt_series: Optional[pd.Series] = None
    date_start = date_end = inferred_freq = None

    if dt_col:
        dt_series = _try_parse_timestamps(df[dt_col])
        valid_dt = dt_series.dropna()
        if not valid_dt.empty:
            date_start = valid_dt.min()
            date_end = valid_dt.max()
            inferred_freq = _infer_sampling_frequency(valid_dt)

    # Missing values
    missing_counts = df.isna().sum().to_dict()

    # Duplicates (row-level)
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        warnings.append(f"{duplicate_rows} duplicate rows detected.")

    # Missing value warnings
    for col, count in missing_counts.items():
        pct = count / len(df) * 100
        if pct > 20:
            warnings.append(f"Column '{col}' has {count} missing values ({pct:.1f} %).")
        elif pct > 5:
            warnings.append(f"Column '{col}' has {count} missing values ({pct:.1f} %).")

    # Size warning
    if len(df) < 100:
        warnings.append(
            "Dataset has fewer than 100 rows.  Model training may be unreliable."
        )

    profile = DataProfile(
        filename=filename,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=list(df.columns),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        date_range_start=date_start,
        date_range_end=date_end,
        missing_counts=missing_counts,
        duplicate_rows=duplicate_rows,
        inferred_freq=inferred_freq,
        warnings=warnings,
        errors=errors,
    )
    return df, profile


def _guess_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically identify the timestamp column without committing to one.

    Looks for columns whose dtype is object/string and whose name contains
    common time-related keywords.
    """
    keywords = ["date", "time", "datetime", "timestamp", "dt", "period"]
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col
    # Fallback: first column
    return df.columns[0] if len(df.columns) > 0 else None
