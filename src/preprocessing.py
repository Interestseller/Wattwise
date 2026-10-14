"""
src/preprocessing.py — Clean, resample, and split energy consumption data.

What this module does
---------------------
1. Remove duplicate timestamps (average them).
2. Clip physically implausible negative values to NaN.
3. Interpolate short gaps; flag long gaps.
4. Resample to a uniform time step.
5. Split the series chronologically into train / validation / test sets.
   — Crucially, we NEVER shuffle time-series data before splitting.
     Shuffling would leak future information into the training set and
     produce overoptimistic evaluation metrics (data leakage).

Why chronological splitting matters
------------------------------------
A time-series model trained on randomly sampled observations "cheats": it
can memorise future patterns that appear earlier in the shuffled training
set. Chronological splitting ensures the model has never seen the evaluation
period during training, which gives a realistic estimate of forecast accuracy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_series(
    df: pd.DataFrame,
    target_col: str,
    lower_clip: Optional[float] = 0.0,
    upper_clip: Optional[float] = None,
    max_interpolation_gap: int = 6,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean a time-indexed DataFrame.

    Steps
    -----
    1. Resolve duplicate timestamps by taking the mean.
    2. Clip values outside the physically valid range.
    3. Interpolate small gaps (up to ``max_interpolation_gap`` consecutive NaNs).
    4. Report remaining missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex and at least one column ``target_col``.
    target_col : str
        The energy consumption column to clean.
    lower_clip : float, optional
        Minimum physically valid value (default 0 — power cannot be negative).
    upper_clip : float, optional
        Maximum physically valid value. Set to None to skip upper clipping.
    max_interpolation_gap : int
        Maximum number of consecutive NaN values to interpolate over.
        Larger gaps are left as NaN and flagged as warnings.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned copy of the input.
    messages : list[str]
        Human-readable notes about the operations performed.
    """
    messages: list[str] = []
    df = df.copy()

    # 1. Resolve duplicate timestamps
    n_dups = df.index.duplicated().sum()
    if n_dups > 0:
        df = df.groupby(df.index).mean(numeric_only=True)
        messages.append(f"Resolved {n_dups} duplicate timestamp(s) by averaging.")

    # 2. Clip physically implausible values
    if lower_clip is not None:
        n_below = (df[target_col] < lower_clip).sum()
        if n_below > 0:
            df.loc[df[target_col] < lower_clip, target_col] = np.nan
            messages.append(
                f"Replaced {n_below} value(s) below {lower_clip} with NaN "
                f"(physically implausible for energy/power)."
            )

    if upper_clip is not None:
        n_above = (df[target_col] > upper_clip).sum()
        if n_above > 0:
            df.loc[df[target_col] > upper_clip, target_col] = np.nan
            messages.append(
                f"Replaced {n_above} value(s) above {upper_clip} with NaN "
                f"(treated as outliers)."
            )

    # 3. Interpolate small gaps (time-based, so gaps proportional to duration)
    n_missing_before = df[target_col].isna().sum()
    if n_missing_before > 0:
        df[target_col] = df[target_col].interpolate(
            method="time",
            limit=max_interpolation_gap,
            limit_direction="forward",
        )
        n_missing_after = df[target_col].isna().sum()
        n_filled = n_missing_before - n_missing_after
        if n_filled > 0:
            messages.append(
                f"Interpolated {n_filled} missing value(s) using time-based interpolation."
            )
        if n_missing_after > 0:
            messages.append(
                f"{n_missing_after} value(s) remain missing (gap > {max_interpolation_gap} steps)."
            )

    return df, messages


def resample_series(
    df: pd.DataFrame,
    target_col: str,
    freq: str = "1h",
    agg_func: str = "mean",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Resample a time-indexed DataFrame to a uniform frequency.

    Power (W, kW) columns should use 'mean' aggregation.
    Energy (Wh, kWh) columns should use 'sum' aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    target_col : str
        Primary energy/power column.
    freq : str
        Target pandas frequency string (e.g. "1h", "15min", "1D").
    agg_func : str
        Aggregation function for numeric columns ("mean" or "sum").

    Returns
    -------
    df_resampled : pd.DataFrame
    messages : list[str]
    """
    messages: list[str] = []
    original_len = len(df)

    # Aggregate all numeric columns; use the specified function for the target
    # and mean for others.
    resampler = df.resample(freq)
    if agg_func == "sum":
        df_resampled = resampler.sum(min_count=1)
    else:
        df_resampled = resampler.mean(numeric_only=True)

    new_len = len(df_resampled)
    pct_missing = df_resampled[target_col].isna().mean() * 100

    messages.append(
        f"Resampled from {original_len} rows to {new_len} rows at '{freq}' frequency."
    )
    if pct_missing > 10:
        messages.append(
            f"Warning: {pct_missing:.1f}% of resampled time steps have no data — "
            "the source data has significant gaps."
        )

    return df_resampled, messages


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = None,
    val_ratio: float = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame into train / validation / test sets
    in chronological order.

    **No shuffling.** Splitting by position (earliest → train, latest → test)
    ensures the model never sees future data during training.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted (chronologically) DataFrame.
    train_ratio : float
        Fraction to use for training. Defaults to config.TRAIN_RATIO.
    val_ratio : float
        Fraction to use for validation. Defaults to config.VALIDATION_RATIO.

    Returns
    -------
    train, val, test : pd.DataFrame, pd.DataFrame, pd.DataFrame
    """
    if train_ratio is None:
        train_ratio = config.TRAIN_RATIO
    if val_ratio is None:
        val_ratio = config.VALIDATION_RATIO

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    return train, val, test


def convert_units(
    series: pd.Series,
    from_unit: str,
    to_unit: str = "kWh",
    sampling_freq_hours: float = 1.0,
) -> pd.Series:
    """
    Convert a power or energy series to kilowatt-hours (kWh) for consistent billing.

    Parameters
    ----------
    series : pd.Series
        Numeric series of energy/power readings.
    from_unit : str
        Source unit: one of "kW", "W", "kWh", "Wh".
    to_unit : str
        Target unit (currently only "kWh" is supported as output).
    sampling_freq_hours : float
        Duration of each time step in hours. Used to convert instantaneous power
        to energy (e.g. 1.0 kW × 1 h = 1.0 kWh). This is set automatically
        from the resampled frequency.

    Returns
    -------
    pd.Series
        Values in the target unit (kWh by default).
    """
    from_unit = from_unit.strip()

    if from_unit == "kW":
        # Instantaneous power → energy: multiply by time step duration
        return series * sampling_freq_hours
    elif from_unit == "W":
        return series / 1000 * sampling_freq_hours
    elif from_unit == "kWh":
        return series
    elif from_unit == "Wh":
        return series / 1000
    else:
        raise ValueError(
            f"Unsupported unit '{from_unit}'. "
            "Supported units: 'kW', 'W', 'kWh', 'Wh'."
        )
