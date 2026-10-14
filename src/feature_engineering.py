"""
src/feature_engineering.py — Build ML-ready features from a time series.

Feature categories
------------------
1. Calendar features (hour, day-of-week, month, …)
   Tell the model what "time" it is. Energy consumption has strong periodic
   patterns driven by human schedules.

2. Lag features (lag_1, lag_24, lag_168, …)
   Tell the model what consumption was at past time steps.
   lag_1  = previous hour (short-term autocorrelation)
   lag_24 = same hour yesterday (daily pattern)
   lag_168 = same hour last week (weekly pattern)

3. Rolling statistics (rolling_mean_24, rolling_std_24, …)
   Smooth local trends and capture recent volatility.

IMPORTANT: lag and rolling features can only use PAST data.
At prediction time, lag_1 means "what was the last known measurement?"
We NEVER use future values to compute these features (which would be
data leakage and make the model appear more accurate than it really is).

When building features for a future forecast we use recursive forecasting:
the model predicts one step ahead, that prediction becomes the lag for
the next step, and so on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based calendar features from a DatetimeIndex.

    Features added
    --------------
    hour            : 0–23
    day_of_week     : 0 (Mon) – 6 (Sun)
    day_of_month    : 1–31
    month           : 1–12
    quarter         : 1–4
    is_weekend      : 1 if Saturday or Sunday, else 0
    time_of_day     : 0=night (0–5), 1=morning (6–11), 2=afternoon (12–17), 3=evening (18–23)
    hour_sin/cos    : cyclic encoding so the model knows hour 23 and hour 0 are adjacent
    dow_sin/cos     : cyclic encoding for day-of-week

    Why cyclic encoding?
    Standard numeric hour (0–23) implies that hour 0 and hour 23 are far apart.
    Sine/cosine transforms "wrap" the scale so midnight and 11 PM are treated as
    close neighbours — which better matches the real periodicity of energy consumption.
    """
    df = df.copy()
    idx = df.index

    df["hour"]         = idx.hour
    df["day_of_week"]  = idx.dayofweek
    df["day_of_month"] = idx.day
    df["month"]        = idx.month
    df["quarter"]      = idx.quarter
    df["is_weekend"]   = (idx.dayofweek >= 5).astype(int)

    # Time-of-day bucket
    conditions = [
        (df["hour"] >= 0)  & (df["hour"] < 6),
        (df["hour"] >= 6)  & (df["hour"] < 12),
        (df["hour"] >= 12) & (df["hour"] < 18),
        (df["hour"] >= 18) & (df["hour"] < 24),
    ]
    choices = [0, 1, 2, 3]  # night, morning, afternoon, evening
    df["time_of_day"] = np.select(conditions, choices, default=0)

    # Cyclic hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Cyclic day-of-week encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Cyclic month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_periods: list[int] = None,
) -> pd.DataFrame:
    """
    Add lagged values of the target column as features.

    lag_N means "the value N time steps ago".

    Why lags matter
    ---------------
    Energy consumption has strong autocorrelation: what you consumed an hour ago
    is a strong predictor of what you will consume in the next hour.
    Weekly patterns (lag_168 for hourly data) capture the same-time-last-week
    consumption which is highly predictive for household loads.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex and a target column.
    target_col : str
        Column to create lags for.
    lag_periods : list[int]
        List of lag steps. Defaults to config.LAG_PERIODS.

    Returns
    -------
    pd.DataFrame with new lag_* columns. Rows with NaN lags are kept but
    flagged — the caller decides whether to drop them.
    """
    if lag_periods is None:
        lag_periods = config.LAG_PERIODS

    df = df.copy()
    for lag in lag_periods:
        col_name = f"lag_{lag}"
        df[col_name] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int] = None,
) -> pd.DataFrame:
    """
    Add rolling-mean and rolling-std features for the target column.

    rolling_mean_N = average of the last N time steps (excluding current)
    rolling_std_N  = standard deviation of the last N time steps

    Why rolling features matter
    ---------------------------
    Rolling means smooth out short spikes and give the model context about
    recent trends. Rolling standard deviation helps the model detect
    periods of higher volatility (e.g. a heat wave causing prolonged high usage).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    windows : list[int]
        Rolling window sizes in time steps. Defaults to config.ROLLING_WINDOWS.

    Returns
    -------
    pd.DataFrame with new rolling_mean_* and rolling_std_* columns.
    """
    if windows is None:
        windows = config.ROLLING_WINDOWS

    df = df.copy()
    for w in windows:
        # shift(1) ensures we only use values UP TO t-1 (no leakage at t)
        shifted = df[target_col].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w, min_periods=1).mean()
        df[f"rolling_std_{w}"]  = shifted.rolling(window=w, min_periods=1).std()

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    lag_periods: list[int] = None,
    rolling_windows: list[int] = None,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build the full feature matrix X and target vector y from a preprocessed DataFrame.

    This is the single entry-point used by the training pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed, resampled DataFrame with a DatetimeIndex.
    target_col : str
        Name of the target variable column.
    lag_periods : list[int], optional
    rolling_windows : list[int], optional
    drop_na : bool
        If True, drop rows where any feature is NaN (caused by initial lag periods).
        Set to False for inference on a full series where you want to keep NaT rows.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (does not include target_col).
    y : pd.Series
        Target vector.
    feature_names : list[str]
        List of feature column names (for saving with the model).
    """
    df = add_calendar_features(df)
    df = add_lag_features(df, target_col, lag_periods)
    df = add_rolling_features(df, target_col, rolling_windows)

    feature_cols = [c for c in df.columns if c != target_col]

    if drop_na:
        df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols]
    y = df[target_col]

    return X, y, feature_cols


def get_feature_descriptions() -> dict[str, str]:
    """Return human-readable descriptions for each feature type."""
    return {
        "hour":           "Hour of day (0–23). Captures within-day consumption patterns.",
        "day_of_week":    "Day of week (0=Monday, 6=Sunday). Captures weekday vs weekend patterns.",
        "day_of_month":   "Day of the month (1–31). Captures monthly billing cycle patterns.",
        "month":          "Month of year (1–12). Captures seasonal variation.",
        "quarter":        "Quarter (1–4). Coarse seasonal grouping.",
        "is_weekend":     "1 if Saturday or Sunday, 0 otherwise. Weekends often have different usage.",
        "time_of_day":    "0=night, 1=morning, 2=afternoon, 3=evening. Coarse time bucket.",
        "hour_sin/cos":   "Cyclic encoding of hour so that hour 23 and hour 0 are adjacent.",
        "dow_sin/cos":    "Cyclic encoding of day-of-week.",
        "month_sin/cos":  "Cyclic encoding of month.",
        "lag_N":          "Consumption N time-steps ago. lag_1=previous step, lag_24=yesterday same hour.",
        "rolling_mean_N": "Average consumption over the last N time steps (excluding current).",
        "rolling_std_N":  "Standard deviation over the last N time steps — captures volatility.",
    }
