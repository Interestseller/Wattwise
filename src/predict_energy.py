"""
src/predict_energy.py — Generate multi-step energy forecasts from a trained model.

How multi-step (recursive) forecasting works
---------------------------------------------
Our ML models were trained to predict ONE step ahead:
    y[t+1] = f(features at time t)

To forecast H steps ahead, we use RECURSIVE forecasting:
    Step 1: predict y[t+1] using known features at t
    Step 2: use predicted y[t+1] as lag_1 to predict y[t+2]
    Step 3: use predicted y[t+1] and y[t+2] as lag features to predict y[t+3]
    ...

This "feeds predictions back as inputs" and is the standard approach for
lag-based ML forecasters when real future data is not available.

The main downside: errors accumulate across steps. Accuracy typically
degrades for longer horizons, which is expected and honest behaviour.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.feature_engineering import add_calendar_features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast(
    model,
    scaler,
    feature_names: list[str],
    history: pd.Series,
    steps: int,
    freq: str = "1h",
    lag_periods: list[int] = None,
    rolling_windows: list[int] = None,
) -> pd.DataFrame:
    """
    Generate a multi-step forecast using recursive prediction.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        The trained forecaster returned by train_and_select_model.
    scaler : fitted StandardScaler or None
        If the model was trained on scaled features, pass the same scaler.
        Pass None for the baseline model (which doesn't use scaling).
    feature_names : list[str]
        Ordered list of feature column names the model expects.
    history : pd.Series
        Historical target values with a DatetimeIndex.
        Must contain at least max(lag_periods) + max(rolling_windows) steps.
    steps : int
        Number of future time steps to forecast.
    freq : str
        Pandas frequency string for the time index (e.g. "1h", "15min").
    lag_periods : list[int], optional
        Must match the training configuration. Defaults to config.LAG_PERIODS.
    rolling_windows : list[int], optional
        Must match the training configuration. Defaults to config.ROLLING_WINDOWS.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp  : future DatetimeIndex
        predicted  : forecast values
        step       : 1-based forecast horizon number
    """
    if lag_periods is None:
        lag_periods = config.LAG_PERIODS
    if rolling_windows is None:
        rolling_windows = config.ROLLING_WINDOWS

    max_lag = max(lag_periods) if lag_periods else 0
    max_win = max(rolling_windows) if rolling_windows else 0
    context_needed = max(max_lag, max_win) + steps

    # Keep the last `context_needed` values for computing lags
    recent = list(history.dropna().values[-context_needed:])
    freq_offset = pd.tseries.frequencies.to_offset(freq)
    last_ts = history.dropna().index[-1]

    forecast_records = []

    for step in range(1, steps + 1):
        ts = last_ts + freq_offset * step

        # ── Build one feature row for this future timestamp ─────────────────
        row = _build_one_row(
            ts=ts,
            history=recent,
            lag_periods=lag_periods,
            rolling_windows=rolling_windows,
            feature_names=feature_names,
        )

        row_df = pd.DataFrame([row], columns=feature_names, index=[ts])

        # ── Predict ─────────────────────────────────────────────────────────
        if scaler is not None:
            try:
                row_scaled = scaler.transform(row_df)
                row_df_scaled = pd.DataFrame(row_scaled, columns=feature_names, index=[ts])
                pred = float(model.predict(row_df_scaled)[0])
            except Exception:
                pred = float(model.predict(row_df)[0])
        else:
            pred = float(model.predict(row_df)[0])

        # Clip to non-negative (energy cannot be negative)
        pred = max(pred, 0.0)

        forecast_records.append({
            "timestamp": ts,
            "predicted": pred,
            "step": step,
        })

        # ── Update rolling context with this prediction ──────────────────────
        recent.append(pred)

    result_df = pd.DataFrame(forecast_records).set_index("timestamp")
    return result_df


def estimate_uncertainty(
    forecast_df: pd.DataFrame,
    history: pd.Series,
    confidence: float = 0.90,
) -> pd.DataFrame:
    """
    Add a simple uncertainty band to a forecast using historical residuals.

    This is a heuristic approach, NOT a statistically rigorous prediction
    interval.  It assumes future errors will be roughly proportional to the
    standard deviation of recent errors.  Use it as a visual guide only.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of forecast().
    history : pd.Series
        Historical target series used to estimate error magnitude.
    confidence : float
        Desired nominal coverage (e.g. 0.90 → ±90% empirical band).

    Returns
    -------
    forecast_df with added 'lower' and 'upper' columns.
    """
    # Use last 30 days of history to estimate volatility
    recent = history.dropna().tail(30 * 24)
    if len(recent) < 10:
        recent = history.dropna()

    # Rolling std as a proxy for forecast uncertainty
    std = float(recent.std())
    z = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96}.get(confidence, 1.645)

    # Uncertainty grows with forecast horizon (errors accumulate)
    df = forecast_df.copy()
    steps = df["step"].values
    multiplier = np.sqrt(steps)  # heuristic: uncertainty ~ sqrt(horizon)

    df["lower"] = (df["predicted"] - z * std * multiplier).clip(lower=0)
    df["upper"] = df["predicted"] + z * std * multiplier

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_one_row(
    ts: pd.Timestamp,
    history: list[float],
    lag_periods: list[int],
    rolling_windows: list[int],
    feature_names: list[str],
) -> dict:
    """
    Build a single feature row for one future timestamp.

    Uses the growing `history` list (which includes already-predicted values).
    """
    row: dict[str, float] = {}

    # Calendar features
    row["hour"]         = ts.hour
    row["day_of_week"]  = ts.dayofweek
    row["day_of_month"] = ts.day
    row["month"]        = ts.month
    row["quarter"]      = ts.quarter
    row["is_weekend"]   = int(ts.dayofweek >= 5)

    conditions = [
        (ts.hour >= 0  and ts.hour < 6),
        (ts.hour >= 6  and ts.hour < 12),
        (ts.hour >= 12 and ts.hour < 18),
        (ts.hour >= 18 and ts.hour < 24),
    ]
    row["time_of_day"] = [0, 1, 2, 3][[i for i, c in enumerate(conditions) if c][0]]

    row["hour_sin"]   = np.sin(2 * np.pi * ts.hour / 24)
    row["hour_cos"]   = np.cos(2 * np.pi * ts.hour / 24)
    row["dow_sin"]    = np.sin(2 * np.pi * ts.dayofweek / 7)
    row["dow_cos"]    = np.cos(2 * np.pi * ts.dayofweek / 7)
    row["month_sin"]  = np.sin(2 * np.pi * ts.month / 12)
    row["month_cos"]  = np.cos(2 * np.pi * ts.month / 12)

    # Lag features: history[-k] = k steps before this forecast step
    for lag in lag_periods:
        idx = -(lag)
        col = f"lag_{lag}"
        if abs(idx) <= len(history):
            row[col] = history[idx]
        else:
            row[col] = float(np.mean(history)) if history else 0.0

    # Rolling features (window over the last w values in history)
    for w in rolling_windows:
        window_data = history[-w:] if len(history) >= w else history
        col_mean = f"rolling_mean_{w}"
        col_std  = f"rolling_std_{w}"
        row[col_mean] = float(np.mean(window_data)) if window_data else 0.0
        row[col_std]  = float(np.std(window_data)) if len(window_data) > 1 else 0.0

    # Build final row with only the expected feature columns (in order)
    return {k: row.get(k, 0.0) for k in feature_names}
