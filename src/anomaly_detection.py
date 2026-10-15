"""
src/anomaly_detection.py — Detect unusual energy consumption patterns.

What is anomaly detection?
--------------------------
Anomaly detection (also called outlier detection) identifies observations
that look unusual compared to the typical consumption pattern.

We implement TWO complementary methods:

1. Isolation Forest (scikit-learn)
   An unsupervised ML algorithm that "isolates" anomalies by building many
   random trees. Points that require few splits to isolate are anomalies.
   Works well when the anomaly shape is unknown.

2. Rolling Z-Score
   For each hour, compare the current value to the rolling mean and std of
   recent history.  Z-score > threshold → unusual.
   Easy to interpret: "this reading is N standard deviations from recent normal."

Important caveats
-----------------
* These methods flag UNUSUAL patterns, not confirmed electrical faults.
* Planned unusual events (Christmas, heat wave) may be flagged as anomalies.
* Results should be reviewed by a human with domain knowledge.
* Anomaly detection is a SEPARATE component from the forecasting model.
  Do not use the forecasting model output to "define" what is anomalous.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_anomalies(
    series: pd.Series,
    method: str = "isolation_forest",
    contamination: float = None,
    zscore_threshold: float = None,
    rolling_window: int = None,
    feature_window: int = 24,
) -> pd.DataFrame:
    """
    Detect unusual consumption readings in an energy time series.

    Parameters
    ----------
    series : pd.Series
        Time-indexed numeric energy/power series.
    method : str
        Detection method: "isolation_forest", "zscore", or "both".
    contamination : float
        Expected fraction of anomalies for Isolation Forest.
        Defaults to config.ANOMALY_CONTAMINATION.
    zscore_threshold : float
        Z-score threshold for the statistical method.
        Defaults to config.ZSCORE_THRESHOLD.
    rolling_window : int
        Window size for rolling Z-score baseline.
        Defaults to config.ROLLING_ZSCORE_WINDOW.
    feature_window : int
        Window size (in time steps) used to build features for Isolation Forest.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp     : index (same as input series)
        value         : original reading
        is_anomaly    : bool (True = flagged)
        anomaly_score : float (higher = more anomalous; method-specific scale)
        expected      : estimated baseline (rolling mean)
        deviation     : value - expected
        method        : detection method name
    """
    if contamination is None:
        contamination = config.ANOMALY_CONTAMINATION
    if zscore_threshold is None:
        zscore_threshold = config.ZSCORE_THRESHOLD
    if rolling_window is None:
        rolling_window = config.ROLLING_ZSCORE_WINDOW

    series = series.dropna()

    if len(series) < 20:
        raise ValueError(
            "Need at least 20 observations for anomaly detection. "
            "Provide more data or a longer date range."
        )

    results = {}

    if method in ("isolation_forest", "both"):
        results["isolation_forest"] = _isolation_forest_detect(
            series, contamination=contamination, feature_window=feature_window
        )

    if method in ("zscore", "both"):
        results["zscore"] = _rolling_zscore_detect(
            series,
            threshold=zscore_threshold,
            window=rolling_window,
        )

    if len(results) == 1:
        return list(results.values())[0]

    # Combine: a point is anomalous if flagged by EITHER method
    df_if = results["isolation_forest"].rename(columns={"is_anomaly": "if_anomaly", "anomaly_score": "if_score"})
    df_zs = results["zscore"].rename(columns={"is_anomaly": "zs_anomaly", "anomaly_score": "zs_score"})
    df = df_if.join(df_zs[["zs_anomaly", "zs_score"]], how="outer")
    df["is_anomaly"] = df["if_anomaly"] | df["zs_anomaly"].fillna(False)
    df["anomaly_score"] = df[["if_score", "zs_score"]].max(axis=1)
    df["method"] = "both"
    return df


def summarise_anomalies(df: pd.DataFrame) -> dict:
    """
    Summarise anomaly detection results in human-readable form.

    Parameters
    ----------
    df : pd.DataFrame
        Output of detect_anomalies().

    Returns
    -------
    dict with keys:
        total_anomalies, pct_anomalous, top_anomalies (DataFrame),
        worst_timestamp, worst_deviation, worst_value
    """
    anomalies = df[df["is_anomaly"] == True].copy()  # noqa: E712

    if anomalies.empty:
        return {
            "total_anomalies": 0,
            "pct_anomalous":   0.0,
            "top_anomalies":   pd.DataFrame(),
            "worst_timestamp": None,
            "worst_deviation": None,
            "worst_value":     None,
        }

    top = anomalies.nlargest(10, "anomaly_score")

    worst = anomalies.loc[anomalies["deviation"].abs().idxmax()]

    return {
        "total_anomalies":  int(len(anomalies)),
        "pct_anomalous":    round(len(anomalies) / len(df) * 100, 2),
        "top_anomalies":    top,
        "worst_timestamp":  worst.name,
        "worst_deviation":  float(worst["deviation"]),
        "worst_value":      float(worst["value"]),
    }


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

def _isolation_forest_detect(
    series: pd.Series,
    contamination: float,
    feature_window: int,
) -> pd.DataFrame:
    """
    Use Isolation Forest to detect anomalies.

    We build features from the raw value, its rolling mean, rolling std,
    and hour-of-day to give the model context beyond the raw reading.
    """
    df = pd.DataFrame({"value": series})
    df["hour"]         = series.index.hour
    df["rolling_mean"] = series.rolling(window=feature_window, min_periods=1).mean()
    df["rolling_std"]  = series.rolling(window=feature_window, min_periods=1).std().fillna(0)
    df["deviation"]    = df["value"] - df["rolling_mean"]

    feature_cols = ["value", "hour", "rolling_mean", "rolling_std"]
    X = df[feature_cols].fillna(0).values

    clf = IsolationForest(
        contamination=contamination,
        random_state=config.RANDOM_STATE,
        n_estimators=100,
    )
    clf.fit(X)

    # IsolationForest returns -1 for outliers, +1 for inliers
    labels = clf.predict(X)
    # decision_function: lower score = more anomalous
    raw_scores = clf.decision_function(X)

    df["is_anomaly"]    = labels == -1
    # Invert and normalise to [0, 1] for interpretability
    df["anomaly_score"] = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
    df["expected"]      = df["rolling_mean"]
    df["method"]        = "isolation_forest"

    return df[["value", "is_anomaly", "anomaly_score", "expected", "deviation", "method"]]


def _rolling_zscore_detect(
    series: pd.Series,
    threshold: float,
    window: int,
) -> pd.DataFrame:
    """
    Detect anomalies using rolling Z-score.

    Z-score at time t = (value[t] - rolling_mean[t]) / rolling_std[t]
    If |Z| > threshold the observation is flagged.
    """
    rolling_mean = series.rolling(window=window, min_periods=max(3, window // 4)).mean()
    rolling_std  = series.rolling(window=window, min_periods=max(3, window // 4)).std()

    # Avoid division by zero when std is 0 (constant consumption period)
    safe_std = rolling_std.replace(0, np.nan).fillna(series.std() or 1.0)

    z_scores = (series - rolling_mean) / safe_std
    deviation = series - rolling_mean

    df = pd.DataFrame({
        "value":         series,
        "is_anomaly":    z_scores.abs() > threshold,
        "anomaly_score": z_scores.abs(),
        "expected":      rolling_mean,
        "deviation":     deviation,
        "method":        "rolling_zscore",
    })
    df["is_anomaly"] = df["is_anomaly"].fillna(False)

    return df
