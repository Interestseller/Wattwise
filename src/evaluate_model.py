"""
src/evaluate_model.py — Compute and explain forecasting evaluation metrics.

Metrics explained
-----------------

MAE — Mean Absolute Error
    Average of |actual - predicted| across all test observations.
    "On average, my predictions are off by X kWh."
    Easy to interpret; same unit as the target.

RMSE — Root Mean Squared Error
    Square root of the average of (actual - predicted)².
    Penalises large errors more heavily than MAE because of the squaring step.
    "Typical error magnitude, with extra weight on big mistakes."

R² — Coefficient of Determination
    How much of the variance in actual values is explained by the model.
    R² = 1.0 → perfect predictions.
    R² = 0.0 → model is no better than always predicting the mean.
    R² < 0   → model is worse than always predicting the mean.

MAPE — Mean Absolute Percentage Error
    Average of |actual - predicted| / |actual| × 100.
    Intuitive percentage interpretation but UNRELIABLE near zero consumption.
    We only compute it when the minimum target value is > 0.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_target_for_mape: float = 0.1,
) -> dict:
    """
    Compute MAE, RMSE, R², and optionally MAPE.

    Parameters
    ----------
    y_true : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Model predictions.
    min_target_for_mape : float
        Compute MAPE only when all actual values are above this threshold.
        This avoids division-by-zero for night-time zero-consumption readings.

    Returns
    -------
    dict with keys: mae, rmse, r2, mape (optional), n_samples
    """
    # Ensure arrays, drop paired NaN
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None, "mape": None, "n_samples": 0}

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None

    mape = None
    if np.all(y_true > min_target_for_mape):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {
        "mae":      round(mae,  4),
        "rmse":     round(rmse, 4),
        "r2":       round(r2, 4) if r2 is not None else None,
        "mape":     round(mape, 2) if mape is not None else None,
        "n_samples": int(len(y_true)),
    }


def explain_metrics(metrics: dict, unit: str = "kWh") -> dict[str, str]:
    """
    Return plain-English explanations for each metric value.

    Parameters
    ----------
    metrics : dict
        Output of evaluate_predictions().
    unit : str
        Unit label for the target variable (used in the explanation text).

    Returns
    -------
    dict mapping metric name → human-readable sentence.
    """
    explanations = {}

    mae = metrics.get("mae")
    if mae is not None:
        explanations["MAE"] = (
            f"MAE = {mae:.4f} {unit}  — On average, predictions differ from "
            f"actual values by {mae:.4f} {unit}."
        )

    rmse = metrics.get("rmse")
    if rmse is not None:
        explanations["RMSE"] = (
            f"RMSE = {rmse:.4f} {unit}  — Typical prediction error is around "
            f"{rmse:.4f} {unit}. Large errors are penalised more heavily than in MAE."
        )

    r2 = metrics.get("r2")
    if r2 is not None:
        pct = r2 * 100
        if r2 >= 0.9:
            quality = "excellent"
        elif r2 >= 0.7:
            quality = "good"
        elif r2 >= 0.4:
            quality = "moderate"
        else:
            quality = "poor"
        explanations["R²"] = (
            f"R² = {r2:.4f} ({pct:.1f}% variance explained) — {quality.capitalize()} fit. "
            f"A value of 1.0 means perfect predictions; 0.0 means the model is no better "
            f"than always predicting the mean."
        )

    mape = metrics.get("mape")
    if mape is not None:
        explanations["MAPE"] = (
            f"MAPE = {mape:.2f}% — The average absolute percentage error. "
            f"Only reliable when consumption is never near zero."
        )
    else:
        explanations["MAPE"] = (
            "MAPE not computed — target contains zero or very low values "
            "where percentage errors become misleading."
        )

    return explanations


def build_comparison_table(all_metrics: dict) -> pd.DataFrame:
    """
    Build a model-comparison DataFrame from per-model metric dicts.

    Parameters
    ----------
    all_metrics : dict
        {model_name: {mae, rmse, r2, mape, n_samples}, …}

    Returns
    -------
    pd.DataFrame sorted by RMSE ascending.
    """
    rows = []
    for model_name, m in all_metrics.items():
        if not isinstance(m, dict) or "rmse" not in m:
            continue
        rows.append({
            "Model":     model_name,
            "MAE":       m.get("mae"),
            "RMSE":      m.get("rmse"),
            "R²":        m.get("r2"),
            "MAPE (%)":  m.get("mape"),
            "Samples":   m.get("n_samples"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Model")
    df = df.sort_values("RMSE", ascending=True)
    return df
