"""
flask_app/services.py — Backend data-service layer for WattWise.

Wraps the existing src/ ML modules and dataset access.
All heavy Python/ML work happens here; routes stay thin.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root is on path so src/ and config are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from src.data_loader import load_from_path, load_from_upload, select_columns
from src.preprocessing import clean_series, resample_series
from src.anomaly_detection import detect_anomalies, summarise_anomalies
from src.train_model import (
    train_and_select_model, save_model,
    load_model, load_metrics, load_metadata,
)
from src.predict_energy import forecast, estimate_uncertainty

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_DEMO_PATH = os.path.join(ROOT, "data", "demo_energy_dataset.csv")
_DATASET_PATH = os.path.join(ROOT, "data", "energy_dataset.csv")


def _get_dataset_path() -> str:
    if os.path.isfile(_DATASET_PATH):
        return _DATASET_PATH
    return _DEMO_PATH


def load_clean_df(freq: str = "1h") -> tuple[pd.DataFrame, bool]:
    """
    Load and clean the energy dataset.

    Returns (df, is_demo) where is_demo=True when falling back to demo data.
    """
    path = _get_dataset_path()
    is_demo = (path == _DEMO_PATH)

    raw_df, profile = load_from_path(path)
    if not profile.is_valid or raw_df.empty:
        return pd.DataFrame(), is_demo

    df, errors = select_columns(raw_df, config.TIMESTAMP_COLUMN, config.TARGET_COLUMN)
    if errors or df.empty:
        return pd.DataFrame(), is_demo

    df, _ = clean_series(df, config.TARGET_COLUMN)
    df, _ = resample_series(df, config.TARGET_COLUMN, freq=freq)
    return df.dropna(subset=[config.TARGET_COLUMN]), is_demo


# ---------------------------------------------------------------------------
# Overview API
# ---------------------------------------------------------------------------

def get_overview_data(days: int = 30) -> dict:
    df, is_demo = load_clean_df()
    if df.empty:
        return {"error": "No dataset available", "is_demo": is_demo}

    col = config.TARGET_COLUMN
    now = df.index.max()
    cutoff = now - pd.Timedelta(days=days)
    recent = df[df.index >= cutoff]
    prev = df[(df.index >= cutoff - pd.Timedelta(days=days)) & (df.index < cutoff)]

    current_usage = float(recent[col].sum())
    prev_usage = float(prev[col].sum()) if not prev.empty else None

    # Bill estimate using config tariff
    bill = current_usage * config.DEFAULT_TARIFF_PER_KWH * (1 + config.DEFAULT_TAX_RATE)
    bill += config.DEFAULT_FIXED_MONTHLY_CHARGE

    # Peak hours: find the hour-of-day with highest average consumption
    hourly_avg = recent.copy()
    hourly_avg["hour"] = hourly_avg.index.hour
    peak_hour = int(hourly_avg.groupby("hour")[col].mean().idxmax())

    # Active anomaly count
    try:
        anomaly_df = detect_anomalies(recent[col], method="zscore")
        n_anomalies = int(anomaly_df["is_anomaly"].sum())
    except Exception:
        n_anomalies = 0

    # Usage trend data (daily sums)
    daily = recent.resample("1D")[col].sum()
    hist_daily = prev.resample("1D")[col].sum() if not prev.empty else pd.Series(dtype=float)

    pct_change = None
    if prev_usage and prev_usage > 0:
        pct_change = round((current_usage - prev_usage) / prev_usage * 100, 1)

    # Category breakdown for donut (by time-of-day)
    hourly_avg["tod"] = pd.cut(
        hourly_avg["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night (0-5h)", "Morning (6-11h)", "Afternoon (12-17h)", "Evening (18-23h)"],
    )
    tod_breakdown = hourly_avg.groupby("tod", observed=True)[col].sum()
    tod_total = tod_breakdown.sum()

    # Weekly comparison (last 4 weeks)
    weeks = []
    for w in range(4, 0, -1):
        wstart = now - pd.Timedelta(weeks=w)
        wend   = now - pd.Timedelta(weeks=w - 1)
        wslice = df[(df.index >= wstart) & (df.index < wend)]
        weeks.append({"label": f"W-{w}", "value": round(float(wslice[col].sum()), 2)})

    return {
        "is_demo": is_demo,
        "current_usage": round(current_usage, 2),
        "prev_usage": round(prev_usage, 2) if prev_usage else None,
        "pct_change": pct_change,
        "bill": round(bill, 2),
        "peak_hour": peak_hour,
        "peak_hour_label": f"{peak_hour}:00 – {peak_hour+1}:00",
        "n_anomalies": n_anomalies,
        "trend": {
            "dates": [str(d.date()) for d in daily.index],
            "current": [round(v, 3) for v in daily.values],
            "historical": [round(v, 3) for v in hist_daily.values] if not hist_daily.empty else [],
            "hist_dates": [str(d.date()) for d in hist_daily.index] if not hist_daily.empty else [],
        },
        "weekly": weeks,
        "tod_breakdown": {
            "labels": list(tod_breakdown.index.astype(str)),
            "values": [round(float(v / tod_total * 100), 1) if tod_total > 0 else 0
                       for v in tod_breakdown.values],
        },
    }


# ---------------------------------------------------------------------------
# Analytics API
# ---------------------------------------------------------------------------

def get_analytics_data(period: str = "daily", days: int = 90) -> dict:
    df, is_demo = load_clean_df()
    if df.empty:
        return {"error": "No dataset available"}

    col = config.TARGET_COLUMN
    now = df.index.max()
    cutoff = now - pd.Timedelta(days=days)
    sub = df[df.index >= cutoff]

    freq_map = {"hourly": "1h", "daily": "1D", "weekly": "1W", "monthly": "1ME"}
    freq = freq_map.get(period, "1D")

    agg = sub.resample(freq)[col].sum()
    prev_agg = df[(df.index >= cutoff - pd.Timedelta(days=days)) & (df.index < cutoff)].resample(freq)[col].sum()

    # Heatmap: day-of-week × hour
    sub_copy = sub.copy()
    sub_copy["dow"] = sub_copy.index.day_name()
    sub_copy["hour"] = sub_copy.index.hour
    heatmap = sub_copy.groupby(["dow", "hour"])[col].mean().unstack(fill_value=0)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap = heatmap.reindex([d for d in dow_order if d in heatmap.index])

    return {
        "is_demo": is_demo,
        "period": period,
        "dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in agg.index],
        "values": [round(float(v), 3) for v in agg.values],
        "prev_dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in prev_agg.index],
        "prev_values": [round(float(v), 3) for v in prev_agg.values],
        "heatmap": {
            "days": list(heatmap.index),
            "hours": list(range(24)),
            "z": [[round(float(heatmap.at[d, h]), 3) if h in heatmap.columns else 0
                   for h in range(24)]
                  for d in heatmap.index],
        },
        "stats": {
            "mean": round(float(agg.mean()), 3),
            "max": round(float(agg.max()), 3),
            "min": round(float(agg.min()), 3),
            "total": round(float(agg.sum()), 3),
        },
    }


# ---------------------------------------------------------------------------
# Forecast API
# ---------------------------------------------------------------------------

def get_forecast_data(steps: int = 24) -> dict:
    df, is_demo = load_clean_df()
    if df.empty:
        return {"error": "No dataset available"}

    col = config.TARGET_COLUMN
    history = df[col].dropna()

    try:
        bundle = load_model()
    except FileNotFoundError:
        return {"error": "No trained model found. Please train a model first."}

    try:
        fc_df = forecast(
            model=bundle["model"],
            scaler=bundle["scaler"],
            feature_names=bundle["feature_names"],
            history=history,
            steps=steps,
            freq="1h",
        )
        fc_df = estimate_uncertainty(fc_df, history)
    except Exception as exc:
        return {"error": f"Forecasting failed: {exc}"}

    # Last N actuals for context
    context = history.tail(steps)

    return {
        "is_demo": is_demo,
        "model_name": bundle.get("model_name", "unknown"),
        "actual": {
            "dates": [str(ts) for ts in context.index],
            "values": [round(float(v), 3) for v in context.values],
        },
        "forecast": {
            "dates": [str(ts) for ts in fc_df.index],
            "predicted": [round(float(v), 3) for v in fc_df["predicted"].values],
            "lower": [round(float(v), 3) for v in fc_df["lower"].values],
            "upper": [round(float(v), 3) for v in fc_df["upper"].values],
        },
    }


# ---------------------------------------------------------------------------
# Model Lab API
# ---------------------------------------------------------------------------

def get_model_info() -> dict:
    metadata = load_metadata()
    metrics = load_metrics()
    model_exists = os.path.isfile(os.path.join(ROOT, config.MODEL_PATH))
    return {
        "model_exists": model_exists,
        "metadata": metadata,
        "metrics": metrics,
    }


def run_training() -> dict:
    df, is_demo = load_clean_df()
    if df.empty:
        return {"error": "No dataset available for training."}

    col = config.TARGET_COLUMN
    try:
        result = train_and_select_model(df, target_col=col)
        save_model(result)
        return {
            "success": True,
            "best_model": result["best_model_name"],
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
            "training_info": result["training_info"],
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Bill Estimator API
# ---------------------------------------------------------------------------

def estimate_bill(
    tariff: float = config.DEFAULT_TARIFF_PER_KWH,
    fixed_charge: float = config.DEFAULT_FIXED_MONTHLY_CHARGE,
    tax_rate: float = config.DEFAULT_TAX_RATE,
    reduction_pct: float = 0.0,
    days: int = 30,
) -> dict:
    df, is_demo = load_clean_df()
    if df.empty:
        return {"error": "No dataset available"}

    col = config.TARGET_COLUMN
    now = df.index.max()
    cutoff = now - pd.Timedelta(days=days)
    recent = df[df.index >= cutoff][col].dropna()

    total_kwh = float(recent.sum())
    adjusted_kwh = total_kwh * (1 - reduction_pct / 100)

    energy_cost = adjusted_kwh * tariff
    tax = energy_cost * tax_rate
    total_bill = energy_cost + tax + fixed_charge

    daily = recent.resample("1D").sum()

    return {
        "is_demo": is_demo,
        "total_kwh": round(total_kwh, 2),
        "adjusted_kwh": round(adjusted_kwh, 2),
        "energy_cost": round(energy_cost, 2),
        "tax": round(tax, 2),
        "fixed_charge": round(fixed_charge, 2),
        "total_bill": round(total_bill, 2),
        "currency": config.CURRENCY_SYMBOL,
        "daily_avg_kwh": round(float(daily.mean()), 3),
        "daily_chart": {
            "dates": [str(d.date()) for d in daily.index],
            "values": [round(float(v), 3) for v in daily.values],
        },
        "tariff": tariff,
        "tax_rate": tax_rate,
        "reduction_pct": reduction_pct,
    }


# ---------------------------------------------------------------------------
# Anomaly Monitor API
# ---------------------------------------------------------------------------

def get_anomaly_data(method: str = "both", days: int = 30) -> dict:
    df, is_demo = load_clean_df()
    if df.empty:
        return {"error": "No dataset available"}

    col = config.TARGET_COLUMN
    now = df.index.max()
    cutoff = now - pd.Timedelta(days=days)
    series = df[df.index >= cutoff][col].dropna()

    try:
        anom_df = detect_anomalies(series, method=method)
        summary = summarise_anomalies(anom_df)
    except Exception as exc:
        return {"error": str(exc)}

    anomalies = anom_df[anom_df["is_anomaly"]].copy()
    anomaly_records = []
    for ts, row in anomalies.iterrows():
        severity = "high" if abs(row.get("deviation", 0)) > series.std() * 2 else \
                   "medium" if abs(row.get("deviation", 0)) > series.std() else "low"
        anomaly_records.append({
            "timestamp": str(ts),
            "value": round(float(row["value"]), 3),
            "expected": round(float(row.get("expected", row["value"])), 3),
            "deviation": round(float(row.get("deviation", 0)), 3),
            "score": round(float(row.get("anomaly_score", 0)), 3),
            "severity": severity,
        })

    return {
        "is_demo": is_demo,
        "total": summary["total_anomalies"],
        "pct": summary["pct_anomalous"],
        "anomalies": anomaly_records,
        "chart": {
            "dates": [str(ts) for ts in anom_df.index],
            "values": [round(float(v), 3) for v in anom_df["value"].values],
            "expected": [round(float(v), 3) if not pd.isna(v) else None
                         for v in anom_df.get("expected", anom_df["value"]).values],
            "is_anomaly": [bool(v) for v in anom_df["is_anomaly"].values],
        },
    }


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def load_dataset(file_storage) -> dict:
    """
    Validate and save an uploaded CSV file as the active dataset.
    file_storage is a Flask FileStorage object (has .read() and .filename).
    """
    try:
        raw_df, profile = load_from_upload(file_storage)
        if not profile.is_valid or raw_df.empty:
            return {"error": "CSV could not be parsed. Check that it has a DateTime and energy column."}
        df, errors = select_columns(raw_df, config.TIMESTAMP_COLUMN, config.TARGET_COLUMN)
        if errors or df.empty:
            return {"error": f"Could not find required columns '{config.TIMESTAMP_COLUMN}' / "
                             f"'{config.TARGET_COLUMN}'. Found: {list(raw_df.columns[:10])}"}
    except Exception as exc:
        return {"error": f"Failed to read CSV: {exc}"}

    # Re-read raw bytes to write to disk (file_storage was already consumed above)
    import io as _io
    try:
        buf = _io.BytesIO()
        raw_df.to_csv(buf, index=False)
        with open(_DATASET_PATH, "wb") as fh:
            fh.write(buf.getvalue())
    except Exception as exc:
        return {"error": f"Could not save file: {exc}"}

    return {
        "success": True,
        "rows": len(raw_df),
        "columns": list(raw_df.columns),
        "filename": getattr(file_storage, "filename", "dataset.csv"),
    }


def export_csv(days: int = 30) -> Optional[str]:
    df, _ = load_clean_df()
    if df.empty:
        return None

    col = config.TARGET_COLUMN
    now = df.index.max()
    cutoff = now - pd.Timedelta(days=days)
    sub = df[df.index >= cutoff].copy()
    sub.index.name = "DateTime"
    return sub.to_csv()
