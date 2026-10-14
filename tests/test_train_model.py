"""
tests/test_train_model.py — Tests for training, evaluation and prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.train_model import train_and_select_model, save_model, load_model, BaselineModel
from src.evaluate_model import evaluate_predictions, build_comparison_table
from src.predict_energy import forecast


RANDOM_STATE = 42


def _make_df(n=500, freq="1h") -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    # Realistic pattern: daily cycle + noise
    hours = np.array(idx.hour)
    vals = 1.0 + 0.5 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.1, n)
    vals = np.clip(vals, 0.1, None)
    return pd.DataFrame({"power": vals}, index=idx)


# ── BaselineModel ─────────────────────────────────────────────────────────────

def test_baseline_model_predicts():
    df = _make_df(n=300)
    from src.feature_engineering import build_feature_matrix
    X, y, _ = build_feature_matrix(df, "power", lag_periods=[1, 24], drop_na=True)
    model = BaselineModel(lag=24)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.all(np.isfinite(preds))


# ── train_and_select_model ────────────────────────────────────────────────────

def test_training_completes():
    df = _make_df(n=500)
    result = train_and_select_model(
        df=df,
        target_col="power",
        lag_periods=[1, 24],
        rolling_windows=[6, 24],
        models_to_train=["baseline", "random_forest"],
        random_state=RANDOM_STATE,
    )
    assert result["best_model"] is not None
    assert result["best_model_name"] in ["baseline", "random_forest"]
    assert len(result["feature_names"]) > 0


def test_best_model_beats_or_matches_baseline():
    """The selected model should not be worse than the baseline."""
    df = _make_df(n=600)
    result = train_and_select_model(
        df=df,
        target_col="power",
        lag_periods=[1, 24],
        models_to_train=["baseline", "random_forest"],
        random_state=RANDOM_STATE,
    )
    val_metrics = result["val_metrics"]
    best = result["best_model_name"]
    best_rmse = val_metrics[best]["rmse"]
    baseline_rmse = val_metrics["baseline"]["rmse"]
    # Best model rmse should be <= baseline rmse (within 5% tolerance for small datasets)
    assert best_rmse <= baseline_rmse * 1.05


def test_chronological_split_maintained():
    """Training set must end before test set starts."""
    df = _make_df(n=500)
    result = train_and_select_model(df, "power", models_to_train=["baseline"], random_state=RANDOM_STATE)
    info = result["training_info"]
    train_end = pd.Timestamp(info["train_end"])
    test_start = pd.Timestamp(info["test_start"])
    assert train_end < test_start


def test_too_small_dataset_raises():
    df = _make_df(n=50)  # too small for default lag_periods
    with pytest.raises(ValueError, match="usable samples"):
        train_and_select_model(df, "power", lag_periods=[1, 24], random_state=RANDOM_STATE)


# ── evaluate_predictions ──────────────────────────────────────────────────────

def test_evaluate_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = evaluate_predictions(y_true, y_true)
    assert metrics["mae"]  == pytest.approx(0.0, abs=1e-9)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["r2"]   == pytest.approx(1.0, abs=1e-6)


def test_evaluate_all_same_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(2/3, abs=1e-3)  # rounded to 4 decimal places


def test_evaluate_handles_nan():
    y_true = np.array([1.0, np.nan, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["n_samples"] == 2


def test_mape_not_computed_near_zero():
    y_true = np.array([0.0, 0.1, 0.05])
    y_pred = np.array([0.1, 0.2, 0.1])
    metrics = evaluate_predictions(y_true, y_pred, min_target_for_mape=0.1)
    assert metrics["mape"] is None


# ── Model save / load ─────────────────────────────────────────────────────────

def test_save_and_load_model(tmp_path):
    df = _make_df(n=500)
    result = train_and_select_model(
        df=df,
        target_col="power",
        lag_periods=[1, 24],
        models_to_train=["baseline"],
        random_state=RANDOM_STATE,
    )
    model_path = save_model(result, models_dir=str(tmp_path))
    assert os.path.isfile(model_path)

    # Reload
    bundle = load_model(models_dir=str(tmp_path))
    assert "model"         in bundle
    assert "scaler"        in bundle
    assert "feature_names" in bundle


def test_load_model_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(models_dir=str(tmp_path))


# ── Forecasting ──────────────────────────────────────────────────────────────

def test_forecast_returns_correct_steps():
    df = _make_df(n=500)
    result = train_and_select_model(
        df=df,
        target_col="power",
        lag_periods=[1, 24],
        models_to_train=["random_forest"],
        random_state=RANDOM_STATE,
    )
    bundle = {
        "model":         result["best_model"],
        "scaler":        result["scaler"],
        "feature_names": result["feature_names"],
    }
    history = df["power"]
    fc = forecast(
        model=bundle["model"],
        scaler=bundle["scaler"],
        feature_names=bundle["feature_names"],
        history=history,
        steps=24,
        freq="1h",
        lag_periods=[1, 24],
        rolling_windows=[6],
    )
    assert len(fc) == 24
    assert "predicted" in fc.columns
    assert (fc["predicted"] >= 0).all()


def test_forecast_timestamps_are_future():
    df = _make_df(n=500)
    result = train_and_select_model(
        df=df,
        target_col="power",
        lag_periods=[1, 24],
        models_to_train=["baseline"],
        random_state=RANDOM_STATE,
    )
    history = df["power"]
    fc = forecast(
        model=result["best_model"],
        scaler=None,
        feature_names=result["feature_names"],
        history=history,
        steps=12,
        freq="1h",
        lag_periods=[1, 24],
        rolling_windows=[6],
    )
    assert fc.index.min() > history.index.max()


# ── Bill calculation ──────────────────────────────────────────────────────────

def test_bill_calculation_basic():
    total_kwh = 300.0
    tariff = 6.5
    fixed = 50.0
    tax_rate = 0.05

    energy_cost = total_kwh * tariff
    subtotal = energy_cost + fixed
    total_bill = subtotal * (1 + tax_rate)

    assert energy_cost == pytest.approx(1950.0, abs=0.01)
    assert total_bill  == pytest.approx(2100.0, abs=0.01)


# ── Anomaly detection ─────────────────────────────────────────────────────────

def test_anomaly_detection_runs():
    from src.anomaly_detection import detect_anomalies
    df = _make_df(n=500)
    series = df["power"]
    result_df = detect_anomalies(series, method="zscore", zscore_threshold=3.0)
    assert "is_anomaly" in result_df.columns
    assert result_df["is_anomaly"].dtype == bool or result_df["is_anomaly"].dtype == object


def test_anomaly_detection_flags_extremes():
    from src.anomaly_detection import detect_anomalies
    df = _make_df(n=300)
    series = df["power"].copy()
    # Plant obvious anomaly
    series.iloc[150] = 999.0
    result_df = detect_anomalies(series, method="zscore", zscore_threshold=2.0)
    assert result_df["is_anomaly"].iloc[150] == True  # noqa: E712


def test_anomaly_detection_too_small_raises():
    from src.anomaly_detection import detect_anomalies
    import pandas as pd
    tiny = pd.Series([1.0, 2.0, 1.5], index=pd.date_range("2024-01-01", periods=3, freq="1h"))
    with pytest.raises(ValueError, match="20 observations"):
        detect_anomalies(tiny, method="isolation_forest")
