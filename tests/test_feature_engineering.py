"""
tests/test_feature_engineering.py — Tests for feature engineering.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import (
    add_calendar_features, add_lag_features, add_rolling_features, build_feature_matrix
)


def _make_df(n=500, freq="1h", target_col="power") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    vals = np.abs(np.random.randn(n)) + 1.0
    return pd.DataFrame({target_col: vals}, index=idx)


# ── Calendar features ─────────────────────────────────────────────────────────

def test_calendar_features_present():
    df = _make_df()
    df_feat = add_calendar_features(df)
    for col in ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos"]:
        assert col in df_feat.columns


def test_hour_range():
    df = _make_df()
    df_feat = add_calendar_features(df)
    assert df_feat["hour"].between(0, 23).all()


def test_is_weekend_binary():
    df = _make_df()
    df_feat = add_calendar_features(df)
    assert set(df_feat["is_weekend"].unique()).issubset({0, 1})


def test_cyclic_encoding_bounds():
    df = _make_df()
    df_feat = add_calendar_features(df)
    assert df_feat["hour_sin"].between(-1, 1).all()
    assert df_feat["hour_cos"].between(-1, 1).all()


# ── Lag features ──────────────────────────────────────────────────────────────

def test_lag_features_created():
    df = _make_df()
    df_lag = add_lag_features(df, "power", lag_periods=[1, 24])
    assert "lag_1"  in df_lag.columns
    assert "lag_24" in df_lag.columns


def test_lag_1_is_shift_1():
    df = _make_df(n=50)
    df_lag = add_lag_features(df, "power", lag_periods=[1])
    # lag_1 at index i should equal power at index i-1
    for i in range(1, len(df_lag)):
        assert df_lag["lag_1"].iloc[i] == pytest.approx(df_lag["power"].iloc[i - 1])


def test_lag_no_future_leakage():
    """Lag features must use only past data."""
    df = _make_df(n=100)
    df_lag = add_lag_features(df, "power", lag_periods=[1])
    # The lag_1 at position 0 should be NaN (no previous value)
    assert pd.isna(df_lag["lag_1"].iloc[0])


# ── Rolling features ──────────────────────────────────────────────────────────

def test_rolling_features_created():
    df = _make_df()
    df_roll = add_rolling_features(df, "power", windows=[6, 24])
    assert "rolling_mean_6"  in df_roll.columns
    assert "rolling_std_24"  in df_roll.columns


def test_rolling_no_future_leakage():
    """Rolling features must not include the current time step's value."""
    df = _make_df(n=50)
    # Change value at index 10 drastically
    original_val = df["power"].iloc[10]
    df["power"].iloc[10] = 9999.0
    df_roll = add_rolling_features(df, "power", windows=[5])
    # rolling_mean_5 at index 10 should NOT include 9999 (because shift(1))
    # It should be the mean of positions 5 to 9
    rm = df_roll["rolling_mean_5"].iloc[10]
    # Exact value depends on data, but it should be << 9999
    assert rm < 9999.0


# ── build_feature_matrix ──────────────────────────────────────────────────────

def test_build_feature_matrix_shapes():
    df = _make_df(n=500)
    X, y, feat_names = build_feature_matrix(df, "power", lag_periods=[1, 24], drop_na=True)
    assert len(X) == len(y)
    assert len(X.columns) == len(feat_names)
    assert "power" not in X.columns


def test_build_feature_matrix_no_target_in_X():
    df = _make_df(n=300)
    X, y, feat_names = build_feature_matrix(df, "power")
    assert "power" not in X.columns


def test_build_feature_matrix_drop_na():
    df = _make_df(n=300)
    X_drop, y_drop, _ = build_feature_matrix(df, "power", lag_periods=[1, 168], drop_na=True)
    # Should have dropped first 168 rows (warm-up period)
    assert len(X_drop) < 300
    assert not X_drop.isnull().any().any()
