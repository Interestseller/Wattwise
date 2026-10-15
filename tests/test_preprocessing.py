"""
tests/test_preprocessing.py — Tests for preprocessing and splitting.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import clean_series, resample_series, chronological_split, convert_units


def _make_series(n=500, freq="1h", with_negatives=False, with_gaps=False) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    vals = np.random.rand(n) + 0.5
    if with_negatives:
        vals[5] = -1.0
    if with_gaps:
        vals[50:55] = np.nan
    return pd.DataFrame({"power": vals}, index=idx)


# ── clean_series ──────────────────────────────────────────────────────────────

def test_clean_series_removes_negatives():
    df = _make_series(with_negatives=True)
    df_clean, msgs = clean_series(df, "power", lower_clip=0.0)
    assert (df_clean["power"] >= 0).all()


def test_clean_series_fills_short_gaps():
    df = _make_series(with_gaps=True)
    n_missing_before = df["power"].isna().sum()
    df_clean, msgs = clean_series(df, "power", max_interpolation_gap=6)
    n_missing_after = df_clean["power"].isna().sum()
    assert n_missing_after < n_missing_before


def test_clean_series_resolves_duplicates():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    idx_dup = idx.append(idx[:2])  # 2 duplicates
    vals = np.ones(len(idx_dup))
    df = pd.DataFrame({"power": vals}, index=idx_dup).sort_index()
    df_clean, msgs = clean_series(df, "power")
    assert df_clean.index.is_unique


# ── resample_series ───────────────────────────────────────────────────────────

def test_resample_series_hourly():
    df = _make_series(n=1440, freq="1min")  # 24 hours of minute data
    df_resampled, msgs = resample_series(df, "power", freq="1h")
    assert len(df_resampled) == 24


def test_resample_series_daily():
    df = _make_series(n=24 * 30, freq="1h")  # 30 days of hourly data
    df_resampled, msgs = resample_series(df, "power", freq="1D")
    assert len(df_resampled) == 30


# ── chronological_split ───────────────────────────────────────────────────────

def test_chronological_split_no_shuffle():
    df = _make_series(n=1000)
    train, val, test = chronological_split(df, train_ratio=0.70, val_ratio=0.15)
    # Train must end before val starts
    assert train.index.max() <= val.index.min()
    assert val.index.max() <= test.index.min()


def test_chronological_split_sizes():
    df = _make_series(n=1000)
    train, val, test = chronological_split(df, train_ratio=0.70, val_ratio=0.15)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert len(train) == pytest.approx(700, abs=5)
    assert len(val)   == pytest.approx(150, abs=5)


def test_chronological_split_no_data_leakage():
    """Future timestamps must never appear in training data."""
    df = _make_series(n=500)
    train, val, test = chronological_split(df)
    assert train.index.max() < test.index.min()


# ── convert_units ─────────────────────────────────────────────────────────────

def test_convert_units_kw_to_kwh():
    s = pd.Series([2.0, 2.0, 2.0])
    result = convert_units(s, from_unit="kW", sampling_freq_hours=1.0)
    assert (result == 2.0).all()


def test_convert_units_w_to_kwh():
    s = pd.Series([2000.0, 2000.0])
    result = convert_units(s, from_unit="W", sampling_freq_hours=1.0)
    assert (result == 2.0).all()


def test_convert_units_wh_to_kwh():
    s = pd.Series([1000.0, 500.0])
    result = convert_units(s, from_unit="Wh")
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(0.5)


def test_convert_units_kwh_passthrough():
    s = pd.Series([3.14, 2.71])
    result = convert_units(s, from_unit="kWh")
    assert (result == s).all()


def test_convert_units_invalid_raises():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="Unsupported unit"):
        convert_units(s, from_unit="megawatt")
