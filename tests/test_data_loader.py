"""
tests/test_data_loader.py — Tests for the data loading module.
"""

import io
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_from_path, load_from_upload, select_columns


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_csv_bytes(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode("utf-8"))


class MockUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        return self._content


MINIMAL_CSV = (
    "DateTime,Global_active_power\n"
    "2024-01-01 00:00:00,1.25\n"
    "2024-01-01 01:00:00,1.18\n"
    "2024-01-01 02:00:00,1.42\n"
)

MISSING_VALUE_CSV = (
    "DateTime,Global_active_power\n"
    "2024-01-01 00:00:00,1.25\n"
    "2024-01-01 01:00:00,\n"
    "2024-01-01 02:00:00,1.42\n"
)

NO_TIMESTAMP_CSV = (
    "Global_active_power\n"
    "1.25\n1.18\n1.42\n"
)

EMPTY_CSV = "DateTime,Global_active_power\n"


# ── Tests: load_from_path ─────────────────────────────────────────────────────

def test_load_from_path_missing_file():
    df, profile = load_from_path("/nonexistent/path/to/file.csv")
    assert df.empty
    assert not profile.is_valid
    assert any("not found" in e.lower() for e in profile.errors)


def test_load_from_path_valid_file(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(MINIMAL_CSV)
    df, profile = load_from_path(str(csv_file))
    assert not df.empty
    assert profile.n_rows == 3
    assert "Global_active_power" in profile.columns
    assert profile.is_valid


def test_load_from_path_empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text(EMPTY_CSV)
    df, profile = load_from_path(str(csv_file))
    # Either empty df or error
    assert df.empty or not profile.is_valid or len(df) == 0


# ── Tests: load_from_upload ───────────────────────────────────────────────────

def test_load_from_upload_minimal():
    uploaded = MockUploadedFile("test.csv", MINIMAL_CSV.encode())
    df, profile = load_from_upload(uploaded)
    assert not df.empty
    assert profile.n_rows == 3
    assert profile.filename == "test.csv"


def test_load_from_upload_invalid_bytes():
    uploaded = MockUploadedFile("bad.csv", b"NOT,VALID,CSV\x00\x01\x02")
    df, profile = load_from_upload(uploaded)
    # Should not crash; may return error in profile
    assert isinstance(profile.errors, list) or df is not None


# ── Tests: select_columns ──────────────────────────────────────────────────────

def test_select_columns_valid():
    uploaded = MockUploadedFile("test.csv", MINIMAL_CSV.encode())
    df_raw, _ = load_from_upload(uploaded)
    df_clean, errors = select_columns(df_raw, "DateTime", "Global_active_power")
    assert len(errors) == 0 or errors == []
    assert isinstance(df_clean.index, pd.DatetimeIndex)
    assert "Global_active_power" in df_clean.columns


def test_select_columns_missing_timestamp():
    uploaded = MockUploadedFile("test.csv", MINIMAL_CSV.encode())
    df_raw, _ = load_from_upload(uploaded)
    _, errors = select_columns(df_raw, "NonExistentCol", "Global_active_power")
    assert len(errors) > 0


def test_select_columns_missing_target():
    uploaded = MockUploadedFile("test.csv", MINIMAL_CSV.encode())
    df_raw, _ = load_from_upload(uploaded)
    _, errors = select_columns(df_raw, "DateTime", "NonExistentTarget")
    assert len(errors) > 0


def test_select_columns_with_missing_values():
    uploaded = MockUploadedFile("test.csv", MISSING_VALUE_CSV.encode())
    df_raw, _ = load_from_upload(uploaded)
    df_clean, errors = select_columns(df_raw, "DateTime", "Global_active_power")
    # Should succeed with warning, not fail
    assert isinstance(df_clean, pd.DataFrame)


# ── Tests: timestamp parsing ───────────────────────────────────────────────────

def test_timestamp_parsing_iso():
    csv = "ts,value\n2024-01-01 00:00:00,1.0\n2024-01-01 01:00:00,2.0\n"
    uploaded = MockUploadedFile("t.csv", csv.encode())
    df_raw, _ = load_from_upload(uploaded)
    df_clean, errors = select_columns(df_raw, "ts", "value")
    assert isinstance(df_clean.index, pd.DatetimeIndex)
    assert len(df_clean) == 2


def test_timestamp_parsing_date_only():
    csv = "date,power\n2024-01-01,100.0\n2024-01-02,120.0\n"
    uploaded = MockUploadedFile("t.csv", csv.encode())
    df_raw, _ = load_from_upload(uploaded)
    df_clean, errors = select_columns(df_raw, "date", "power")
    assert isinstance(df_clean.index, pd.DatetimeIndex)
