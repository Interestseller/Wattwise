"""
app/utils.py — Shared helper functions for the Streamlit application.

This module provides:
- Session state management (persistent data across page interactions)
- Chart helpers (consistent Plotly theme)
- Number formatting utilities
- Dataset caching wrappers
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _load_demo_into_session():
    """Load and process the demo dataset into session state (silent, no UI)."""
    try:
        # Import here to avoid circular imports at module level
        from data.generate_demo_data import generate_demo_dataset
        from src.data_loader import load_from_path
        from src.preprocessing import clean_series, resample_series

        demo_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "demo_energy_dataset.csv"
        )
        # Generate the CSV if it doesn't exist yet
        if not os.path.exists(demo_path):
            demo_df = generate_demo_dataset()
            os.makedirs(os.path.dirname(demo_path), exist_ok=True)
            demo_df.to_csv(demo_path, index=False)

        df_raw, profile = load_from_path(demo_path)
        if profile is None or not profile.is_valid:
            return

        ts_col  = config.TIMESTAMP_COLUMN
        tgt_col = config.TARGET_COLUMN
        freq    = config.DEFAULT_RESAMPLE_FREQ

        from src.data_loader import select_columns
        df_sel, errs = select_columns(df_raw, ts_col, tgt_col)
        if errs:
            return

        df_clean, _ = clean_series(df_sel, tgt_col)
        df_resampled, _ = resample_series(df_clean, tgt_col, freq=freq)
        df_resampled = df_resampled.dropna(subset=[tgt_col])

        st.session_state["df"]            = df_resampled
        st.session_state["df_raw"]        = df_raw
        st.session_state["profile"]       = profile
        st.session_state["timestamp_col"] = ts_col
        st.session_state["target_col"]    = tgt_col
        st.session_state["energy_unit"]   = config.DEFAULT_UNIT
        st.session_state["resample_freq"] = freq
        st.session_state["dataset_source"] = "demo"
    except Exception:
        # Never crash the app on startup because of demo data
        pass


def init_session_state():
    """Initialise all session-state keys with safe defaults.

    On the very first run the demo dataset is automatically pre-loaded so
    users can explore the app immediately without uploading anything.
    """
    defaults = {
        "df":              None,   # processed DataFrame with DatetimeIndex
        "df_raw":          None,   # raw loaded DataFrame
        "profile":         None,   # DataProfile object
        "timestamp_col":   config.TIMESTAMP_COLUMN,
        "target_col":      config.TARGET_COLUMN,
        "energy_unit":     config.DEFAULT_UNIT,
        "resample_freq":   config.DEFAULT_RESAMPLE_FREQ,
        "model_bundle":    None,   # {model, scaler, feature_names, target_col}
        "train_result":    None,   # full result dict from train_and_select_model
        "tariff":          config.DEFAULT_TARIFF_PER_KWH,
        "fixed_charge":    config.DEFAULT_FIXED_MONTHLY_CHARGE,
        "tax_rate":        config.DEFAULT_TAX_RATE,
        "currency":        config.CURRENCY_SYMBOL,
        "forecast_df":     None,
        "anomaly_df":      None,
        "dataset_source":  "none",  # "uploaded", "path", "demo"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Auto-load sample data on first visit (only when no dataset is loaded yet)
    if st.session_state["df"] is None:
        _load_demo_into_session()


def set_dataset(df: pd.DataFrame, source: str = "uploaded"):
    """Store a processed DataFrame and mark the source."""
    st.session_state["df"] = df
    st.session_state["dataset_source"] = source


def get_dataset() -> Optional[pd.DataFrame]:
    return st.session_state.get("df")


def has_dataset() -> bool:
    return st.session_state.get("df") is not None and len(st.session_state["df"]) > 0


def has_trained_model() -> bool:
    return st.session_state.get("model_bundle") is not None


def get_target_col() -> str:
    return st.session_state.get("target_col", config.TARGET_COLUMN)


def get_energy_unit() -> str:
    return st.session_state.get("energy_unit", config.DEFAULT_UNIT)


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def fmt(value: float, decimals: int = 2, unit: str = "") -> str:
    """Format a number with optional unit."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{value:,.{decimals}f}{suffix}"


def fmt_delta(current: float, previous: float) -> tuple[float, str]:
    """Compute percentage change and direction arrow."""
    if previous == 0:
        return 0.0, "→"
    pct = (current - previous) / abs(previous) * 100
    arrow = "↑" if pct > 0 else "↓"
    return round(pct, 1), arrow


# ---------------------------------------------------------------------------
# Chart theme
# ---------------------------------------------------------------------------

CHART_COLORS = {
    "actual":    "#2563eb",   # blue
    "predicted": "#f97316",   # orange
    "forecast":  "#16a34a",   # green
    "anomaly":   "#dc2626",   # red
    "baseline":  "#9333ea",   # purple
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, sans-serif", size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
)


def apply_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    """Apply the standard chart layout."""
    layout = {**LAYOUT_DEFAULTS, **kwargs}
    if title:
        layout["title"] = dict(text=title, font=dict(size=15))
    fig.update_layout(**layout)
    return fig


def line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: dict,   # {display_name: column_name_or_series}
    title: str = "",
    y_axis_label: str = "",
    height: int = 380,
) -> go.Figure:
    """Create a consistent multi-line Plotly chart."""
    fig = go.Figure()
    colors = list(CHART_COLORS.values())
    for i, (name, col) in enumerate(y_cols.items()):
        if isinstance(col, pd.Series):
            y_data = col
            x_data = col.index
        else:
            y_data = df[col]
            x_data = df[x_col] if x_col in df.columns else df.index
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, name=name,
            line=dict(color=color, width=2),
            mode="lines",
        ))
    apply_layout(fig, title=title, height=height)
    if y_axis_label:
        fig.update_yaxes(title_text=y_axis_label)
    return fig


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_aggregated(
    df: pd.DataFrame,
    target_col: str,
    freq: str = "1D",
) -> pd.DataFrame:
    """Resample a DataFrame to the given frequency for display."""
    return df[[target_col]].resample(freq).mean()


def compute_daily_pattern(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Average consumption by hour of day."""
    return df.groupby(df.index.hour)[target_col].mean()


def compute_weekly_pattern(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Average consumption by day of week (0=Mon)."""
    return df.groupby(df.index.dayofweek)[target_col].mean()


def compute_monthly_totals(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Monthly sum for bar charts (assumes kWh data)."""
    monthly = df[[target_col]].resample("MS").sum()
    monthly.index = monthly.index.strftime("%b %Y")
    return monthly


# ---------------------------------------------------------------------------
# Streamlit UI helpers
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str = "", help_text: str = ""):
    """Render a styled metric card using st.metric."""
    st.metric(label=label, value=value, delta=delta, help=help_text)


def error_box(message: str):
    """Show a user-friendly error (no technical traceback)."""
    st.error(f"⚠️ {message}")


def warning_box(message: str):
    st.warning(f"ℹ️ {message}")


def success_box(message: str):
    st.success(f"✅ {message}")


def info_expander(title: str, content: str):
    """Create an expandable info section."""
    with st.expander(f"ℹ️ {title}"):
        st.markdown(content)
