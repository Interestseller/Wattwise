"""
app/pages/1_Dashboard.py — Dataset loading and summary dashboard.

This page allows users to:
1. Upload a CSV file or use the demo dataset
2. Configure column mappings and units
3. View dataset quality report
4. See summary statistics and key metrics
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from app.utils import (
    init_session_state, error_box, warning_box, success_box,
    info_expander, apply_layout, CHART_COLORS
)
from src.data_loader import load_from_path, load_from_upload, select_columns
from src.preprocessing import clean_series, resample_series
from data.generate_demo_data import generate_demo_dataset

# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard — Smart Home Energy",
    page_icon="📊",
    layout="wide",
)
init_session_state()

st.title("📊 Dashboard")
st.markdown("Load your energy dataset, review data quality, and explore key statistics.")

# ── Data source selector ─────────────────────────────────────────────────────
st.markdown("### 1. Dataset Source")

# Determine the default tab: if sample data is already loaded keep "Sample Data"
# selected; once the user has loaded their own data keep that tab active.
_current_source = st.session_state.get("dataset_source", "demo")
_default_tab = 1 if _current_source in ("uploaded", "path") else 0  # 0 = Sample, 1 = Own

tab_sample, tab_own = st.tabs(["🧪 Sample Data", "📂 Upload / Load Your Own Data"])

df_raw = None
profile = None
filename = ""

# ── Tab 0 : Sample data ──────────────────────────────────────────────────────
with tab_sample:
    st.info(
        "The **sample dataset** has been pre-loaded automatically — you can explore "
        "all features of the app right away without uploading anything."
    )
    st.warning(
        "⚠️ **Synthetic data**: This dataset is artificially generated, not from real household "
        "measurements. Do not draw real-world conclusions from models trained on it."
    )

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Reload sample data", help="Re-generate and reload the built-in sample dataset."):
            demo_df = generate_demo_dataset()
            demo_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "demo_energy_dataset.csv"
            )
            os.makedirs(os.path.dirname(demo_path), exist_ok=True)
            demo_df.to_csv(demo_path, index=False)
            df_raw, profile = load_from_path(demo_path)
            filename = "demo_energy_dataset.csv"
            st.session_state["dataset_source"] = "demo"
    with col_status:
        if _current_source == "demo" and st.session_state.get("df") is not None:
            st.success("✅ Sample dataset is active.")

# ── Tab 1 : User's own data ──────────────────────────────────────────────────
with tab_own:
    source_option = st.radio(
        "How would you like to load your data?",
        options=["Upload CSV file", "Load from disk path"],
        horizontal=True,
        help="Choose where to load your energy consumption data from.",
    )

    if source_option == "Upload CSV file":
        uploaded = st.file_uploader(
            "Upload your energy consumption CSV",
            type=["csv"],
            help="CSV file with at least one timestamp column and one numeric energy column.",
        )
        if uploaded:
            df_raw, profile = load_from_upload(uploaded)
            filename = uploaded.name
            st.session_state["dataset_source"] = "uploaded"

    elif source_option == "Load from disk path":
        path_input = st.text_input(
            "Dataset file path",
            value=config.DATASET_PATH,
            help="Absolute or relative path to your CSV file.",
        )
        if st.button("Load from path"):
            df_raw, profile = load_from_path(path_input)
            filename = os.path.basename(path_input)
            st.session_state["dataset_source"] = "path"

# ── Column selection ─────────────────────────────────────────────────────────
if df_raw is not None and profile is not None:
    if not profile.is_valid:
        for err in profile.errors:
            error_box(err)
    else:
        st.markdown("### 2. Configure Columns")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Guess the timestamp column
            default_ts = (
                config.TIMESTAMP_COLUMN
                if config.TIMESTAMP_COLUMN in profile.columns
                else profile.columns[0]
            )
            timestamp_col = st.selectbox(
                "Timestamp column",
                options=profile.columns,
                index=profile.columns.index(default_ts) if default_ts in profile.columns else 0,
                help="Column containing date/time values.",
            )

        with col2:
            # Guess the target column (numeric, not the timestamp)
            numeric_candidates = [
                c for c in profile.columns
                if c != timestamp_col and profile.dtypes.get(c, "") in ("float64", "int64", "float32")
            ]
            if not numeric_candidates:
                numeric_candidates = [c for c in profile.columns if c != timestamp_col]
            default_target = (
                config.TARGET_COLUMN
                if config.TARGET_COLUMN in numeric_candidates
                else (numeric_candidates[0] if numeric_candidates else profile.columns[-1])
            )
            target_col = st.selectbox(
                "Target column (energy/power)",
                options=numeric_candidates if numeric_candidates else profile.columns,
                index=(
                    numeric_candidates.index(default_target)
                    if default_target in numeric_candidates else 0
                ),
                help="Column containing the energy or power readings to predict.",
            )

        with col3:
            unit = st.selectbox(
                "Unit",
                options=config.SUPPORTED_UNITS,
                index=config.SUPPORTED_UNITS.index(config.DEFAULT_UNIT),
                help=(
                    "kW / W = instantaneous power (will be converted to kWh for billing).\n"
                    "kWh / Wh = energy already in energy units."
                ),
            )

        resample_freq = st.selectbox(
            "Resample to frequency",
            options=["1h", "15min", "30min", "1D"],
            index=0,
            help=(
                "The ML pipeline works on a uniform time grid. "
                "High-frequency data (< 1 min) is automatically resampled here."
            ),
        )

        if st.button("✅ Apply column selection & process", type="primary"):
            df_selected, errs = select_columns(df_raw, timestamp_col, target_col)

            if errs:
                for e in errs:
                    error_box(e)
            else:
                # Clean
                df_clean, clean_msgs = clean_series(df_selected, target_col)
                # Resample
                df_resampled, resample_msgs = resample_series(df_clean, target_col, freq=resample_freq)
                # Drop remaining NaN in target
                df_resampled = df_resampled.dropna(subset=[target_col])

                st.session_state["df"]             = df_resampled
                st.session_state["df_raw"]         = df_raw
                st.session_state["profile"]        = profile
                st.session_state["timestamp_col"]  = timestamp_col
                st.session_state["target_col"]     = target_col
                st.session_state["energy_unit"]    = unit
                st.session_state["resample_freq"]  = resample_freq

                success_box(f"Dataset processed: {len(df_resampled):,} rows at '{resample_freq}' frequency.")
                for msg in clean_msgs + resample_msgs:
                    st.caption(f"  • {msg}")

# ── Dataset report ────────────────────────────────────────────────────────────
if st.session_state.get("df") is not None:
    df = st.session_state["df"]
    target = st.session_state.get("target_col", config.TARGET_COLUMN)
    unit   = st.session_state.get("energy_unit", config.DEFAULT_UNIT)
    profile = st.session_state.get("profile")

    st.divider()
    st.markdown("### 3. Dataset Summary")

    # Summary cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Records", f"{len(df):,}")
    with c2:
        st.metric("Columns", str(len(df.columns) + 1))  # +1 for index
    with c3:
        date_start = df.index.min().strftime("%Y-%m-%d")
        st.metric("Start date", date_start)
    with c4:
        date_end = df.index.max().strftime("%Y-%m-%d")
        st.metric("End date", date_end)
    with c5:
        n_days = (df.index.max() - df.index.min()).days
        st.metric("Days covered", str(n_days))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(f"Mean {target}", f"{df[target].mean():.3f} {unit}")
    with c2:
        st.metric(f"Max {target}", f"{df[target].max():.3f} {unit}")
    with c3:
        missing_pct = df[target].isna().mean() * 100
        st.metric("Missing values", f"{missing_pct:.1f}%")
    with c4:
        resample_freq = st.session_state.get("resample_freq", "1h")
        st.metric("Sampling freq", resample_freq)

    # Warnings from profile
    if profile and profile.warnings:
        st.markdown("#### ⚠️ Data Quality Warnings")
        for w in profile.warnings:
            st.warning(w)

    # Quick consumption chart
    st.markdown("#### Consumption Overview")
    # For a quick overview, resample to daily if hourly
    if resample_freq in ("1h", "15min", "30min"):
        display_df = df[[target]].resample("1D").mean()
        agg_label = "Daily average"
    else:
        display_df = df[[target]]
        agg_label = target

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_df.index,
        y=display_df[target],
        mode="lines",
        name=agg_label,
        line=dict(color=CHART_COLORS["actual"], width=1.5),
    ))
    apply_layout(fig, title=f"Historical Energy Consumption ({unit})", height=320)
    fig.update_yaxes(title_text=f"{target} ({unit})")
    st.plotly_chart(fig, use_container_width=True)

    # Missing values detail
    with st.expander("Missing values per column"):
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values.")
        else:
            st.dataframe(
                missing.rename("Missing count").to_frame(),
                use_container_width=True,
            )

    # Dataset info box
    info_expander(
        "About this dataset profile",
        """
The dashboard shows a **data quality profile** computed from your loaded dataset.

**What the numbers mean:**
- **Records**: Total number of time steps after resampling.
- **Missing values**: Percentage of target values that were NaN after interpolation.
- **Sampling freq**: The uniform time step the data was resampled to.

A good dataset for this application has:
- At least **2 weeks** of data (ideally 3+ months)
- Less than **10% missing values** in the target column
- A **consistent sampling frequency** (e.g. always hourly)
        """
    )

    # Raw data preview
    with st.expander("Preview processed data (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)

