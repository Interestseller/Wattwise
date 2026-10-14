"""
app/pages/6_Anomaly_Detection.py — Detect unusual energy consumption patterns.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from app.utils import init_session_state, apply_layout, CHART_COLORS, error_box
from src.anomaly_detection import detect_anomalies, summarise_anomalies

st.set_page_config(page_title="Anomaly Detection", page_icon="⚠️", layout="wide")
init_session_state()

st.title("⚠️ Unusual Consumption Detection")
st.markdown(
    "Identify time periods with unusual energy consumption patterns. "
    "**Note: flagged events are unusual consumption patterns, NOT confirmed electrical faults.**"
)

if st.session_state.get("df") is None:
    st.info("📊 Load a dataset on the **Dashboard** page first.")
    st.stop()

df     = st.session_state["df"]
target = st.session_state.get("target_col", config.TARGET_COLUMN)
unit   = st.session_state.get("energy_unit", config.DEFAULT_UNIT)

# ── Configuration ─────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Detection Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    method = st.selectbox(
        "Detection method",
        options=["isolation_forest", "zscore", "both"],
        format_func=lambda x: {
            "isolation_forest": "Isolation Forest (ML-based)",
            "zscore": "Rolling Z-Score (statistical)",
            "both": "Both methods (flag if either detects)",
        }[x],
        help=(
            "Isolation Forest: unsupervised ML algorithm. "
            "Z-Score: statistical deviation from rolling mean."
        ),
    )

with col2:
    contamination = st.slider(
        "Isolation Forest contamination",
        min_value=0.01, max_value=0.15,
        value=config.ANOMALY_CONTAMINATION,
        step=0.01,
        format="%.2f",
        help="Expected fraction of observations that are anomalies.",
    )

with col3:
    zscore_threshold = st.slider(
        "Z-Score threshold",
        min_value=1.5, max_value=5.0,
        value=config.ZSCORE_THRESHOLD,
        step=0.1,
        format="%.1f",
        help="Readings more than N standard deviations from the rolling mean are flagged.",
    )

rolling_window = st.slider(
    "Rolling window (time steps)",
    min_value=6, max_value=168,
    value=config.ROLLING_ZSCORE_WINDOW,
    help="Window size for rolling mean/std calculation.",
)

# ── Date range ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("From", value=df.index.min().date())
with col2:
    end_date = st.date_input("To",   value=df.index.max().date())

mask = (df.index.date >= start_date) & (df.index.date <= end_date)
df_filtered = df[mask]

if len(df_filtered) < 20:
    st.warning("Select a longer date range (minimum 20 observations needed).")
    st.stop()

st.caption(f"Analysing {len(df_filtered):,} observations from {start_date} to {end_date}")

# ── Run detection ─────────────────────────────────────────────────────────────
if st.button("🔍 Detect Anomalies", type="primary"):
    try:
        series = df_filtered[target].dropna()
        with st.spinner("Detecting unusual patterns…"):
            anomaly_df = detect_anomalies(
                series,
                method=method,
                contamination=contamination,
                zscore_threshold=zscore_threshold,
                rolling_window=rolling_window,
            )
        st.session_state["anomaly_df"] = anomaly_df
        st.session_state["anomaly_series"] = series
        st.success(f"Detection complete: {anomaly_df['is_anomaly'].sum()} unusual observations flagged.")
    except Exception as exc:
        error_box(str(exc))
        st.stop()

# ── Display results ────────────────────────────────────────────────────────────
anomaly_df = st.session_state.get("anomaly_df")
if anomaly_df is None:
    st.info("Configure detection above and click **Detect Anomalies** to see results.")
    st.stop()

summary = summarise_anomalies(anomaly_df)

st.divider()
st.markdown("### 📊 Detection Results")

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total observations", f"{len(anomaly_df):,}")
with c2: st.metric("Flagged as unusual", f"{summary['total_anomalies']:,}")
with c3: st.metric("% flagged", f"{summary['pct_anomalous']:.1f}%")
with c4:
    if summary["worst_deviation"] is not None:
        st.metric("Max deviation", f"{summary['worst_deviation']:+.3f} {unit}")
    else:
        st.metric("Max deviation", "—")

# ── Consumption chart with anomalies highlighted ────────────────────────────
anomalies_only = anomaly_df[anomaly_df["is_anomaly"] == True]  # noqa

fig = go.Figure()

# Normal consumption
fig.add_trace(go.Scatter(
    x=anomaly_df.index,
    y=anomaly_df["value"],
    mode="lines",
    name="Consumption",
    line=dict(color=CHART_COLORS["actual"], width=1.5),
))

# Expected baseline (rolling mean)
if "expected" in anomaly_df.columns:
    fig.add_trace(go.Scatter(
        x=anomaly_df.index,
        y=anomaly_df["expected"],
        mode="lines",
        name="Rolling baseline",
        line=dict(color="#6b7280", width=1, dash="dot"),
    ))

# Anomaly markers
if not anomalies_only.empty:
    fig.add_trace(go.Scatter(
        x=anomalies_only.index,
        y=anomalies_only["value"],
        mode="markers",
        name="Unusual pattern",
        marker=dict(
            color=CHART_COLORS["anomaly"],
            size=9, symbol="x",
            line=dict(width=2, color="darkred"),
        ),
    ))

apply_layout(fig, title="Energy Consumption with Unusual Patterns Highlighted", height=420)
fig.update_yaxes(title_text=unit)
st.plotly_chart(fig, use_container_width=True)

# ── Top anomalies table ────────────────────────────────────────────────────────
if summary["total_anomalies"] > 0:
    st.markdown("### 🔝 Top Unusual Events")
    top_df = summary["top_anomalies"].copy()

    display_cols = ["value", "expected", "deviation"]
    if "anomaly_score" in top_df.columns:
        display_cols.append("anomaly_score")
    if "method" in top_df.columns:
        display_cols.append("method")

    display_df = top_df[display_cols].copy()
    display_df.index.name = "Timestamp"
    display_df.columns = [
        "Measured", "Expected baseline", "Deviation",
        *display_cols[3:],
    ]
    st.dataframe(display_df.round(4), use_container_width=True)

    # Anomaly score distribution
    if "anomaly_score" in anomaly_df.columns:
        with st.expander("Anomaly score distribution"):
            fig_hist = go.Figure(go.Histogram(
                x=anomaly_df["anomaly_score"].dropna(),
                nbinsx=50,
                marker_color=CHART_COLORS["actual"],
                opacity=0.75,
            ))
            apply_layout(fig_hist, title="Distribution of Anomaly Scores", height=260)
            fig_hist.update_xaxes(title_text="Score (higher = more unusual)")
            st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.success("No unusual consumption patterns detected with the current settings.")

# ── Method explanations ───────────────────────────────────────────────────────
with st.expander("ℹ️ How does anomaly detection work?"):
    st.markdown("""
### Isolation Forest
A machine learning algorithm that builds many random decision trees. Points that
are easier to "isolate" (reach the bottom of a tree quickly) are more anomalous.
- **Advantage**: Works without assuming a specific distribution
- **Limitation**: The `contamination` parameter must be tuned; too high flags normal events

### Rolling Z-Score
For each time step, computes:
```
Z = (value - rolling_mean) / rolling_std
```
A Z-score above the threshold (e.g. 3.0) means the reading is 3+ standard deviations
from recent normal.
- **Advantage**: Easy to interpret ("3 sigma above normal")
- **Limitation**: Assumes recent history represents "normal"; doesn't work well during long unusual periods

### Interpreting results
- Flagged events are **unusual patterns**, not confirmed faults
- Common causes: holiday periods, weather extremes, meter errors, or genuinely unusual usage
- Use your domain knowledge to decide whether each flagged event is truly problematic
    """)
