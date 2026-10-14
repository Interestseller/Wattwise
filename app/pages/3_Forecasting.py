"""
app/pages/3_Forecasting.py — Multi-step energy forecasting page.
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
from src.predict_energy import forecast, estimate_uncertainty
from src.train_model import load_model

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")
init_session_state()

st.title("🔮 Energy Usage Forecasting")
st.markdown(
    "Generate future energy consumption forecasts from a trained ML model. "
    "Forecasts use **recursive prediction** — each predicted value feeds back as an input for the next step."
)

# ── Prerequisites check ───────────────────────────────────────────────────────
if st.session_state.get("df") is None:
    st.info("📊 Load a dataset on the **Dashboard** page first.")
    st.stop()

# Try to load model from session state, or from disk
bundle = st.session_state.get("model_bundle")
if bundle is None:
    try:
        bundle = load_model()
        st.session_state["model_bundle"] = bundle
    except FileNotFoundError:
        st.info(
            "🤖 No trained model found. "
            "Go to **🤖 Train ML Model** to train one first."
        )
        st.stop()

df     = st.session_state["df"]
target = st.session_state.get("target_col", config.TARGET_COLUMN)
unit   = st.session_state.get("energy_unit", config.DEFAULT_UNIT)
freq   = st.session_state.get("resample_freq", "1h")

model         = bundle["model"]
scaler        = bundle["scaler"]
feature_names = bundle["feature_names"]
model_name    = bundle.get("model_name", "trained model")

st.success(f"🤖 Using model: **{model_name}** | Dataset: {len(df):,} rows | Freq: `{freq}`")

# ── Forecast configuration ────────────────────────────────────────────────────
st.markdown("### ⚙️ Forecast Configuration")

col1, col2 = st.columns(2)
with col1:
    horizon_label = st.selectbox(
        "Forecast horizon",
        options=list(config.FORECAST_HORIZONS.keys()),
        index=0,
    )
    horizon_steps = config.FORECAST_HORIZONS[horizon_label]

    # Check if we have enough history
    min_required = config.MIN_HISTORY_FOR_HORIZON.get(horizon_steps, 48)
    if len(df) < min_required:
        st.warning(
            f"'{horizon_label}' requires at least {min_required} historical observations. "
            f"Dataset has {len(df)}. Consider a shorter horizon or more data."
        )

with col2:
    show_uncertainty = st.checkbox(
        "Show uncertainty band",
        value=True,
        help=(
            "Adds a heuristic uncertainty band based on historical volatility. "
            "This is NOT a statistically guaranteed prediction interval — use as a visual guide only."
        ),
    )
    history_window = st.slider(
        "History window to display (days)",
        min_value=7, max_value=90, value=30,
        help="How many days of historical data to show alongside the forecast.",
    )

# ── Generate forecast ─────────────────────────────────────────────────────────
if st.button("🔮 Generate Forecast", type="primary"):
    try:
        history_series = df[target].dropna()

        with st.spinner(f"Generating {horizon_label} forecast…"):
            forecast_df = forecast(
                model=model,
                scaler=scaler,
                feature_names=feature_names,
                history=history_series,
                steps=horizon_steps,
                freq=freq,
                lag_periods=config.LAG_PERIODS,
                rolling_windows=config.ROLLING_WINDOWS,
            )

            if show_uncertainty:
                forecast_df = estimate_uncertainty(forecast_df, history_series, confidence=0.90)

        st.session_state["forecast_df"] = forecast_df
        st.success(f"Forecast generated: {len(forecast_df)} steps ({horizon_label})")

    except Exception as exc:
        error_box(f"Forecasting failed: {exc}")
        st.stop()

# ── Display forecast ───────────────────────────────────────────────────────────
forecast_df = st.session_state.get("forecast_df")
if forecast_df is None:
    st.info("Configure the forecast above and click **Generate Forecast** to see results.")
    st.stop()

st.divider()
st.markdown("### 📊 Forecast Results")

# Metadata
meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
with meta_c1:
    st.metric("Horizon", horizon_label if "forecast_df" in st.session_state else "—")
with meta_c2:
    st.metric("Steps", f"{len(forecast_df)}")
with meta_c3:
    total_pred = forecast_df["predicted"].sum()
    st.metric(f"Total forecast ({unit})", f"{total_pred:.2f}")
with meta_c4:
    avg_pred = forecast_df["predicted"].mean()
    st.metric(f"Average ({unit})", f"{avg_pred:.3f}")

# Forecast chart — combine history + forecast
history_series = df[target].dropna()
history_window_steps = min(history_window * 24, len(history_series))  # approx for hourly
history_display = history_series.iloc[-history_window_steps:]

fig = go.Figure()

# Historical trace
fig.add_trace(go.Scatter(
    x=history_display.index,
    y=history_display.values,
    name="Historical",
    line=dict(color=CHART_COLORS["actual"], width=2),
    mode="lines",
))

# Uncertainty band
if "lower" in forecast_df.columns and "upper" in forecast_df.columns:
    fig.add_trace(go.Scatter(
        x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
        y=forecast_df["upper"].tolist() + forecast_df["lower"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(22, 163, 74, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Uncertainty band (90%)",
        hoverinfo="skip",
    ))

# Forecast trace
fig.add_trace(go.Scatter(
    x=forecast_df.index,
    y=forecast_df["predicted"],
    name=f"Forecast ({model_name})",
    line=dict(color=CHART_COLORS["forecast"], width=2.5, dash="dot"),
    mode="lines",
))

# Mark the forecast boundary
boundary_ts = history_display.index[-1]
fig.add_vline(
    x=boundary_ts, line_dash="dash", line_color="#9333ea",
    annotation_text="Forecast starts", annotation_position="top right",
)

apply_layout(fig, title=f"Energy Forecast — {horizon_label}", height=450)
fig.update_yaxes(title_text=unit)
st.plotly_chart(fig, use_container_width=True)

# Forecast data table
with st.expander("View forecast data table"):
    display_cols = ["predicted"]
    if "lower" in forecast_df.columns:
        display_cols += ["lower", "upper"]
    st.dataframe(
        forecast_df[display_cols].round(4),
        use_container_width=True,
    )

# Download forecast
csv_data = forecast_df[["predicted"]].to_csv()
st.download_button(
    "⬇️ Download forecast as CSV",
    data=csv_data,
    file_name="energy_forecast.csv",
    mime="text/csv",
)

# Forecasting explanation
with st.expander("ℹ️ How does recursive forecasting work?"):
    st.markdown(f"""
**Recursive (iterated) forecasting** is used because our model predicts one time step ahead:

```
y[t+1] = model(features_at_t)
```

To forecast H steps ahead:
1. Predict step 1 using real historical data as lag features
2. Use the step 1 prediction as `lag_1` to predict step 2
3. Continue until all {len(forecast_df)} steps are predicted

**What this means for accuracy:**
- Steps close to "now" are typically more accurate
- Uncertainty grows with forecast horizon as errors accumulate
- The uncertainty band above shows this growth (wider = less certain)

**Model used:** {model_name}
**Feature set:** {len(feature_names)} features including calendar and lag features
    """)
