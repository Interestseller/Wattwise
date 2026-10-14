"""
app/pages/2_Analytics.py — Interactive consumption analytics and charts.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from app.utils import (
    init_session_state, apply_layout, CHART_COLORS,
    compute_daily_pattern, compute_weekly_pattern, compute_monthly_totals,
)

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")
init_session_state()

st.title("📈 Energy Consumption Analytics")

if st.session_state.get("df") is None:
    st.info("📊 No dataset loaded. Go to **Dashboard** to load your data first.")
    st.stop()

df     = st.session_state["df"]
target = st.session_state.get("target_col", config.TARGET_COLUMN)
unit   = st.session_state.get("energy_unit", config.DEFAULT_UNIT)

# ── Date range selector ──────────────────────────────────────────────────────
st.markdown("#### 🗓️ Date Range Filter")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "From", value=df.index.min().date(), min_value=df.index.min().date()
    )
with col2:
    end_date = st.date_input(
        "To", value=df.index.max().date(), max_value=df.index.max().date()
    )

mask = (df.index.date >= start_date) & (df.index.date <= end_date)
df_filtered = df[mask]

if df_filtered.empty:
    st.warning("No data in the selected date range.")
    st.stop()

st.caption(f"Showing {len(df_filtered):,} rows from {start_date} to {end_date}")

# ────────────────────────────────────────────────────────────────────────────
# A. Historical Consumption Line Chart
# ────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### A. Historical Consumption")

agg_choice = st.radio(
    "Aggregation", ["Hourly (raw)", "Daily average", "Weekly average", "Monthly average"],
    horizontal=True,
)

agg_map = {
    "Hourly (raw)":       None,
    "Daily average":      "1D",
    "Weekly average":     "1W",
    "Monthly average":    "MS",
}
agg_freq = agg_map[agg_choice]

if agg_freq:
    display_df = df_filtered[[target]].resample(agg_freq).mean()
else:
    display_df = df_filtered[[target]]

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=display_df.index, y=display_df[target],
    mode="lines", name=f"{target} ({agg_choice})",
    line=dict(color=CHART_COLORS["actual"], width=1.5),
    fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.08)",
))
apply_layout(fig_hist, title=f"Historical Energy Consumption ({unit})", height=380)
fig_hist.update_yaxes(title_text=f"{unit}")
st.plotly_chart(fig_hist, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# B. Actual vs Predicted (if model trained)
# ────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### B. Actual vs Predicted")

train_result = st.session_state.get("train_result")
if train_result is None:
    st.info("Train a model on the **🤖 Train ML Model** page to see actual vs predicted.")
else:
    test_df    = train_result.get("test_df")
    y_test     = train_result.get("y_test")
    test_preds = train_result.get("test_predictions", {})
    best_name  = train_result.get("best_model_name", "best model")

    if test_df is not None and y_test is not None:
        # Choose which model to display
        model_names = list(test_preds.keys())
        chosen_model = st.selectbox("Model to display", model_names, index=model_names.index(best_name) if best_name in model_names else 0)

        preds = test_preds.get(chosen_model)
        if preds is not None:
            fig_avp = go.Figure()
            fig_avp.add_trace(go.Scatter(
                x=y_test.index, y=y_test.values,
                mode="lines", name="Actual",
                line=dict(color=CHART_COLORS["actual"], width=2),
            ))
            fig_avp.add_trace(go.Scatter(
                x=y_test.index, y=preds,
                mode="lines", name=f"Predicted ({chosen_model})",
                line=dict(color=CHART_COLORS["predicted"], width=2, dash="dash"),
            ))
            apply_layout(fig_avp, title="Actual vs Predicted (Test Set)", height=380)
            fig_avp.update_yaxes(title_text=unit)
            st.plotly_chart(fig_avp, use_container_width=True)

            # Residual plot
            with st.expander("Residuals (errors)"):
                residuals = y_test.values - preds
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=y_test.index, y=residuals,
                    mode="lines+markers",
                    name="Residual",
                    line=dict(color="#f97316", width=1),
                    marker=dict(size=3),
                ))
                fig_res.add_hline(y=0, line_dash="dash", line_color="grey")
                apply_layout(fig_res, title="Prediction Residuals (actual − predicted)", height=280)
                fig_res.update_yaxes(title_text=unit)
                st.plotly_chart(fig_res, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# C. Daily & Weekly Patterns
# ────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### C. Consumption Patterns")

pat_col1, pat_col2 = st.columns(2)

with pat_col1:
    st.markdown("**By hour of day**")
    hourly_pattern = compute_daily_pattern(df_filtered, target)
    fig_hour = go.Figure(go.Bar(
        x=hourly_pattern.index,
        y=hourly_pattern.values,
        marker_color=CHART_COLORS["actual"],
    ))
    apply_layout(fig_hour, title="Average Consumption by Hour of Day", height=300)
    fig_hour.update_xaxes(title_text="Hour")
    fig_hour.update_yaxes(title_text=f"Avg {unit}")
    st.plotly_chart(fig_hour, use_container_width=True)

with pat_col2:
    st.markdown("**By day of week**")
    weekly_pattern = compute_weekly_pattern(df_filtered, target)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_week = go.Figure(go.Bar(
        x=[day_names[i] for i in weekly_pattern.index],
        y=weekly_pattern.values,
        marker_color=[
            "#ef4444" if i >= 5 else CHART_COLORS["actual"]
            for i in weekly_pattern.index
        ],
    ))
    apply_layout(fig_week, title="Average Consumption by Day of Week", height=300)
    fig_week.update_xaxes(title_text="Day")
    fig_week.update_yaxes(title_text=f"Avg {unit}")
    st.plotly_chart(fig_week, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# D. Consumption Distribution
# ────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### D. Consumption Distribution")

# Check for sub-meter / appliance columns
all_cols = [c for c in df_filtered.columns if c != target]
sub_cols = [c for c in all_cols if any(kw in c.lower() for kw in ["sub", "appliance", "meter"])]

if sub_cols:
    st.markdown("**Energy by appliance / sub-meter**")
    col_sums = {c: df_filtered[c].sum() for c in sub_cols if pd.api.types.is_numeric_dtype(df_filtered[c])}
    if col_sums:
        fig_pie = go.Figure(go.Pie(
            labels=list(col_sums.keys()),
            values=list(col_sums.values()),
            hole=0.4,
        ))
        apply_layout(fig_pie, title="Energy by Appliance", height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.markdown("**Consumption by time of day segment**")
    df_tmp = df_filtered[[target]].copy()
    df_tmp["hour"] = df_tmp.index.hour
    labels = {0: "Night (0–5)", 1: "Morning (6–11)", 2: "Afternoon (12–17)", 3: "Evening (18–23)"}
    bins = [-1, 5, 11, 17, 23]
    df_tmp["period"] = pd.cut(df_tmp["hour"], bins=bins, labels=list(labels.values()))
    period_sums = df_tmp.groupby("period")[target].mean()

    fig_pie = go.Figure(go.Pie(
        labels=period_sums.index.tolist(),
        values=period_sums.values,
        hole=0.4,
        marker=dict(colors=["#1e40af", "#3b82f6", "#93c5fd", "#f97316"]),
    ))
    apply_layout(fig_pie, title="Average Consumption by Time of Day", height=320)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption(
        "ℹ️ No appliance-level data detected. "
        "Showing consumption distribution by time-of-day period instead. "
        "For appliance breakdown, your dataset needs separate appliance columns."
    )

# ────────────────────────────────────────────────────────────────────────────
# E. Monthly Comparison
# ────────────────────────────────────────────────────────────────────────────
n_days = (df_filtered.index.max() - df_filtered.index.min()).days
if n_days >= 60:
    st.divider()
    st.markdown("### E. Monthly Comparison")
    monthly = df_filtered[[target]].resample("MS").sum()

    # If we have test predictions spanning months
    monthly_pred = None
    if train_result and train_result.get("y_test") is not None:
        y_test = train_result["y_test"]
        preds_arr = train_result["test_predictions"].get(train_result.get("best_model_name", ""))
        if preds_arr is not None:
            pred_series = pd.Series(preds_arr, index=y_test.index, name="predicted")
            monthly_pred = pred_series.resample("MS").sum()

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=monthly.index.strftime("%b %Y"),
        y=monthly[target],
        name="Actual",
        marker_color=CHART_COLORS["actual"],
    ))
    if monthly_pred is not None:
        common_months = monthly.index.intersection(monthly_pred.index)
        fig_bar.add_trace(go.Bar(
            x=monthly_pred.loc[common_months].index.strftime("%b %Y"),
            y=monthly_pred.loc[common_months].values,
            name="Predicted (test set)",
            marker_color=CHART_COLORS["predicted"],
        ))

    apply_layout(fig_bar, title=f"Monthly Energy Consumption ({unit})", height=340)
    fig_bar.update_layout(barmode="group")
    fig_bar.update_yaxes(title_text=unit)
    st.plotly_chart(fig_bar, use_container_width=True)
