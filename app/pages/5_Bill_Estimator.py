"""
app/pages/5_Bill_Estimator.py — Electricity bill estimator and energy insights.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from app.utils import init_session_state, apply_layout, CHART_COLORS
from src.preprocessing import convert_units

st.set_page_config(page_title="Bill Estimator", page_icon="💰", layout="wide")
init_session_state()

st.title("💰 Electricity Bill Estimator")
st.markdown(
    "Estimate your electricity bill based on actual consumption and ML forecasts. "
    "**This is an estimate only — not an official electricity bill.**"
)

if st.session_state.get("df") is None:
    st.info("📊 Load a dataset on the **Dashboard** page first.")
    st.stop()

df     = st.session_state["df"]
target = st.session_state.get("target_col", config.TARGET_COLUMN)
unit   = st.session_state.get("energy_unit", config.DEFAULT_UNIT)
freq   = st.session_state.get("resample_freq", "1h")

# ── Tariff configuration ─────────────────────────────────────────────────────
st.markdown("### ⚡ Tariff Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    currency = st.text_input("Currency symbol", value=st.session_state.get("currency", "₹"))
    tariff   = st.number_input(
        f"Tariff per kWh ({currency})",
        min_value=0.01, max_value=100.0,
        value=float(st.session_state.get("tariff", config.DEFAULT_TARIFF_PER_KWH)),
        step=0.5,
        format="%.2f",
        help="Electricity cost per kilowatt-hour. Check your utility bill for the exact rate.",
    )
with col2:
    fixed_charge = st.number_input(
        f"Fixed monthly charge ({currency})",
        min_value=0.0, max_value=10000.0,
        value=float(st.session_state.get("fixed_charge", config.DEFAULT_FIXED_MONTHLY_CHARGE)),
        step=10.0,
        format="%.2f",
        help="Fixed component that appears on every bill regardless of usage.",
    )
with col3:
    tax_rate = st.number_input(
        "Tax / surcharge rate (%)",
        min_value=0.0, max_value=50.0,
        value=float(st.session_state.get("tax_rate", config.DEFAULT_TAX_RATE) * 100),
        step=1.0,
        format="%.1f",
    ) / 100

# Slab tariff option
with st.expander("⚙️ Optional: Tiered / slab tariff"):
    st.markdown(
        "Many Indian utilities use tiered pricing where higher consumption attracts higher rates. "
        "Enter slab limits and rates (or leave the single tariff above for flat-rate billing)."
    )
    use_slabs = st.checkbox("Enable slab pricing")
    slabs = []
    if use_slabs:
        n_slabs = st.number_input("Number of slabs", min_value=2, max_value=5, value=3, step=1)
        for i in range(int(n_slabs)):
            sc1, sc2 = st.columns(2)
            with sc1:
                limit = st.number_input(f"Slab {i+1} upper limit (kWh)", value=(i+1)*100.0, key=f"slab_limit_{i}")
            with sc2:
                rate  = st.number_input(f"Slab {i+1} rate ({currency}/kWh)", value=tariff + i*1.0, key=f"slab_rate_{i}")
            slabs.append((limit, rate))

# Save to session state
st.session_state["tariff"]       = tariff
st.session_state["fixed_charge"] = fixed_charge
st.session_state["tax_rate"]     = tax_rate
st.session_state["currency"]     = currency

# ── Helper: compute bill ──────────────────────────────────────────────────────
def compute_bill(total_kwh: float) -> dict:
    """Compute estimated bill for a given energy amount."""
    if use_slabs and slabs:
        # Tiered billing
        energy_cost = 0.0
        remaining = total_kwh
        prev_limit = 0.0
        for limit, rate in slabs:
            slab_usage = min(remaining, limit - prev_limit)
            energy_cost += slab_usage * rate
            remaining -= slab_usage
            prev_limit = limit
            if remaining <= 0:
                break
        # Remaining (above all slabs) at last slab rate
        if remaining > 0:
            energy_cost += remaining * slabs[-1][1]
    else:
        energy_cost = total_kwh * tariff

    subtotal    = energy_cost + fixed_charge
    tax_amount  = subtotal * tax_rate
    total_bill  = subtotal + tax_amount

    return {
        "total_kwh":    total_kwh,
        "energy_cost":  energy_cost,
        "fixed_charge": fixed_charge,
        "tax_amount":   tax_amount,
        "total_bill":   total_bill,
    }


# ── Convert to kWh ────────────────────────────────────────────────────────────
# Determine hours per step
freq_to_hours = {"1min": 1/60, "5min": 5/60, "15min": 0.25, "30min": 0.5, "1h": 1.0, "1D": 24.0}
step_hours = freq_to_hours.get(freq, 1.0)

try:
    series_kwh = convert_units(df[target], from_unit=unit, sampling_freq_hours=step_hours)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ── Historical bill estimate ──────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Historical Bill Estimates")

# Group by month
monthly_kwh = series_kwh.resample("MS").sum()

if len(monthly_kwh) < 1:
    st.warning("Not enough data for monthly billing breakdown.")
else:
    # Bill for each month
    monthly_bills = {ts: compute_bill(kwh) for ts, kwh in monthly_kwh.items()}

    # Summary for most recent full month
    last_month_ts   = monthly_kwh.index[-1]
    last_month_bill = monthly_bills[last_month_ts]
    last_month_label = last_month_ts.strftime("%B %Y")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            f"{last_month_label} — Energy used",
            f"{last_month_bill['total_kwh']:.1f} kWh",
        )
    with c2:
        st.metric("Energy charge", f"{currency}{last_month_bill['energy_cost']:.2f}")
    with c3:
        st.metric("Fixed + tax", f"{currency}{last_month_bill['fixed_charge'] + last_month_bill['tax_amount']:.2f}")
    with c4:
        st.metric(
            "Estimated total bill",
            f"{currency}{last_month_bill['total_bill']:.2f}",
        )
    st.caption(
        f"⚠️ Estimate based on configured tariff ({currency}{tariff:.2f}/kWh). "
        "Actual bill may differ due to local tariff structures, subsidies, or surcharges."
    )

    # Monthly chart
    months = [ts.strftime("%b %Y") for ts in monthly_kwh.index]
    totals = [monthly_bills[ts]["total_bill"] for ts in monthly_kwh.index]
    kwhs   = [monthly_bills[ts]["total_kwh"]  for ts in monthly_kwh.index]

    fig_bill = go.Figure()
    fig_bill.add_trace(go.Bar(
        x=months, y=totals, name=f"Estimated bill ({currency})",
        marker_color=CHART_COLORS["actual"],
        hovertemplate="%{x}<br>Bill: " + currency + "%{y:.2f}<extra></extra>",
    ))
    apply_layout(fig_bill, title=f"Monthly Estimated Bill ({currency})", height=320)
    fig_bill.update_yaxes(title_text=currency)
    st.plotly_chart(fig_bill, use_container_width=True)

# ── Forecast bill estimate ─────────────────────────────────────────────────────
forecast_df = st.session_state.get("forecast_df")
if forecast_df is not None:
    st.divider()
    st.markdown("### 🔮 Forecast Bill Estimate")
    st.caption("Based on the forecast from the Forecasting page.")

    try:
        fc_kwh_series = convert_units(
            forecast_df["predicted"], from_unit=unit, sampling_freq_hours=step_hours
        )
        fc_total_kwh = fc_kwh_series.sum()
        fc_days = (forecast_df.index[-1] - forecast_df.index[0]).days or 1
        fc_monthly_kwh_est = fc_total_kwh / fc_days * 30  # rough monthly rate

        fc_bill = compute_bill(fc_total_kwh)
        fc_monthly_bill = compute_bill(fc_monthly_kwh_est)

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric("Forecast period energy", f"{fc_total_kwh:.1f} kWh")
        with fc2:
            st.metric("Forecast period cost", f"{currency}{fc_bill['total_bill']:.2f}")
        with fc3:
            st.metric("Projected monthly cost", f"{currency}{fc_monthly_bill['total_bill']:.2f}")

        st.caption(
            "⚠️ Forecast bill is an estimate derived from ML model predictions, "
            "which may be inaccurate. This is NOT an official electricity bill."
        )

    except Exception as e:
        st.warning(f"Could not compute forecast bill: {e}")

# ── Energy Savings Simulator ───────────────────────────────────────────────────
st.divider()
st.markdown("### 💡 Smart Energy Insights")

target_series = df[target].dropna()

# Pattern insights
insights = []

# Weekend vs weekday
weekend_mask = target_series.index.dayofweek >= 5
weekend_avg  = target_series[weekend_mask].mean()
weekday_avg  = target_series[~weekend_mask].mean()
if weekend_avg > weekday_avg * 1.05:
    diff_pct = (weekend_avg - weekday_avg) / weekday_avg * 100
    insights.append(
        f"📅 Average weekend consumption is **{diff_pct:.1f}% higher** than weekday consumption "
        f"({weekend_avg:.3f} vs {weekday_avg:.3f} {unit})."
    )
elif weekday_avg > weekend_avg * 1.05:
    diff_pct = (weekday_avg - weekend_avg) / weekend_avg * 100
    insights.append(
        f"📅 Average weekday consumption is **{diff_pct:.1f}% higher** than weekends "
        f"(possibly due to work-from-home appliances or occupancy patterns)."
    )

# Peak hours
hourly_avg = target_series.groupby(target_series.index.hour).mean()
peak_hour  = int(hourly_avg.idxmax())
low_hour   = int(hourly_avg.idxmin())
insights.append(
    f"🕐 Consumption is highest around **{peak_hour:02d}:00** and lowest around **{low_hour:02d}:00**."
)

# Evening vs other
evening_mask = (target_series.index.hour >= 18) & (target_series.index.hour < 23)
evening_avg  = target_series[evening_mask].mean()
overall_avg  = target_series.mean()
if evening_avg > overall_avg * 1.2:
    pct_above = (evening_avg - overall_avg) / overall_avg * 100
    insights.append(
        f"🌆 Evening hours (18:00–23:00) account for above-average consumption "
        f"(**{pct_above:.1f}% above** the day-long average)."
    )

# Forecast insight
if forecast_df is not None:
    fc_avg  = forecast_df["predicted"].mean()
    hist_avg = target_series.tail(7 * 24).mean()  # last 7 days
    if fc_avg > hist_avg * 1.10:
        insights.append(
            f"🔮 The model forecasts **higher usage** than recent average "
            f"({fc_avg:.3f} vs {hist_avg:.3f} {unit}/step)."
        )
    elif fc_avg < hist_avg * 0.90:
        insights.append(
            f"🔮 The model forecasts **lower usage** than recent average "
            f"({fc_avg:.3f} vs {hist_avg:.3f} {unit}/step)."
        )

for insight in insights:
    st.info(insight)

if not insights:
    st.info("Not enough pattern data to generate insights yet.")

# Savings simulator
st.markdown("#### 💸 Savings Simulator")
reduction_pct = st.slider(
    "What if you reduced consumption by (%)", min_value=1, max_value=40, value=10
)
if len(monthly_kwh) > 0:
    last_kwh    = monthly_bills[last_month_ts]["total_kwh"]
    reduced_kwh = last_kwh * (1 - reduction_pct / 100)
    original_bill  = compute_bill(last_kwh)["total_bill"]
    reduced_bill   = compute_bill(reduced_kwh)["total_bill"]
    savings         = original_bill - reduced_bill

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Current est. monthly bill", f"{currency}{original_bill:.2f}")
    with sc2:
        st.metric(f"With {reduction_pct}% reduction", f"{currency}{reduced_bill:.2f}")
    with sc3:
        st.metric("Monthly saving", f"{currency}{savings:.2f}", delta=f"-{savings:.2f}")

    st.caption(
        "This simulator uses your current consumption data and configured tariff. "
        "Actual savings will depend on which appliances you reduce usage of."
    )
