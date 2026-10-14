"""
data/generate_demo_data.py — Generate a synthetic energy consumption dataset.

This script creates demo_energy_dataset.csv for testing the application
interface before you have your own dataset.

⚠️  IMPORTANT: This is SYNTHETIC (artificial) data, not real household data.
    Do not use it to draw conclusions about actual electricity consumption.
    Do not use demo-dataset model results to make real financial decisions.

The synthetic data is designed to be realistic-looking by incorporating:
- Daily patterns (higher usage morning/evening, lower overnight)
- Weekly patterns (slightly different weekend vs weekday usage)
- Seasonal variation (higher usage in summer and winter)
- Random noise
- A small number of planted anomalies

Run with: python data/generate_demo_data.py
"""

import os
import numpy as np
import pandas as pd

# Seed for reproducibility
rng = np.random.default_rng(42)

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE = "2023-01-01"
END_DATE   = "2024-12-31"
FREQ       = "1h"           # hourly data
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "demo_energy_dataset.csv")


def _hourly_pattern(hour: int) -> float:
    """Typical household energy consumption multiplier by hour of day."""
    # Low overnight, peaks in morning (7–9) and evening (18–22)
    pattern = [
        0.35, 0.30, 0.28, 0.28, 0.30, 0.40,   # 00–05
        0.60, 0.90, 1.00, 0.80, 0.70, 0.65,   # 06–11
        0.65, 0.60, 0.60, 0.65, 0.75, 0.85,   # 12–17
        1.00, 1.10, 1.05, 0.90, 0.70, 0.50,   # 18–23
    ]
    return pattern[hour]


def _seasonal_factor(month: int) -> float:
    """Higher consumption in winter (heating) and summer (cooling)."""
    factors = {
        1: 1.20, 2: 1.15, 3: 1.05, 4: 0.95,
        5: 0.90, 6: 1.00, 7: 1.15, 8: 1.20,
        9: 1.00, 10: 0.95, 11: 1.05, 12: 1.20,
    }
    return factors.get(month, 1.0)


def generate_demo_dataset() -> pd.DataFrame:
    timestamps = pd.date_range(START_DATE, END_DATE, freq=FREQ)
    n = len(timestamps)

    # Base consumption level (kW)
    base_kw = 1.20

    values = []
    for ts in timestamps:
        hour   = ts.hour
        month  = ts.month
        is_weekend = ts.dayofweek >= 5

        h_factor  = _hourly_pattern(hour)
        s_factor  = _seasonal_factor(month)
        w_factor  = 1.05 if is_weekend else 1.0

        # Base value with factors
        v = base_kw * h_factor * s_factor * w_factor

        # Gaussian noise (~10% std)
        v += rng.normal(0, 0.10) * v

        # Clip to non-negative
        v = max(v, 0.0)
        values.append(v)

    df = pd.DataFrame({
        "DateTime": timestamps,
        "Global_active_power": np.round(values, 4),
        "Voltage": np.round(rng.normal(240, 2, n), 1),
        "Global_intensity": np.round(np.array(values) / 240 * 1000, 2),
    })

    # ── Plant a few anomalies (unusually high consumption) ───────────────────
    n_anomalies = int(n * 0.002)  # ~0.2% of records
    anomaly_idx = rng.choice(n, size=n_anomalies, replace=False)
    df.loc[anomaly_idx, "Global_active_power"] *= rng.uniform(2.5, 4.0, len(anomaly_idx))

    # ── Introduce a few short gaps (missing values) ──────────────────────────
    gap_starts = rng.choice(n - 5, size=5, replace=False)
    for gs in gap_starts:
        gap_len = rng.integers(1, 4)
        df.loc[gs:gs + gap_len, "Global_active_power"] = np.nan

    return df


if __name__ == "__main__":
    print("Generating synthetic demo dataset …")
    df = generate_demo_dataset()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df):,} rows to: {OUTPUT_FILE}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    print(f"Columns: {list(df.columns)}")
    print("\n⚠️  This is SYNTHETIC data — not real household measurements.")
