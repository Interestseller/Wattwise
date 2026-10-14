"""
config.py — Central configuration for the Smart Home Energy Predictor.

All project-wide defaults live here. Override them through the Streamlit UI
or by editing this file before running the application.
"""

import os

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
# Place your CSV file here (or upload it through the UI).
DATASET_PATH = "./data/energy_dataset.csv"

# Default column names. These are used as starting guesses; the UI lets you
# pick different columns if your file uses different names.
TIMESTAMP_COLUMN = "DateTime"
TARGET_COLUMN = "Global_active_power"

# ---------------------------------------------------------------------------
# Sampling / resampling
# ---------------------------------------------------------------------------
# Target resampling frequency used when the raw data is irregular or very
# high-frequency.  Common choices: "1min", "15min", "1h", "1D".
DEFAULT_RESAMPLE_FREQ = "1h"

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
# Fraction of chronological data used for training (remainder = validation + test).
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
# TEST_RATIO is implied: 1 - TRAIN_RATIO - VALIDATION_RATIO

# Fixed random seed for reproducible results.
RANDOM_STATE = 42

# Minimum number of samples required before attempting to train a model.
MIN_SAMPLES_FOR_TRAINING = 168  # 1 week of hourly data

# Lag feature windows (in number of time steps).
LAG_PERIODS = [1, 2, 3, 6, 12, 24, 48, 168]  # up to 1 week of hourly lags

# Rolling window sizes (in number of time steps).
ROLLING_WINDOWS = [3, 6, 12, 24]

# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------
MODELS_DIR = "./models"
MODEL_FILENAME = "energy_forecaster.joblib"
METADATA_FILENAME = "model_metadata.json"
METRICS_FILENAME = "evaluation_metrics.json"

MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
METADATA_PATH = os.path.join(MODELS_DIR, METADATA_FILENAME)
METRICS_PATH = os.path.join(MODELS_DIR, METRICS_FILENAME)

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
# Available forecast horizons (in hours when data is hourly).
FORECAST_HORIZONS = {
    "Next 24 hours": 24,
    "Next 7 days": 168,
    "Next 30 days": 720,
}

# Minimum historical observations required per forecast horizon.
MIN_HISTORY_FOR_HORIZON = {
    24: 48,
    168: 336,
    720: 1440,
}

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
# Contamination parameter for Isolation Forest (expected fraction of outliers).
ANOMALY_CONTAMINATION = 0.05

# Z-score threshold for statistical deviation detection.
ZSCORE_THRESHOLD = 3.0

# Rolling window for rolling z-score.
ROLLING_ZSCORE_WINDOW = 24

# ---------------------------------------------------------------------------
# Electricity billing
# ---------------------------------------------------------------------------
# All monetary values are in the configured CURRENCY_SYMBOL.
CURRENCY_SYMBOL = "₹"
DEFAULT_TARIFF_PER_KWH = 6.50       # ₹ per kWh (example; varies by state/utility)
DEFAULT_FIXED_MONTHLY_CHARGE = 50.0  # ₹ fixed monthly charge
DEFAULT_TAX_RATE = 0.05              # 5% surcharge/tax on energy cost

# ---------------------------------------------------------------------------
# Data units
# ---------------------------------------------------------------------------
# Supported energy/power units for display.
SUPPORTED_UNITS = ["kWh", "Wh", "kW", "W"]
DEFAULT_UNIT = "kW"  # raw Global_active_power is typically in kilowatts

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_TITLE = "Smart Home Energy Intelligence"
APP_ICON = "⚡"
APP_VERSION = "1.0.0"
