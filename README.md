# Smart Home Energy Intelligence

> An AI-powered electricity consumption analytics and forecasting platform built with Python, Scikit-learn, and Streamlit.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)](https://streamlit.io)

---

## Overview

This application helps home owners and data science students understand, predict, and reduce electricity consumption using genuine machine learning — not hard-coded rules.

**What it does:**
- Upload historical electricity data and get an instant quality report
- Train and compare ML models (Baseline, Random Forest, Gradient Boosting)
- Forecast future consumption using recursive multi-step prediction
- Estimate electricity bills with a configurable tariff
- Detect unusual consumption patterns with Isolation Forest and Z-score methods
- Generate data-driven insights about your consumption habits

---

## Problem Statement

Households typically receive an electricity bill once a month with no breakdown of when or how energy was consumed. This application analyses hourly or sub-hourly smart-meter data to answer questions like:

- When do I use the most electricity?
- How much will I use next week?
- What will my bill be?
- Are there any unusual consumption events I should investigate?

---

## Features

| Feature | Description |
|---------|-------------|
| **Dataset validation** | Automatic quality check: missing values, duplicates, sampling frequency, date range |
| **Column configurability** | Works with any CSV column names — no hard-coded assumptions |
| **Preprocessing** | Duplicate removal, interpolation, resampling, unit conversion |
| **Feature engineering** | Calendar features, lag features, rolling statistics |
| **Baseline model** | Same-hour-yesterday prediction (reference point) |
| **Random Forest** | Ensemble of decision trees |
| **Gradient Boosting** | Sequential tree ensemble |
| **XGBoost (optional)** | If installed |
| **Chronological evaluation** | Train/validation/test split with no data shuffling |
| **Recursive forecasting** | 24h / 7d / 30d ahead predictions |
| **Bill estimation** | Configurable tariff, fixed charges, tax, slab pricing |
| **Anomaly detection** | Isolation Forest + rolling Z-score |
| **Energy insights** | Data-driven recommendations based on your usage patterns |
| **Model persistence** | Save and reload trained models across sessions |

---

## Architecture

```
smart-home-energy-predictor/
│
├── data/                       ← PUT YOUR CSV FILE HERE
│   ├── energy_dataset.csv      ← your dataset (not committed to git)
│   ├── demo_energy_dataset.csv ← synthetic demo data (safe to commit)
│   ├── generate_demo_data.py   ← script to regenerate demo data
│   └── README.md               ← dataset format documentation
│
├── models/                     ← trained model artifacts (auto-created)
│   ├── energy_forecaster.joblib
│   ├── model_metadata.json
│   └── evaluation_metrics.json
│
├── src/                        ← ML pipeline (framework-agnostic)
│   ├── data_loader.py          ← load & validate CSV
│   ├── preprocessing.py        ← clean, resample, split
│   ├── feature_engineering.py  ← calendar + lag + rolling features
│   ├── train_model.py          ← train, compare, persist models
│   ├── evaluate_model.py       ← MAE, RMSE, R², MAPE
│   ├── predict_energy.py       ← recursive multi-step forecasting
│   └── anomaly_detection.py    ← Isolation Forest + Z-score
│
├── app/                        ← Streamlit user interface
│   ├── main.py                 ← app entry point
│   ├── utils.py                ← shared helpers, chart theme
│   └── pages/
│       ├── 1_Dashboard.py      ← load data, quality report
│       ├── 2_Analytics.py      ← interactive charts
│       ├── 3_Forecasting.py    ← future consumption forecast
│       ├── 4_Train_Model.py    ← train & evaluate ML models
│       ├── 5_Bill_Estimator.py ← bill calculator + insights
│       └── 6_Anomaly_Detection.py
│
├── tests/                      ← pytest test suite (53 tests)
├── config.py                   ← all configurable defaults
└── requirements.txt
```

---

## Dataset Requirements

### Where to put your data

Place your CSV file at:
```
smart-home-energy-predictor/data/energy_dataset.csv
```

Or upload it directly through the Dashboard page UI.

### Minimum required format

```csv
DateTime,Global_active_power
2024-01-01 00:00:00,1.25
2024-01-01 01:00:00,1.18
2024-01-01 02:00:00,1.42
```

- **At least one timestamp column** (any parseable datetime format)
- **At least one numeric column** (power in kW/W or energy in kWh/Wh)

### Extended format (more features when available)

```csv
DateTime,Global_active_power,Voltage,Global_intensity
2024-01-01 00:00:00,1.25,240.1,5.2
2024-01-01 01:00:00,1.18,239.8,4.9
```

### Recommendations

| Attribute | Minimum | Recommended |
|-----------|---------|-------------|
| Duration | 2 weeks | 3–12 months |
| Frequency | 15 min | 1 hour |
| Target missing | < 20% | < 5% |

### Obtaining a real dataset

| Dataset | Description | Link |
|---------|-------------|------|
| UCI Household Power | 2-year, 1-minute data | https://archive.ics.uci.edu/dataset/235 |
| UK Smart Meters | Half-hourly London data | https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households |

> Always verify the licence before using public datasets.

---

## Setup & Installation

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd smart-home-energy-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install XGBoost

```bash
pip install xgboost
```

### 5. Generate the demo dataset

```bash
python data/generate_demo_data.py
```

---

## Running the Application

```bash
streamlit run app/main.py
```

The app opens at **http://localhost:8501** in your browser.

### Workflow

1. **Dashboard** — upload your CSV (or use the demo dataset)
2. **Dashboard** — select timestamp column, target column, unit
3. **Train ML Model** — configure training settings and click "Train Models"
4. **Analytics** — explore historical consumption charts
5. **Forecasting** — generate a 24h / 7d / 30d forecast
6. **Bill Estimator** — configure your tariff and see cost estimates
7. **Anomaly Detection** — detect unusual consumption events

---

## Model Architecture & Feature Engineering

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `hour` | Calendar | Hour of day (0–23) |
| `day_of_week` | Calendar | Day of week (0=Mon, 6=Sun) |
| `month` | Calendar | Month of year |
| `is_weekend` | Calendar | 1 if Sat/Sun |
| `hour_sin`, `hour_cos` | Cyclic | Circular encoding so hour 23 ≈ hour 0 |
| `lag_1`, `lag_24` | Lag | Consumption 1 and 24 steps ago |
| `lag_168` | Lag | Same hour last week (hourly data) |
| `rolling_mean_24` | Rolling | Mean over last 24 steps |
| `rolling_std_24` | Rolling | Std deviation over last 24 steps |

### Why cyclic encoding?
Standard integer encoding treats hour 0 and hour 23 as 23 units apart. Sine/cosine encoding wraps the cycle so they are adjacent — matching the physical reality that midnight and 11 PM are close neighbours.

### Why no data shuffling?
Time-series observations are not independent. Shuffling before splitting causes **data leakage**: the model would see future patterns during training, making evaluation unrealistically optimistic. All splits are strictly chronological.

### Recursive forecasting
For multi-step forecasts:
1. Predict step t+1 using real historical lags
2. Use the prediction as `lag_1` for step t+2
3. Repeat until the full horizon is covered

Error accumulates with forecast horizon — this is honest and expected behaviour.

---

## Evaluation Metrics

| Metric | Meaning | Good value |
|--------|---------|------------|
| MAE | Mean absolute error — average absolute difference | Small; same units as target |
| RMSE | Root mean squared error — penalises large errors | Close to MAE |
| R² | Variance explained | Close to 1.0 |
| MAPE | Mean absolute % error | < 10–15% (omitted near zero) |

### Example results (synthetic demo dataset)

Trained on `demo_energy_dataset.csv` (2 years, hourly, synthetic data):

| Model | MAE (kW) | RMSE (kW) | R² |
|-------|----------|-----------|-----|
| Baseline (prev-day) | 0.21 | 0.28 | 0.64 |
| Random Forest | 0.13 | 0.17 | 0.84 |
| Gradient Boosting | 0.12 | 0.16 | 0.86 |

> These figures are from **synthetic demonstration data** and do not represent real household consumption accuracy. Results on real data will vary.

---

## Limitations

1. **Recursive forecasting errors grow**: long-horizon forecasts (30 days) are inherently less accurate than 24-hour forecasts.
2. **No weather integration**: adding temperature and weather data would significantly improve accuracy for HVAC-heavy households.
3. **Single-household model**: models are trained on a single household's data; they are not transferable to other homes.
4. **Uncertainty bands are heuristic**: the uncertainty band shown on forecasts is based on historical volatility, not a statistically rigorous prediction interval.
5. **Minimum data requirement**: at least 168 observations (1 week hourly) are needed for the lag-based models.
6. **Appliance detection not implemented**: identifying individual appliances from aggregate readings (NILM) requires a separate, validated model.

---

## Privacy Considerations

- **Do not commit real household data** to a public repository. Real consumption data reveals occupancy patterns and may be protected under GDPR or similar regulations.
- The `.gitignore` excludes `data/*.csv` (except the demo file).
- Trained model files (`.joblib`) may indirectly encode personal patterns. Treat them as sensitive data.
- The application is designed for **local use** — no data is transmitted externally.

---

## Running Tests

```bash
pytest tests/ -v
```

All 53 tests should pass. The suite covers:
- Dataset loading and validation
- Timestamp parsing
- Missing value handling
- Feature engineering (including data-leakage checks)
- Chronological train/test splitting
- Model training and evaluation
- Multi-step forecasting
- Bill calculations
- Anomaly detection
- Invalid input handling

---

## Future Improvements

- [ ] Weather API integration (temperature, humidity)
- [ ] LSTM / Transformer-based forecasting for longer horizons
- [ ] Multi-household comparison
- [ ] Appliance-level NILM (Non-Intrusive Load Monitoring)
- [ ] Automated daily report generation
- [ ] Mobile-responsive layout
- [ ] Docker deployment
- [ ] Statistically rigorous prediction intervals (conformal prediction)

---

## Licence

MIT — see `LICENSE` for details.

---

*Built as an AI/ML internship project using Python, Scikit-learn, Plotly, and Streamlit.*
