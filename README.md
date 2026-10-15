# ⚡ WattWise — Energy Intelligence Platform

A polished, professional energy analytics web application built with **Flask**, **HTML5/CSS3/JavaScript**, **Plotly.js**, and a **scikit-learn** ML backend.

---

## Screenshots

The application matches the EnergyDash reference design: dark charcoal dashboard shell, fixed sidebar, purple/lavender accent palette, compact KPI cards, and interactive Plotly charts on every page.

---

## Features

| Page | Functionality |
|------|---------------|
| **Overview** | KPI cards (usage, bill, peak hour/anomalies), energy trend chart, weekly comparison, time-of-day breakdown, insight panel |
| **Energy Analytics** | Period & range filters (hourly/daily/weekly/monthly), stats row, usage comparison chart, day-of-week heatmap, hourly profile |
| **AI Forecasting** | Selectable horizon (24h / 7d / 30d), actual context + forecast line + uncertainty band, summary stats |
| **Model Lab** | Train model button, real training status, metric cards (MAE/RMSE/R²/MAPE), model comparison table and charts |
| **Bill Estimator** | Configurable tariff, tax, fixed charge, billing period, reduction scenario slider, transparent breakdown |
| **Anomaly Monitor** | Isolation Forest + Z-Score detection, severity badges (high/medium/low), highlighted chart, sortable table |
| **Settings** | Dataset path, column config, tariff defaults, app info |

---

## Tech Stack

- **Backend:** Python 3.10+, Flask 3.x
- **ML:** scikit-learn (Random Forest, Gradient Boosting, Isolation Forest), pandas, NumPy, joblib
- **Frontend:** HTML5, CSS3, vanilla JavaScript, Plotly.js (CDN)
- **Templating:** Jinja2

---

## Project Structure

```
Wattwise/
├── run.py                     ← Flask entry point
├── config.py                  ← Central configuration
├── requirements.txt
│
├── flask_app/
│   ├── __init__.py
│   ├── routes.py              ← All page routes + JSON API endpoints
│   ├── services.py            ← Data/ML service layer
│   ├── templates/
│   │   ├── base.html          ← Sidebar, header, layout shell
│   │   └── pages/
│   │       ├── overview.html
│   │       ├── analytics.html
│   │       ├── forecasting.html
│   │       ├── model_lab.html
│   │       ├── bill_estimator.html
│   │       ├── anomaly_monitor.html
│   │       └── settings.html
│   └── static/
│       ├── css/main.css       ← Full design system
│       └── js/main.js         ← Shared Plotly theme + API helpers
│
├── src/                       ← ML modules (unchanged)
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict_energy.py
│   ├── anomaly_detection.py
│   └── evaluate_model.py
│
├── data/
│   └── demo_energy_dataset.csv   ← Demo data (used when no real dataset present)
│
├── models/                    ← Saved model artefacts
│   ├── energy_forecaster.joblib
│   ├── model_metadata.json
│   └── evaluation_metrics.json
│
└── tests/                     ← 53 passing unit tests
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Add your own dataset

Place a CSV file at `data/energy_dataset.csv` with at minimum:
- A `DateTime` column (parseable timestamps)
- A `Global_active_power` column (numeric, kW)

If absent, the app runs in **demo mode** using `data/demo_energy_dataset.csv`.

### 3. Run the application

```bash
python run.py
```

Or with Flask's development server:

```bash
flask --app run run --debug
```

Open **http://localhost:5000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/overview` | KPI and trend data |
| GET | `/api/analytics` | Historical breakdown (`?period=daily&days=90`) |
| GET | `/api/forecast` | ML forecast (`?steps=24`) |
| GET | `/api/models` | Saved model info and metrics |
| POST | `/api/train` | Trigger model training |
| POST | `/api/bill-estimate` | Bill calculation (JSON body) |
| GET | `/api/anomalies` | Anomaly detection (`?method=both&days=30`) |
| GET | `/api/export/csv` | Download dataset as CSV |

---

## Configuration

Edit [`config.py`](config.py) to change:
- Dataset path and column names
- Default tariff and billing settings
- Model training parameters (train/val/test split, lag periods, random seed)
- Anomaly detection thresholds

---

## Running Tests

```bash
python -m pytest tests/ -v
```

53 tests, 0 failures.

---

## Demo Mode

When no `data/energy_dataset.csv` is present, the application automatically falls back to the bundled demo dataset and shows a yellow banner. All ML features, forecasts, and anomaly detection remain fully functional in demo mode.
