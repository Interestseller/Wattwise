"""
flask_app/routes.py — All Flask routes and JSON API endpoints for WattWise.
"""

from __future__ import annotations

import io
from flask import (
    Blueprint, render_template, jsonify, request, Response, current_app
)
from flask_app.services import (
    get_overview_data,
    get_analytics_data,
    get_forecast_data,
    get_model_info,
    run_training,
    estimate_bill,
    get_anomaly_data,
    export_csv,
    load_dataset,
)
import config as app_config

main = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@main.route("/")
@main.route("/overview")
def overview():
    return render_template("pages/overview.html",
                           page="overview",
                           title="Overview",
                           subtitle=f"Energy intelligence dashboard · Last 30 days")


@main.route("/analytics")
def analytics():
    return render_template("pages/analytics.html",
                           page="analytics",
                           title="Energy Analytics",
                           subtitle="Historical usage, breakdowns, and peak analysis")


@main.route("/forecasting")
def forecasting():
    return render_template("pages/forecasting.html",
                           page="forecasting",
                           title="AI Forecasting",
                           subtitle="Machine-learning energy consumption forecasts")


@main.route("/model-lab")
def model_lab():
    return render_template("pages/model_lab.html",
                           page="model_lab",
                           title="Model Lab",
                           subtitle="Train, evaluate, and compare ML models")


@main.route("/bill-estimator")
def bill_estimator():
    return render_template("pages/bill_estimator.html",
                           page="bill_estimator",
                           title="Bill Estimator",
                           subtitle="Transparent electricity cost projections")


@main.route("/anomaly-monitor")
def anomaly_monitor():
    return render_template("pages/anomaly_monitor.html",
                           page="anomaly_monitor",
                           title="Anomaly Monitor",
                           subtitle="Unusual consumption patterns and alerts")


@main.route("/settings")
def settings():
    return render_template("pages/settings.html",
                           page="settings",
                           title="Settings",
                           subtitle="Application configuration and preferences")


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------

@main.route("/api/overview")
def api_overview():
    days = request.args.get("days", 30, type=int)
    return jsonify(get_overview_data(days=days))


@main.route("/api/analytics")
def api_analytics():
    period = request.args.get("period", "daily")
    days   = request.args.get("days", 90, type=int)
    if period not in ("hourly", "daily", "weekly", "monthly"):
        return jsonify({"error": "Invalid period"}), 400
    return jsonify(get_analytics_data(period=period, days=days))


@main.route("/api/forecast")
def api_forecast():
    steps = request.args.get("steps", 24, type=int)
    if steps not in (24, 168, 720):
        steps = 24
    return jsonify(get_forecast_data(steps=steps))


@main.route("/api/models")
def api_models():
    return jsonify(get_model_info())


@main.route("/api/train", methods=["POST"])
def api_train():
    result = run_training()
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@main.route("/api/bill-estimate", methods=["POST"])
def api_bill_estimate():
    data = request.get_json(silent=True) or {}
    try:
        tariff      = float(data.get("tariff",       app_config.DEFAULT_TARIFF_PER_KWH))
        fixed       = float(data.get("fixed_charge", app_config.DEFAULT_FIXED_MONTHLY_CHARGE))
        tax_rate    = float(data.get("tax_rate",     app_config.DEFAULT_TAX_RATE))
        reduction   = float(data.get("reduction_pct", 0.0))
        days        = int(data.get("days", 30))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    return jsonify(estimate_bill(
        tariff=tariff, fixed_charge=fixed,
        tax_rate=tax_rate, reduction_pct=reduction, days=days,
    ))


@main.route("/api/anomalies")
def api_anomalies():
    method = request.args.get("method", "both")
    days   = request.args.get("days", 30, type=int)
    if method not in ("isolation_forest", "zscore", "both"):
        method = "both"
    return jsonify(get_anomaly_data(method=method, days=days))


@main.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400
    result = load_dataset(f)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@main.route("/api/export/csv")
def api_export_csv():
    days = request.args.get("days", 30, type=int)
    csv_data = export_csv(days=days)
    if csv_data is None:
        return jsonify({"error": "No data available"}), 404
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=wattwise_export.csv"},
    )
