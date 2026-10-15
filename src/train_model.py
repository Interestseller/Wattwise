"""
src/train_model.py — Train, compare, and save ML energy forecasting models.

ML concepts explained
---------------------

Supervised learning
    We teach the model by showing it many examples of (features → target).
    The model learns a mapping from features (hour, lag values, etc.) to
    the target (energy consumption).

Regression
    Our target is a continuous number (e.g. 1.25 kWh), not a category.
    Regression models predict numeric values rather than class labels.

Baseline model
    A naïve predictor that uses no ML at all. We compare every ML model
    against this baseline — if a model can't beat a naïve prediction,
    it adds no value.  Here we use "same hour yesterday" (lag_24).

Random Forest
    An ensemble of many decision trees. Each tree is trained on a random
    sample of the data and a random subset of features. The final prediction
    is the average of all trees. Random forests are robust, require little
    tuning, and resist overfitting.

Gradient Boosting
    Another tree ensemble, but trees are built sequentially: each tree
    tries to correct the errors of the previous one. Often more accurate
    than random forests but slightly more prone to overfitting without tuning.

Overfitting
    When a model memorises training data instead of learning general patterns.
    We detect this by comparing training error with validation error.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from src.preprocessing import chronological_split
from src.feature_engineering import build_feature_matrix
from src.evaluate_model import evaluate_predictions


# ---------------------------------------------------------------------------
# Baseline model (no ML)
# ---------------------------------------------------------------------------

class BaselineModel:
    """
    Naïve baseline: predict using the same-hour-yesterday value (lag_24).

    If lag_24 is unavailable, fall back to the rolling mean of the last
    24 observations.  This is not machine learning — it is a reference point
    to check whether our ML models add value over a simple heuristic.
    """

    def __init__(self, lag: int = 24):
        self.lag = lag
        self.fallback_mean_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        self.fallback_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag_col = f"lag_{self.lag}"
        if lag_col in X.columns:
            preds = X[lag_col].values.copy()
            # Where lag is NaN, substitute the training mean
            preds = np.where(np.isnan(preds), self.fallback_mean_, preds)
        else:
            # No lag column available — use training mean
            preds = np.full(len(X), self.fallback_mean_)
        return preds.astype(float)

    def get_params(self, deep: bool = True) -> dict:
        return {"lag": self.lag}


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_and_select_model(
    df: pd.DataFrame,
    target_col: str,
    lag_periods: list[int] = None,
    rolling_windows: list[int] = None,
    train_ratio: float = None,
    val_ratio: float = None,
    models_to_train: list[str] = None,
    rf_params: dict = None,
    gb_params: dict = None,
    random_state: int = None,
) -> dict:
    """
    Full ML training pipeline: preprocess → feature engineering →
    chronological split → train candidates → evaluate → select best model.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed, resampled DataFrame with a DatetimeIndex.
    target_col : str
        Name of the target column.
    lag_periods, rolling_windows : lists
        Feature engineering configuration. Defaults from config.py.
    train_ratio, val_ratio : float
        Split fractions. Defaults from config.py.
    models_to_train : list[str]
        Subset of ["baseline", "random_forest", "gradient_boosting"] to train.
    rf_params : dict
        Keyword arguments forwarded to RandomForestRegressor.
    gb_params : dict
        Keyword arguments forwarded to GradientBoostingRegressor.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    result : dict  containing:
        "best_model"       : the best fitted model object
        "best_model_name"  : str name of the best model
        "feature_names"    : list[str]
        "scaler"           : fitted StandardScaler (or None if not used)
        "all_metrics"      : dict of per-model evaluation metrics
        "training_info"    : metadata dict (durations, split sizes, etc.)
        "train_df"         : training split DataFrame
        "val_df"           : validation split DataFrame
        "test_df"          : test split DataFrame
        "predictions"      : dict model_name → predictions on test set
    """
    if random_state is None:
        random_state = config.RANDOM_STATE
    if models_to_train is None:
        models_to_train = ["baseline", "random_forest", "gradient_boosting"]

    start_time = time.time()

    # ── 1. Build features ────────────────────────────────────────────────────
    X_all, y_all, feature_names = build_feature_matrix(
        df, target_col,
        lag_periods=lag_periods,
        rolling_windows=rolling_windows,
        drop_na=True,
    )

    if len(X_all) < config.MIN_SAMPLES_FOR_TRAINING:
        raise ValueError(
            f"Only {len(X_all)} usable samples after feature engineering "
            f"(minimum required: {config.MIN_SAMPLES_FOR_TRAINING}). "
            "Provide more historical data or reduce the number of lag periods."
        )

    # ── 2. Chronological split ───────────────────────────────────────────────
    # We split on the full feature DataFrame so that all three splits have
    # consistent features and indices.
    full_df = X_all.copy()
    full_df[target_col] = y_all

    train_df, val_df, test_df = chronological_split(
        full_df, train_ratio=train_ratio, val_ratio=val_ratio
    )

    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_val,   y_val   = val_df.drop(columns=[target_col]),   val_df[target_col]
    X_test,  y_test  = test_df.drop(columns=[target_col]),  test_df[target_col]

    # ── 3. Scale features (improves gradient boosting; RF is scale-invariant) ─
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # Wrap back in DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_val_scaled   = pd.DataFrame(X_val_scaled,   columns=feature_names, index=X_val.index)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=feature_names, index=X_test.index)

    # ── 4. Define candidate models ───────────────────────────────────────────
    default_rf_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "n_jobs": -1,
        "random_state": random_state,
    }
    default_gb_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": random_state,
    }

    candidates = {}
    if "baseline" in models_to_train:
        candidates["baseline"] = BaselineModel(lag=24)
    if "random_forest" in models_to_train:
        params = {**default_rf_params, **(rf_params or {})}
        candidates["random_forest"] = RandomForestRegressor(**params)
    if "gradient_boosting" in models_to_train:
        params = {**default_gb_params, **(gb_params or {})}
        candidates["gradient_boosting"] = GradientBoostingRegressor(**params)

    # Try XGBoost if available
    if "xgboost" in models_to_train:
        try:
            import xgboost as xgb  # noqa: F401
            candidates["xgboost"] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state,
                verbosity=0,
            )
        except ImportError:
            pass  # XGBoost not installed; silently skip

    # ── 5. Train all candidates and evaluate on validation set ───────────────
    model_fit_times: dict[str, float] = {}
    val_metrics: dict[str, dict] = {}
    test_predictions: dict[str, np.ndarray] = {}
    fitted_models: dict = {}

    for name, model in candidates.items():
        t0 = time.time()

        # Baseline does not benefit from scaling (it only uses lag columns)
        if name == "baseline":
            model.fit(X_train, y_train)
            val_preds  = model.predict(X_val)
            test_preds = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            val_preds  = model.predict(X_val_scaled)
            test_preds = model.predict(X_test_scaled)

        model_fit_times[name] = time.time() - t0

        # Evaluate on validation split (used for model selection)
        val_metrics[name] = evaluate_predictions(y_val.values, val_preds)
        test_predictions[name] = test_preds
        fitted_models[name] = model

    # ── 6. Select best model based on validation RMSE ────────────────────────
    best_model_name = min(val_metrics, key=lambda k: val_metrics[k]["rmse"])
    best_model = fitted_models[best_model_name]

    # ── 7. Evaluate best model on the held-out test set ──────────────────────
    test_metrics: dict[str, dict] = {}
    for name in candidates:
        test_preds = test_predictions[name]
        test_metrics[name] = evaluate_predictions(y_test.values, test_preds)

    # ── 8. Build training info metadata ─────────────────────────────────────
    total_time = time.time() - start_time
    training_info = {
        "total_rows": len(X_all),
        "train_rows": len(X_train),
        "val_rows":   len(X_val),
        "test_rows":  len(X_test),
        "train_start": str(X_train.index.min()),
        "train_end":   str(X_train.index.max()),
        "val_start":   str(X_val.index.min()),
        "val_end":     str(X_val.index.max()),
        "test_start":  str(X_test.index.min()),
        "test_end":    str(X_test.index.max()),
        "total_training_seconds": round(total_time, 2),
        "model_fit_seconds": model_fit_times,
        "target_col": target_col,
        "feature_count": len(feature_names),
    }

    return {
        "best_model":      best_model,
        "best_model_name": best_model_name,
        "feature_names":   feature_names,
        "scaler":          scaler,
        "val_metrics":     val_metrics,
        "test_metrics":    test_metrics,
        "training_info":   training_info,
        "train_df":        train_df,
        "val_df":          val_df,
        "test_df":         test_df,
        "test_predictions": test_predictions,
        "fitted_models":   fitted_models,
        "y_test":          y_test,
        "target_col":      target_col,
    }


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(result: dict, models_dir: str = None) -> str:
    """
    Save the best model, scaler, metadata, and evaluation metrics to disk.

    Returns
    -------
    str : Path to the saved model file.
    """
    if models_dir is None:
        models_dir = config.MODELS_DIR

    os.makedirs(models_dir, exist_ok=True)

    # Save model + scaler together as a bundle
    bundle = {
        "model":         result["best_model"],
        "scaler":        result["scaler"],
        "feature_names": result["feature_names"],
        "target_col":    result["target_col"],
        "model_name":    result["best_model_name"],
    }
    model_path = os.path.join(models_dir, config.MODEL_FILENAME)
    joblib.dump(bundle, model_path)

    # Metadata JSON
    metadata = {
        "model_name":    result["best_model_name"],
        "feature_names": result["feature_names"],
        "target_col":    result["target_col"],
        "training_info": result["training_info"],
        "app_version":   config.APP_VERSION,
    }
    metadata_path = os.path.join(models_dir, config.METADATA_FILENAME)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Evaluation metrics JSON
    metrics_to_save = {
        "val_metrics":  result["val_metrics"],
        "test_metrics": result["test_metrics"],
    }
    metrics_path = os.path.join(models_dir, config.METRICS_FILENAME)
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2, default=str)

    return model_path


def load_model(models_dir: str = None) -> dict:
    """
    Load a previously saved model bundle.

    Returns
    -------
    dict with keys: model, scaler, feature_names, target_col, model_name

    Raises
    ------
    FileNotFoundError if the model file does not exist.
    """
    if models_dir is None:
        models_dir = config.MODELS_DIR

    model_path = os.path.join(models_dir, config.MODEL_FILENAME)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Please train a model first using the 'Train Your ML Model' page."
        )

    bundle = joblib.load(model_path)

    # Validate expected keys
    required_keys = {"model", "scaler", "feature_names", "target_col"}
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ValueError(
            f"Saved model bundle is missing keys: {missing}. "
            "The model file may be corrupted or from an incompatible version. "
            "Please retrain."
        )

    return bundle


def load_metrics(models_dir: str = None) -> dict:
    """Load saved evaluation metrics. Returns empty dict if not found."""
    if models_dir is None:
        models_dir = config.MODELS_DIR

    metrics_path = os.path.join(models_dir, config.METRICS_FILENAME)
    if not os.path.isfile(metrics_path):
        return {}

    with open(metrics_path) as f:
        return json.load(f)


def load_metadata(models_dir: str = None) -> dict:
    """Load saved model metadata. Returns empty dict if not found."""
    if models_dir is None:
        models_dir = config.MODELS_DIR

    metadata_path = os.path.join(models_dir, config.METADATA_FILENAME)
    if not os.path.isfile(metadata_path):
        return {}

    with open(metadata_path) as f:
        return json.load(f)
