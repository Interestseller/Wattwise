"""
app/pages/4_Train_Model.py — ML model training interface.
"""

import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from app.utils import init_session_state, apply_layout, CHART_COLORS, error_box, success_box
from src.train_model import train_and_select_model, save_model, load_model
from src.evaluate_model import explain_metrics, build_comparison_table

st.set_page_config(page_title="Train ML Model", page_icon="🤖", layout="wide")
init_session_state()

st.title("🤖 Train Your ML Model")
st.markdown(
    "Configure and train energy forecasting models on your dataset. "
    "The pipeline will train multiple candidates and select the best performer on a held-out validation set."
)

if st.session_state.get("df") is None:
    st.info("📊 No dataset loaded. Go to **Dashboard** to load your data first.")
    st.stop()

df     = st.session_state["df"]
target = st.session_state.get("target_col", config.TARGET_COLUMN)
unit   = st.session_state.get("energy_unit", config.DEFAULT_UNIT)
freq   = st.session_state.get("resample_freq", "1h")

st.caption(f"Dataset: **{len(df):,} rows** | Target: `{target}` | Freq: `{freq}`")

# ── Training configuration ───────────────────────────────────────────────────
st.markdown("### ⚙️ Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Train/Val/Test split (%)**")
    train_pct = st.slider("Training %", 50, 85, 70, step=5)
    val_pct   = st.slider("Validation %", 5, 30, 15, step=5)
    test_pct  = 100 - train_pct - val_pct
    st.caption(f"Test: {test_pct}%")
    if test_pct < 5:
        st.warning("Test set is very small.")

with col2:
    st.markdown("**Models to train**")
    train_baseline = st.checkbox("Baseline (previous-day same-hour)", value=True)
    train_rf       = st.checkbox("Random Forest", value=True)
    train_gb       = st.checkbox("Gradient Boosting", value=True)
    train_xgb      = st.checkbox("XGBoost (if installed)", value=False)

with col3:
    st.markdown("**Lag feature configuration**")
    max_lag_hrs = st.selectbox(
        "Maximum lag (hours)",
        options=[24, 48, 168],
        index=1,
        help="lag_24 = same hour yesterday, lag_168 = same hour last week.",
    )

    lag_presets = {
        24:  [1, 2, 3, 6, 12, 24],
        48:  [1, 2, 3, 6, 12, 24, 48],
        168: [1, 2, 3, 6, 12, 24, 48, 168],
    }
    lag_periods = lag_presets[max_lag_hrs]
    st.caption(f"Lag periods: {lag_periods}")

# ── Advanced hyperparameters ─────────────────────────────────────────────────
with st.expander("⚙️ Advanced: Model hyperparameters"):
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("**Random Forest**")
        rf_n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
        rf_max_depth    = st.slider("max_depth", 3, 20, 10)
    with ac2:
        st.markdown("**Gradient Boosting**")
        gb_n_estimators  = st.slider("n_estimators (GB)", 50, 500, 200, step=50)
        gb_learning_rate = st.select_slider(
            "learning_rate", options=[0.01, 0.02, 0.05, 0.1, 0.2], value=0.05
        )
        gb_max_depth     = st.slider("max_depth (GB)", 2, 10, 5)

# ── Training button ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚀 Start Training")

if len(df) < config.MIN_SAMPLES_FOR_TRAINING:
    st.warning(
        f"Dataset has only {len(df)} rows. Minimum required: {config.MIN_SAMPLES_FOR_TRAINING}. "
        "Load more data or reduce the lag period."
    )
    st.stop()

models_to_train = []
if train_baseline: models_to_train.append("baseline")
if train_rf:       models_to_train.append("random_forest")
if train_gb:       models_to_train.append("gradient_boosting")
if train_xgb:      models_to_train.append("xgboost")

if not models_to_train:
    st.warning("Select at least one model to train.")
    st.stop()

if st.button("▶️ Train Models", type="primary"):
    progress = st.progress(0, text="Preparing…")
    status   = st.empty()

    try:
        rf_params = {"n_estimators": rf_n_estimators, "max_depth": rf_max_depth}
        gb_params = {
            "n_estimators": gb_n_estimators,
            "learning_rate": gb_learning_rate,
            "max_depth": gb_max_depth,
        }

        status.info("⚙️ Building features and splitting data…")
        progress.progress(10, text="Building features…")
        t0 = time.time()

        result = train_and_select_model(
            df=df,
            target_col=target,
            lag_periods=lag_periods,
            train_ratio=train_pct / 100,
            val_ratio=val_pct / 100,
            models_to_train=models_to_train,
            rf_params=rf_params,
            gb_params=gb_params,
        )

        progress.progress(80, text="Saving model…")
        model_path = save_model(result)
        # Also load bundle into session state
        bundle = {
            "model":         result["best_model"],
            "scaler":        result["scaler"],
            "feature_names": result["feature_names"],
            "target_col":    result["target_col"],
            "model_name":    result["best_model_name"],
        }
        st.session_state["model_bundle"] = bundle
        st.session_state["train_result"] = result

        elapsed = time.time() - t0
        progress.progress(100, text="Complete!")
        status.empty()

        success_box(
            f"Training complete in {elapsed:.1f}s. "
            f"Best model: **{result['best_model_name']}**. Saved to `{model_path}`."
        )

    except Exception as exc:
        progress.empty()
        status.empty()
        error_box(f"Training failed: {exc}")
        st.stop()

# ── Training results ─────────────────────────────────────────────────────────
result = st.session_state.get("train_result")
if result:
    st.divider()
    st.markdown("### 📊 Training Results")

    info = result["training_info"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Training rows", f"{info['train_rows']:,}")
    with c2: st.metric("Validation rows", f"{info['val_rows']:,}")
    with c3: st.metric("Test rows", f"{info['test_rows']:,}")
    with c4: st.metric("Features", str(info["feature_count"]))

    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Train: {info['train_start'][:10]} → {info['train_end'][:10]}")
    with c2:
        st.caption(f"Test:  {info['test_start'][:10]} → {info['test_end'][:10]}")

    st.markdown("#### Model Comparison (Validation Set)")
    comp_df = build_comparison_table(result["val_metrics"])
    if not comp_df.empty:
        # Highlight best (first row after sorting)
        best = result["best_model_name"]
        st.dataframe(
            comp_df.style.highlight_min(axis=0, subset=["RMSE", "MAE"], color="#d1fae5"),
            use_container_width=True,
        )
        st.caption(f"✅ Selected model: **{best}** (lowest validation RMSE)")

    st.markdown("#### Model Comparison (Test Set — unbiased final evaluation)")
    test_comp_df = build_comparison_table(result["test_metrics"])
    if not test_comp_df.empty:
        st.dataframe(test_comp_df, use_container_width=True)

    # Metric explanations
    best_name = result["best_model_name"]
    best_test_metrics = result["test_metrics"].get(best_name, {})
    if best_test_metrics:
        with st.expander("📖 What do these metrics mean?"):
            explanations = explain_metrics(best_test_metrics, unit=unit)
            for metric_name, explanation in explanations.items():
                st.markdown(f"**{metric_name}**: {explanation}")

    # Actual vs predicted plot on test set
    y_test = result.get("y_test")
    test_preds_dict = result.get("test_predictions", {})
    if y_test is not None and best_name in test_preds_dict:
        st.markdown("#### Actual vs Predicted (Test Set)")
        preds = test_preds_dict[best_name]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test.index, y=y_test.values,
            name="Actual", line=dict(color=CHART_COLORS["actual"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=y_test.index, y=preds,
            name=f"Predicted ({best_name})",
            line=dict(color=CHART_COLORS["predicted"], width=2, dash="dash"),
        ))
        apply_layout(fig, title="Best Model: Actual vs Predicted", height=380)
        fig.update_yaxes(title_text=unit)
        st.plotly_chart(fig, use_container_width=True)

    # Data leakage explanation
    with st.expander("ℹ️ Why chronological splitting? (no data leakage)"):
        st.markdown("""
**Data leakage** happens when information from the future accidentally helps the model
predict the past — making evaluation metrics unrealistically good.

We **never shuffle** time-series data before splitting. Instead:
- **Training set**: earliest observations (e.g. first 70%)
- **Validation set**: next observations (e.g. next 15%) — used to SELECT the best model
- **Test set**: most recent observations (e.g. last 15%) — used ONCE for final reporting

This way, the model has never seen the test period during training or model selection,
giving a realistic estimate of how it would perform on truly unseen future data.
        """)
