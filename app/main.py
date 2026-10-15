"""
app/main.py — Smart Home Energy Intelligence — Main Streamlit Application

Entry point for the Streamlit multi-page app.

Run with:
    cd smart-home-energy-predictor
    streamlit run app/main.py
"""

import os
import sys

import streamlit as st

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from app.utils import init_session_state

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2328;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #57606a;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2328;
        border-bottom: 2px solid #3b82d4;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .demo-badge {
        background: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {config.APP_ICON} {config.APP_TITLE}")
    st.markdown("---")

    pages = {
        "📊 Dashboard":          "pages/1_Dashboard.py",
        "📈 Analytics":          "pages/2_Analytics.py",
        "🔮 Forecasting":        "pages/3_Forecasting.py",
        "🤖 Train ML Model":     "pages/4_Train_Model.py",
        "💰 Bill Estimator":     "pages/5_Bill_Estimator.py",
        "⚠️ Anomaly Detection":  "pages/6_Anomaly_Detection.py",
    }

    # Dataset status indicator
    st.markdown("---")
    if st.session_state.get("df") is not None:
        src = st.session_state.get("dataset_source", "")
        label = {
            "uploaded": "📂 Custom dataset loaded",
            "path":     "📂 Dataset loaded from disk",
            "demo":     "🧪 Demo dataset active",
        }.get(src, "📂 Dataset loaded")
        st.success(label)

        n_rows = len(st.session_state["df"])
        target = st.session_state.get("target_col", "?")
        st.caption(f"{n_rows:,} rows · target: `{target}`")
    else:
        st.info("No dataset loaded yet.\nGo to **📊 Dashboard** to load data.")

    if st.session_state.get("model_bundle") is not None:
        meta = st.session_state["model_bundle"]
        st.success(f"🤖 Model: **{meta.get('model_name', 'trained')}**")
    else:
        st.warning("No trained model yet.")

    st.markdown("---")
    st.caption(f"v{config.APP_VERSION}")

# ── Default home content ────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚡ Smart Home Energy Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered energy analytics and forecasting for your home</div>', unsafe_allow_html=True)

# Show a banner when the sample dataset is already pre-loaded
if st.session_state.get("dataset_source") == "demo" and st.session_state.get("df") is not None:
    st.success(
        "🧪 **Sample dataset pre-loaded** — you can explore every page right now! "
        "Head to **📊 Dashboard** → *Upload / Load Your Own Data* tab whenever you want to use your own data."
    )

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    #### 🚀 Getting Started
    A **sample dataset is pre-loaded** so you can explore immediately.

    To use your own data:
    1. Go to **📊 Dashboard** → *Upload / Load Your Own Data* tab
    2. Upload your CSV or enter a file path, then apply column selection
    3. Train an ML model on **🤖 Train ML Model**
    4. View forecasts on **🔮 Forecasting**
    5. Detect unusual patterns on **⚠️ Anomaly Detection**
    6. Estimate your bill on **💰 Bill Estimator**
    """)

with col2:
    st.markdown("""
    #### 📋 Supported Data Format
    ```csv
    DateTime,Global_active_power
    2024-01-01 00:00:00,1.25
    2024-01-01 01:00:00,1.18
    ```
    Any CSV with a **timestamp** and a **numeric energy column** will work.
    Column names are configurable.
    """)

with col3:
    st.markdown("""
    #### 🤖 ML Models Used
    - **Baseline**: Previous-day same-hour value
    - **Random Forest**: Ensemble of decision trees
    - **Gradient Boosting**: Sequential tree ensemble

    Models are evaluated on **chronological holdout** data
    to give realistic accuracy estimates.
    """)

st.markdown("---")
st.info(
    "👆 Use the **sidebar** or the page links above to navigate. "
    "A sample dataset is already loaded — or go to **📊 Dashboard** to use your own data."
)
