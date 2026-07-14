


import json
import sys
import os
import datetime as dt

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import CLEANED_DATA_DIR, FEATURES_DATA_DIR, ARTIFACTS_DIR, CALIBRATION_PATH
from predict import LAG_HOURS, run_forecast

LONG_HORIZON_WARNING_HOURS = 24  # beyond this, flag that the CI understates compounding error

st.set_page_config(page_title="Energy Demand Forecast", page_icon="⚡", layout="centered")


# ---------------------------------------------------------------------------
# Real pipeline: load trained models, calibration, and history (unchanged
# from the working version — only the UI around this is new)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    return {
        "XGBoost": joblib.load(ARTIFACTS_DIR / "xgboost_model.joblib"),
        "LightGBM": joblib.load(ARTIFACTS_DIR / "lightgbm_model.joblib"),
        "Random Forest": joblib.load(ARTIFACTS_DIR / "random_forest_model.joblib"),
        "Prophet": joblib.load(ARTIFACTS_DIR / "prophet_model.joblib"),
    }


@st.cache_data
def load_calibration():
    return json.loads(CALIBRATION_PATH.read_text())


@st.cache_data
def load_history():
    cleaned = pd.read_parquet(CLEANED_DATA_DIR / "energy_cleaned.parquet")
    return cleaned.set_index("datetime")["demand_mw"].sort_index()


@st.cache_data
def load_feature_cols():
    features = pd.read_parquet(FEATURES_DATA_DIR / "energy_features.parquet")
    return [c for c in features.columns if c not in ("datetime", "demand_mw")]


models = load_models()
calibration = load_calibration()
history = load_history()
feature_cols = load_feature_cols()
min_valid_dt = history.index.min() + pd.Timedelta(hours=max(LAG_HOURS))
last_historical_dt = history.index.max()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ---------------------------------------------------------------------------
# CSS — dark navy / tech / energy theme, glassmorphism, animated background orb
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0A1628;
    --bg-secondary: #0F2040;
    --accent-start: #1E90FF;
    --accent-end: #00D4FF;
    --text-primary: #E8F0FE;
    --text-secondary: #8AA8D8;
    --glass-bg: rgba(19, 35, 71, 0.55);
    --glass-border: rgba(30, 58, 106, 0.6);
}

.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

/* Animated glowing energy orb, fixed behind everything */
.bg-orb {
    position: fixed;
    top: -20%;
    left: 50%;
    width: 900px;
    height: 900px;
    transform: translateX(-50%);
    background: radial-gradient(circle, rgba(30,144,255,0.25) 0%, rgba(0,212,255,0.08) 40%, transparent 70%);
    filter: blur(40px);
    z-index: 0;
    animation: orbPulse 8s ease-in-out infinite;
    pointer-events: none;
}
@keyframes orbPulse {
    0%, 100% { opacity: 0.6; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-48%) scale(1.08); }
}

.gradient-text {
    background: linear-gradient(90deg, var(--accent-start), var(--accent-end));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.app-title {
    font-weight: 800;
    font-size: 2.1rem;
    margin-bottom: 0;
}
.app-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 0.2rem;
}

/* Info icon button (circular) */
.st-key-info_icon button {
    border-radius: 50%;
    width: 42px;
    height: 42px;
    padding: 0;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    font-size: 18px;
    float: right;
}

/* Main glass card wrapping the form */
.st-key-main_card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
}

/* Get Prediction button — gradient, hover scale + glow */
.st-key-predict_btn button {
    background: linear-gradient(90deg, var(--accent-start), var(--accent-end));
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    width: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.st-key-predict_btn button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 22px rgba(30, 144, 255, 0.6);
}

/* Modal backdrop (invisible full-viewport button — click to close) */
.st-key-modal_backdrop button,
.st-key-about_backdrop button {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(5, 12, 24, 0.75) !important;
    border: none !important;
    border-radius: 0 !important;
    z-index: 998 !important;
}

/* Modal card */
.st-key-modal_card, .st-key-about_card {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1000;
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 2rem;
    width: min(90vw, 440px);
    max-height: 80vh;
    overflow-y: auto;
    animation: modalFadeIn 0.25s ease;
}
@keyframes modalFadeIn {
    from { opacity: 0; transform: translate(-50%, -48%); }
    to { opacity: 1; transform: translate(-50%, -50%); }
}

/* Close "x" buttons inside modals */
.st-key-close_prediction button, .st-key-close_about button {
    position: absolute;
    top: 14px;
    right: 14px;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    padding: 0;
    z-index: 1001;
}

.prediction-range {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent-start), var(--accent-end));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0.5rem 0;
}

.footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--glass-border);
    color: var(--text-secondary);
    font-size: 0.85rem;
}

@media (max-width: 600px) {
    .app-title { font-size: 1.5rem; }
    .st-key-main_card, .st-key-modal_card, .st-key-about_card { padding: 1.2rem; }
    .footer { flex-direction: column; gap: 0.4rem; text-align: center; }
}
</style>
<div class="bg-orb"></div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header_col1, header_col2 = st.columns([6, 1])
with header_col1:
    st.markdown(
        '<div class="app-title">Energy Demand <span class="gradient-text">Forecast</span></div>'
        '<div class="app-subtitle">Hourly grid demand forecasts from real NLDC/SCADA data</div>',
        unsafe_allow_html=True,
    )
with header_col2:
    with st.container(key="info_icon"):
        if st.button("ℹ️", key="info_icon_btn"):
            st.session_state.show_about = True


# ---------------------------------------------------------------------------
# Main form card
# ---------------------------------------------------------------------------
with st.container(key="main_card"):
    picked_date = st.date_input(
        "Date",
        value=dt.date.today(),
        min_value=min_valid_dt.date(),
    )
    picked_hour = st.selectbox(
        "Hour", options=list(range(24)), format_func=lambda h: f"{h:02d}:00"
    )
    model_choice = st.selectbox(
        "Model", options=["XGBoost", "LightGBM", "Random Forest", "Prophet", "Ensemble"]
    )
    if st.button("Get Prediction", key="predict_btn"):
        target_dt = pd.Timestamp(dt.datetime.combine(picked_date, dt.time(hour=picked_hour)))
        if target_dt < min_valid_dt:
            st.error(
                f"Pick a date on or after {min_valid_dt.date()} — earlier dates "
                "don't have enough history for the lag features."
            )
        else:
            with st.spinner("Computing forecast..."):
                prediction, margin = run_forecast(
                    model_choice, target_dt, models, calibration, history, feature_cols
                )
            horizon_hours = max(0, (target_dt - last_historical_dt) / pd.Timedelta(hours=1))
            st.session_state.prediction_result = {
                "model": model_choice,
                "datetime": target_dt,
                "lower": prediction - margin,
                "upper": prediction + margin,
                "horizon_hours": horizon_hours,
            }
            st.session_state.show_prediction = True


# ---------------------------------------------------------------------------
# Prediction modal
# ---------------------------------------------------------------------------
if st.session_state.show_prediction and st.session_state.prediction_result:
    result = st.session_state.prediction_result

    with st.container(key="modal_backdrop"):
        if st.button(" ", key="modal_backdrop_btn"):
            st.session_state.show_prediction = False
            st.rerun()

    with st.container(key="modal_card"):
        if st.button("✕", key="close_prediction"):
            st.session_state.show_prediction = False
            st.rerun()

        st.markdown(f"**Model:** {result['model']}")
        st.markdown(f"**Date & hour:** {result['datetime'].strftime('%Y-%m-%d %H:%M')}")
        st.markdown(
            f'<div class="prediction-range">{result["lower"]:.1f} – {result["upper"]:.1f} MW</div>',
            unsafe_allow_html=True,
        )
        st.caption("95% prediction interval, from real conformal calibration on held-out data.")

        if result["horizon_hours"] > LONG_HORIZON_WARNING_HOURS:
            st.warning(
                f"This date is {result['horizon_hours']:.0f} hours beyond the last known "
                "data point (June 2025), so it was forecast recursively. The interval above "
                "is calibrated for one-step-ahead error and does not account for compounding "
                "error over a multi-step recursive forecast — true uncertainty at this horizon "
                "is larger than shown."
            )


# ---------------------------------------------------------------------------
# About modal
# ---------------------------------------------------------------------------
if st.session_state.show_about:
    with st.container(key="about_backdrop"):
        if st.button(" ", key="about_backdrop_btn"):
            st.session_state.show_about = False
            st.rerun()

    with st.container(key="about_card"):
        if st.button("✕", key="close_about"):
            st.session_state.show_about = False
            st.rerun()

        st.markdown("### About this app")
        st.markdown(
            "Energy Demand Forecast predicts hourly electricity demand (MW) for the "
            "power grid, using models trained on real NLDC/SCADA data covering "
            "**September 2021 – June 2025** (~33,000 hourly readings, after cleaning "
            "and interpolating gaps in the raw feed)."
        )
        st.markdown("**Models:**")
        st.markdown(
            "- **XGBoost / LightGBM / Random Forest** — tree-based models using "
            "calendar features and recent demand history (lag features)\n"
            "- **Prophet** — trend/seasonality model, doesn't use recent demand history\n"
            "- **Ensemble** — average of all four"
        )
        st.markdown(
            "Every prediction interval is a genuine **95% conformal prediction interval**, "
            "calibrated on a held-out slice of real historical data — not a fixed or "
            "assumed margin."
        )
        st.markdown("**Tech stack:** Python, pandas, NumPy, scikit-learn, XGBoost, "
                     "LightGBM, Prophet, Streamlit")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer">'
    '<span>© 2026 All Rights Reserved</span>'
    '<span>♦ Made with precision by Nafih</span>'
    "</div>",
    unsafe_allow_html=True,
)