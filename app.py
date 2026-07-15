import sys
import os
import datetime as dt

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from storage import load_models, load_calibration, load_history, load_feature_cols
from predict import LAG_HOURS, run_forecast

LONG_HORIZON_WARNING_HOURS = 24  # beyond this, flag that the CI understates compounding error

st.set_page_config(page_title="Energy Demand Forecast", page_icon="⚡", layout="centered")


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
# ---------------------------------------------------------------------------
# CSS — dark navy / tech / energy theme, glassmorphism, animated background orb
# ---------------------------------------------------------------------------
def load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css(os.path.join(os.path.dirname(__file__), "static", "style.css"))
st.markdown('<div class="bg-orb"></div>', unsafe_allow_html=True)


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