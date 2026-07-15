"""Storage abstraction for models and data (S3 or local)."""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import CLEANED_DATA_DIR, FEATURES_DATA_DIR, ARTIFACTS_DIR, CALIBRATION_PATH


def is_streamlit_cloud():
    """Check if running on Streamlit Community Cloud."""
    return "STREAMLIT_RUNTIME_MEDIA_MUTED" in os.environ


@st.cache_resource
def load_models():
    """Load models from S3 (Streamlit Cloud) or local filesystem (dev)."""
    if is_streamlit_cloud():
        from s3_utils import load_models_from_s3
        bucket_name = st.secrets.get("S3_BUCKET_NAME")
        if not bucket_name:
            st.error("S3_BUCKET_NAME not found in secrets. See deployment guide.")
            raise ValueError("S3_BUCKET_NAME not configured")
        return load_models_from_s3(bucket_name)
    else:
        # Local development
        return {
            "XGBoost": joblib.load(ARTIFACTS_DIR / "xgboost_model.joblib"),
            "LightGBM": joblib.load(ARTIFACTS_DIR / "lightgbm_model.joblib"),
            "Random Forest": joblib.load(ARTIFACTS_DIR / "random_forest_model.joblib"),
            "Prophet": joblib.load(ARTIFACTS_DIR / "prophet_model.joblib"),
        }


@st.cache_data
def load_calibration():
    """Load calibration from S3 (Streamlit Cloud) or local filesystem (dev)."""
    if is_streamlit_cloud():
        from s3_utils import load_calibration_from_s3
        bucket_name = st.secrets.get("S3_BUCKET_NAME")
        if not bucket_name:
            st.error("S3_BUCKET_NAME not found in secrets. See deployment guide.")
            raise ValueError("S3_BUCKET_NAME not configured")
        return load_calibration_from_s3(bucket_name)
    else:
        return json.loads(CALIBRATION_PATH.read_text())


@st.cache_data
def load_history():
    """Load historical data from S3 (Streamlit Cloud) or local filesystem (dev)."""
    if is_streamlit_cloud():
        from s3_utils import load_history_from_s3
        bucket_name = st.secrets.get("S3_BUCKET_NAME")
        if not bucket_name:
            st.error("S3_BUCKET_NAME not found in secrets. See deployment guide.")
            raise ValueError("S3_BUCKET_NAME not configured")
        return load_history_from_s3(bucket_name)
    else:
        cleaned = pd.read_parquet(CLEANED_DATA_DIR / "energy_cleaned.parquet")
        return cleaned.set_index("datetime")["demand_mw"].sort_index()


@st.cache_data
def load_feature_cols():
    """Load feature column names from S3 or local (infer from history)."""
    if is_streamlit_cloud():
        # On cloud, infer from cached history
        history = load_history()
        return ["hour", "day_of_week", "day_of_month", "month", "quarter", "year", "is_weekend",
                "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h"]
    else:
        features = pd.read_parquet(FEATURES_DATA_DIR / "energy_features.parquet")
        return [c for c in features.columns if c not in ("datetime", "demand_mw")]