"""
Storage abstraction layer: loads models, calibration, history, and feature columns.
- If S3 bucket name is configured in st.secrets, loads from S3.
- Otherwise, falls back to local filesystem (for development).
"""

import os
import joblib
import pandas as pd
import json
from pathlib import Path
import streamlit as st

# --- Path definitions (adjust if your folder structure differs) ---
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"   # as seen in the error
DATA_CLEANED_DIR = BASE_DIR / "data" / "cleaned"
DATA_FEATURES_DIR = BASE_DIR / "data" / "features"
CALIBRATION_PATH = BASE_DIR / "calibration.json"
HISTORY_PATH = DATA_CLEANED_DIR / "energy_cleaned.parquet"
FEATURES_PATH = DATA_FEATURES_DIR / "energy_features.parquet"

# -----------------------------------------------------------------
# Helper: decide whether to use S3 based on secrets
# -----------------------------------------------------------------
def _use_s3() -> bool:
    """Return True if S3 bucket name is configured in Streamlit secrets."""
    try:
        bucket = st.secrets.get("S3_BUCKET_NAME")
        return bool(bucket and bucket.strip())
    except (KeyError, AttributeError):
        return False

# -----------------------------------------------------------------
# Load functions (with Streamlit caching)
# -----------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all trained models from S3 (if configured) or local artifacts."""
    if _use_s3():
        from s3_utils import load_models_from_s3
        bucket = st.secrets["S3_BUCKET_NAME"]
        return load_models_from_s3(bucket)
    else:
        # Local development fallback
        return {
            "XGBoost": joblib.load(ARTIFACTS_DIR / "xgboost_model.joblib"),
            "LightGBM": joblib.load(ARTIFACTS_DIR / "lightgbm_model.joblib"),
            "Random Forest": joblib.load(ARTIFACTS_DIR / "random_forest_model.joblib"),
            "Prophet": joblib.load(ARTIFACTS_DIR / "prophet_model.joblib"),
        }

@st.cache_resource
def load_calibration():
    """Load calibration factors from S3 or local file."""
    if _use_s3():
        from s3_utils import load_calibration_from_s3
        bucket = st.secrets["S3_BUCKET_NAME"]
        return load_calibration_from_s3(bucket)
    else:
        with open(CALIBRATION_PATH, "r") as f:
            return json.load(f)

@st.cache_resource
def load_history():
    """Load historical energy data from S3 or local Parquet."""
    if _use_s3():
        from s3_utils import load_history_from_s3
        bucket = st.secrets["S3_BUCKET_NAME"]
        return load_history_from_s3(bucket)
    else:
        return pd.read_parquet(HISTORY_PATH)

@st.cache_resource
def load_feature_cols():
    """Load feature column list from S3 or local Parquet."""
    if _use_s3():
        from s3_utils import load_feature_cols_from_s3
        bucket = st.secrets["S3_BUCKET_NAME"]
        return load_feature_cols_from_s3(bucket)
    else:
        df = pd.read_parquet(FEATURES_PATH)
        # Assume the features are all columns except the target and datetime
        # Adjust according to your actual feature set.
        exclude = ["datetime", "target"]   # change if needed
        return [col for col in df.columns if col not in exclude]