"""Utilities for downloading models and data from S3."""

import os
import json
import tempfile
from pathlib import Path

import boto3
import joblib
import pandas as pd
import streamlit as st


def get_s3_client():
    """Create S3 client using Streamlit secrets."""
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets.get("AWS_REGION", "ap-south-1"),
    )


@st.cache_resource
def load_models_from_s3(bucket_name):
    """Download and cache all model artifacts from S3."""
    s3 = get_s3_client()
    models = {}
    
    model_files = {
        "XGBoost": "xgboost_model.joblib",
        "LightGBM": "lightgbm_model.joblib",
        "Random Forest": "random_forest_model.joblib",
        "Prophet": "prophet_model.joblib",
    }
    
    try:
        for model_name, filename in model_files.items():
            key = f"models/artifacts/{filename}"
            
            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                s3.download_file(bucket_name, key, tmp.name)
                models[model_name] = joblib.load(tmp.name)
                os.unlink(tmp.name)
            
            print(f"✅ Loaded {model_name} from S3")
        
        return models
    except Exception as e:
        raise RuntimeError(f"Failed to load models from S3: {str(e)}")


@st.cache_resource
def load_calibration_from_s3(bucket_name):
    """Download and cache calibration JSON from S3."""
    s3 = get_s3_client()
    
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
            s3.download_file(bucket_name, "models/calibration.json", tmp.name)
            with open(tmp.name) as f:
                calibration = json.load(f)
            os.unlink(tmp.name)
        
        print("✅ Loaded calibration from S3")
        return calibration
    except Exception as e:
        raise RuntimeError(f"Failed to load calibration from S3: {str(e)}")


@st.cache_resource
def load_history_from_s3(bucket_name):
    """Download and cache cleaned data from S3."""
    s3 = get_s3_client()
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
            s3.download_file(bucket_name, "data/energy_cleaned.parquet", tmp.name)
            df = pd.read_parquet(tmp.name)
            os.unlink(tmp.name)
        
        # Convert to series indexed by datetime for lag feature computation
        history = df.set_index("datetime")["demand_mw"].sort_index()
        print("✅ Loaded history from S3")
        return history
    except Exception as e:
        raise RuntimeError(f"Failed to load history from S3: {str(e)}")


def download_all_from_s3(bucket_name):
    """Wrapper to download models, calibration, and history with error handling."""
    try:
        models = load_models_from_s3(bucket_name)
        calibration = load_calibration_from_s3(bucket_name)
        history = load_history_from_s3(bucket_name)
        
        return models, calibration, history
    except RuntimeError as e:
        st.error(f"⚠️ Failed to load from S3: {str(e)}")
        st.info(
            "**Troubleshooting:**\n"
            "- Check AWS credentials in Streamlit secrets\n"
            "- Verify bucket name and object keys exist\n"
            "- Confirm IAM user has S3ReadOnly permissions"
        )
        raise

@st.cache_resource
def load_feature_cols_from_s3(bucket_name):
    """Download feature column list from S3 and return as a list."""
    s3 = get_s3_client()
    try:
        # Adjust the key to match your bucket structure
        key = "data/features/energy_features.parquet"   # same as in storage.py

        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
            s3.download_file(bucket_name, key, tmp.name)
            df = pd.read_parquet(tmp.name)
            os.unlink(tmp.name)

        # Exclude datetime and target columns (adjust if needed)
        exclude = ["datetime", "target"]
        feature_cols = [col for col in df.columns if col not in exclude]
        print("✅ Loaded feature columns from S3")
        return feature_cols
    except Exception as e:
        raise RuntimeError(f"Failed to load feature columns from S3: {str(e)}")