"""Model training pipeline."""

import json
import sys
import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import FEATURES_DATA_DIR, ARTIFACTS_DIR, CALIBRATION_PATH

ALPHA = 0.05  # target 95% coverage

# Load features
df = pd.read_parquet(FEATURES_DATA_DIR / "energy_features.parquet")
feature_cols = [c for c in df.columns if c not in ("datetime", "demand_mw")]

# Chronological split: 70% train / 15% calibration / 15% test
# No shuffling — this is a time series, shuffling would leak future rows into training
n = len(df)
train_end = int(n * 0.70)
cal_end = int(n * 0.85)

train_df = df.iloc[:train_end]
cal_df = df.iloc[train_end:cal_end]
test_df = df.iloc[cal_end:]

X_train, y_train = train_df[feature_cols], train_df["demand_mw"]
X_cal, y_cal = cal_df[feature_cols], cal_df["demand_mw"]
X_test, y_test = test_df[feature_cols], test_df["demand_mw"]

print(f"Train: {len(train_df)} rows | Calibration: {len(cal_df)} rows | Test: {len(test_df)} rows")


def calibrate_and_evaluate(name, cal_preds, test_preds):
    """Split-conformal calibration on cal set, evaluation on test set."""
    margin = float(np.quantile(np.abs(y_cal.values - cal_preds), 1 - ALPHA))
    mae = float(np.mean(np.abs(y_test.values - test_preds)))
    rmse = float(np.sqrt(np.mean((y_test.values - test_preds) ** 2)))
    lower, upper = test_preds - margin, test_preds + margin
    coverage = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))
    print(f"{name}: MAE={mae:.2f} RMSE={rmse:.2f} margin={margin:.2f} coverage={coverage:.2%}")
    return {"alpha": ALPHA, "margin": margin, "mae": mae, "rmse": rmse, "coverage": coverage}


ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
calibration = {}
cal_preds_by_model = {}
test_preds_by_model = {}

# --- XGBoost ---
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
cal_preds_by_model["xgboost"] = xgb_model.predict(X_cal)
test_preds_by_model["xgboost"] = xgb_model.predict(X_test)
calibration["xgboost"] = calibrate_and_evaluate("xgboost", cal_preds_by_model["xgboost"], test_preds_by_model["xgboost"])
joblib.dump(xgb_model, ARTIFACTS_DIR / "xgboost_model.joblib")

# --- LightGBM ---
lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.fit(X_train, y_train)
cal_preds_by_model["lightgbm"] = lgbm_model.predict(X_cal)
test_preds_by_model["lightgbm"] = lgbm_model.predict(X_test)
calibration["lightgbm"] = calibrate_and_evaluate("lightgbm", cal_preds_by_model["lightgbm"], test_preds_by_model["lightgbm"])
joblib.dump(lgbm_model, ARTIFACTS_DIR / "lightgbm_model.joblib")

# --- Random Forest ---
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
cal_preds_by_model["random_forest"] = rf_model.predict(X_cal)
test_preds_by_model["random_forest"] = rf_model.predict(X_test)
calibration["random_forest"] = calibrate_and_evaluate("random_forest", cal_preds_by_model["random_forest"], test_preds_by_model["random_forest"])
joblib.dump(rf_model, ARTIFACTS_DIR / "random_forest_model.joblib")

# --- Prophet ---
# Prophet has its own built-in seasonality/trend model and only wants ds/y —
# it does not use the lag/calendar feature columns the other models use.
prophet_train = train_df[["datetime", "demand_mw"]].rename(columns={"datetime": "ds", "demand_mw": "y"})
prophet_model = Prophet()
prophet_model.fit(prophet_train)

prophet_cal_future = cal_df[["datetime"]].rename(columns={"datetime": "ds"})
prophet_test_future = test_df[["datetime"]].rename(columns={"datetime": "ds"})
cal_preds_by_model["prophet"] = prophet_model.predict(prophet_cal_future)["yhat"].values
test_preds_by_model["prophet"] = prophet_model.predict(prophet_test_future)["yhat"].values
calibration["prophet"] = calibrate_and_evaluate("prophet", cal_preds_by_model["prophet"], test_preds_by_model["prophet"])
joblib.dump(prophet_model, ARTIFACTS_DIR / "prophet_model.joblib")
# Note: Prophet's own serialization (prophet.serialize.model_to_json) is more
# robust across library version upgrades than joblib/pickle. Using joblib here
# to keep saving consistent with the other models — revisit if you upgrade Prophet later.

# --- Ensemble (simple average of the four models) ---
cal_ensemble = np.mean([cal_preds_by_model[m] for m in cal_preds_by_model], axis=0)
test_ensemble = np.mean([test_preds_by_model[m] for m in test_preds_by_model], axis=0)
calibration["ensemble"] = calibrate_and_evaluate("ensemble", cal_ensemble, test_ensemble)
# No separate model artifact to save — the ensemble is just the mean of the four saved models' predictions.

CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2))
print(f"\nSaved all model artifacts to {ARTIFACTS_DIR}")
print(f"Saved calibration to {CALIBRATION_PATH}")