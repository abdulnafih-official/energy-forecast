"""Prediction interface for trained models."""

import numpy as np
import pandas as pd

LAG_HOURS = [1, 24, 168]
ROLLING_WINDOW_HOURS = 24


def build_feature_row(target_dt, history_series, feature_cols):
    """Build one row of calendar + lag + rolling features for target_dt from a history Series."""
    row = {
        "hour": target_dt.hour,
        "day_of_week": target_dt.dayofweek,
        "day_of_month": target_dt.day,
        "month": target_dt.month,
        "quarter": target_dt.quarter,
        "year": target_dt.year,
        "is_weekend": int(target_dt.dayofweek >= 5),
    }
    
    # 1. Fetch lags safely from our clean history series
    for lag in LAG_HOURS:
        lag_dt = target_dt - pd.Timedelta(hours=lag)
        if lag_dt not in history_series.index:
            raise ValueError(f"Not enough history to compute lag_{lag}h for {target_dt}")
        row[f"lag_{lag}h"] = float(history_series.loc[lag_dt])

    # 2. Compute rolling mean safely
    window_start = target_dt - pd.Timedelta(hours=ROLLING_WINDOW_HOURS)
    window_end = target_dt - pd.Timedelta(hours=1)
    window = history_series.loc[window_start:window_end]
    if len(window) < ROLLING_WINDOW_HOURS:
        raise ValueError(
            f"Not enough history to compute rolling_mean_{ROLLING_WINDOW_HOURS}h for {target_dt}"
        )
    row[f"rolling_mean_{ROLLING_WINDOW_HOURS}h"] = float(window.mean())

    # 3. Filter out 'demand_mw' from feature_cols to prevent KeyError mismatches
    clean_features = [col for col in feature_cols if col != "demand_mw"]

    return pd.DataFrame([row])[clean_features]


def forecast_tree_model(model, target_dt, history, feature_cols):
    """Direct prediction if target_dt is historical; recursive step-forward
    prediction (using this model's own outputs as lag/rolling inputs) if
    target_dt is beyond the last known hour."""
    
    # Standardize 'history' to a flat Pandas Series immediately to prevent DataFrame bugs
    if isinstance(history, pd.DataFrame):
        history_series = history["demand_mw"].copy()
    else:
        history_series = history.copy()

    last_dt = history_series.index.max()
    
    # If target is in the past, directly predict using the historical data
    if target_dt <= last_dt:
        row = build_feature_row(target_dt, history_series, feature_cols)
        return float(model.predict(row)[0])

    # If target is in the future, predict recursively step-by-step
    current_dt = last_dt + pd.Timedelta(hours=1)
    while current_dt <= target_dt:
        row = build_feature_row(current_dt, history_series, feature_cols)
        pred = float(model.predict(row)[0])
        history_series.loc[current_dt] = pred
        current_dt += pd.Timedelta(hours=1)
        
    return float(history_series.loc[target_dt])


def forecast_prophet(model, target_dt):
    future = pd.DataFrame({"ds": [target_dt]})
    return float(model.predict(future)["yhat"].iloc[0])


def run_forecast(model_choice, target_dt, models, calibration, history, feature_cols):
    """Returns (prediction, margin) for the chosen model."""
    if model_choice == "Ensemble":
        preds = []
        for name, model in models.items():
            if name == "Prophet":
                preds.append(forecast_prophet(model, target_dt))
            else:
                preds.append(forecast_tree_model(model, target_dt, history, feature_cols))
        return float(np.mean(preds)), calibration["ensemble"]["margin"]
    elif model_choice == "Prophet":
        return forecast_prophet(models["Prophet"], target_dt), calibration["prophet"]["margin"]
    else:
        key_map = {"XGBoost": "xgboost", "LightGBM": "lightgbm", "Random Forest": "random_forest"}
        pred = forecast_tree_model(models[model_choice], target_dt, history, feature_cols)
        return pred, calibration[key_map[model_choice]]["margin"]