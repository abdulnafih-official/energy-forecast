"""Optimized prediction interface for trained models."""

import numpy as np
import pandas as pd

LAG_HOURS = [1, 24, 168]
ROLLING_WINDOW_HOURS = 24


def build_feature_row(target_dt, history_series, feature_cols):
    """Build one row of calendar + lag + rolling features for target_dt from a history Series.
    
    Used only for fast single-step or historical lookups.
    """
    row = {
        "hour": target_dt.hour,
        "day_of_week": target_dt.dayofweek,
        "day_of_month": target_dt.day,
        "month": target_dt.month,
        "quarter": target_dt.quarter,
        "year": target_dt.year,
        "is_weekend": int(target_dt.dayofweek >= 5),
    }
    
    for lag in LAG_HOURS:
        lag_dt = target_dt - pd.Timedelta(hours=lag)
        if lag_dt not in history_series.index:
            raise ValueError(f"Not enough history to compute lag_{lag}h for {target_dt}")
        row[f"lag_{lag}h"] = float(history_series.loc[lag_dt])

    window_start = target_dt - pd.Timedelta(hours=ROLLING_WINDOW_HOURS)
    window_end = target_dt - pd.Timedelta(hours=1)
    window = history_series.loc[window_start:window_end]
    if len(window) < ROLLING_WINDOW_HOURS:
        raise ValueError(
            f"Not enough history to compute rolling_mean_{ROLLING_WINDOW_HOURS}h for {target_dt}"
        )
    row[f"rolling_mean_{ROLLING_WINDOW_HOURS}h"] = float(window.mean())

    clean_features = [col for col in feature_cols if col != "demand_mw"]
    return pd.DataFrame([row])[clean_features]


def forecast_tree_model(model, target_dt, history, feature_cols):
    """Direct prediction if target_dt is historical; ultra-fast vectorized recursive 
    step-forward prediction if target_dt is beyond the last known hour."""
    
    # Standardize 'history' to a flat Pandas Series
    if isinstance(history, pd.DataFrame):
        history_series = history["demand_mw"].copy()
    else:
        history_series = history.copy()

    last_dt = history_series.index.max()
    
    # If target is in the past, directly predict using the historical data
    if target_dt <= last_dt:
        row = build_feature_row(target_dt, history_series, feature_cols)
        return float(model.predict(row)[0])

    # --- ULTRA-FAST VECTORIZED RECURSIVE FORECASTING ---
    # 1. Determine the horizon size and create a datetime index for the future steps
    future_range = pd.date_range(start=last_dt + pd.Timedelta(hours=1), end=target_dt, freq='h')
    n_steps = len(future_range)

    # 2. Extract feature columns (ignoring demand_mw target)
    clean_features = [col for col in feature_cols if col != "demand_mw"]
    
    # Map each column to its index in the prediction matrix for instant array writes
    col_to_idx = {col: i for i, col in enumerate(clean_features)}

    # 3. Create pre-computed calendar feature arrays for the entire future range
    hours = future_range.hour.values
    dayofweeks = future_range.dayofweek.values
    dayofmonths = future_range.day.values
    months = future_range.month.values
    quarters = future_range.quarter.values
    years = future_range.year.values
    is_weekends = (dayofweeks >= 5).astype(int)

    # 4. Pre-allocate an array for the historical values plus our future predictions
    # This avoids resizing Pandas objects inside the loop
    max_needed_history_hours = max(LAG_HOURS + [ROLLING_WINDOW_HOURS])
    historical_tail = history_series.tail(max_needed_history_hours)
    
    # This flat array represents [ ...historical values... | ...future predictions (initially NaN)... ]
    full_history_arr = np.concatenate([historical_tail.values, np.zeros(n_steps)])
    hist_len = len(historical_tail)

    # 5. Fast pointer lookups using negative indexing
    # If history length is 168 and we are at step 0:
    # lag 1h is at index (hist_len + step - 1)
    # lag 24h is at index (hist_len + step - 24)
    # rolling window spans (hist_len + step - 24) to (hist_len + step)
    
    # Pre-allocate a single row array to pass to the model
    row_arr = np.empty((1, len(clean_features)))

    # Run the recursive simulation in NumPy space
    for step in range(n_steps):
        # Fill calendar features instantly
        row_arr[0, col_to_idx["hour"]] = hours[step]
        row_arr[0, col_to_idx["day_of_week"]] = dayofweeks[step]
        row_arr[0, col_to_idx["day_of_month"]] = dayofmonths[step]
        row_arr[0, col_to_idx["month"]] = months[step]
        row_arr[0, col_to_idx["quarter"]] = quarters[step]
        row_arr[0, col_to_idx["year"]] = years[step]
        row_arr[0, col_to_idx["is_weekend"]] = is_weekends[step]

        # Fill lag features safely using index arithmetic
        curr_idx = hist_len + step
        for lag in LAG_HOURS:
            row_arr[0, col_to_idx[f"lag_{lag}h"]] = full_history_arr[curr_idx - lag]

        # Fill rolling mean features using slicing on the NumPy array
        window_vals = full_history_arr[curr_idx - ROLLING_WINDOW_HOURS : curr_idx]
        row_arr[0, col_to_idx[f"rolling_mean_{ROLLING_WINDOW_HOURS}h"]] = window_vals.mean()

        # Build a single-row DataFrame instantly (avoids parsing a dict)
        df_row = pd.DataFrame(row_arr, columns=clean_features)
        
        # Predict the next step and write it directly back to our flat array
        pred = float(model.predict(df_row)[0])
        full_history_arr[curr_idx] = pred

    # Return the final predicted value
    return float(full_history_arr[-1])


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