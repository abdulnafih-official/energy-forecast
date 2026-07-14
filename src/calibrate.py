"""Conformal calibration utilities."""

import numpy as np


def calibrate_and_evaluate(name, y_cal, cal_preds, y_test, test_preds, alpha=0.05):
    """Split-conformal calibration on cal set, evaluation on test set."""
    margin = float(np.quantile(np.abs(y_cal.values - cal_preds), 1 - alpha))
    mae = float(np.mean(np.abs(y_test.values - test_preds)))
    rmse = float(np.sqrt(np.mean((y_test.values - test_preds) ** 2)))
    lower, upper = test_preds - margin, test_preds + margin
    coverage = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))
    print(f"{name}: MAE={mae:.2f} RMSE={rmse:.2f} margin={margin:.2f} coverage={coverage:.2%}")
    return {"alpha": alpha, "margin": margin, "mae": mae, "rmse": rmse, "coverage": coverage}
