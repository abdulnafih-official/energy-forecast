"""Feature engineering utilities."""

import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import CLEANED_DATA_DIR, FEATURES_DATA_DIR

# Load cleaned data
df = pd.read_parquet(CLEANED_DATA_DIR / "energy_cleaned.parquet")
df = df.set_index("datetime")

# Calendar features
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["day_of_month"] = df.index.day
df["month"] = df.index.month
df["quarter"] = df.index.quarter
df["year"] = df.index.year
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Lag features (raw past demand, no leakage since shift() only looks backward)
df["lag_1h"] = df["demand_mw"].shift(1)
df["lag_24h"] = df["demand_mw"].shift(24)
df["lag_168h"] = df["demand_mw"].shift(168)

# Drop rows with NaNs from lag windows (leading edge of the series only)
before = df.shape[0]
df = df.dropna()
after = df.shape[0]
print(f"Dropped {before - after} rows with NaN lag values")

df = df.reset_index()

print(f"\nShape: {df.shape}")
print(df.head())
print(f"\nColumns: {list(df.columns)}")

output_path = FEATURES_DATA_DIR / "energy_features.parquet"
df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")