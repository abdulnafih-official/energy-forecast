import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import COMBINED_DATA_DIR, CLEANED_DATA_DIR

# Load raw parquet
df = pd.read_parquet(COMBINED_DATA_DIR / "energy_combined.parquet")

# Set datetime as index for resampling
df = df.set_index("datetime")

# Resample to hourly mean
# Old files: mean of 360 ten-second readings per hour
# New file: already hourly, mean of 1 value = same value
df_hourly = df.resample("h").mean()

# Check gaps before filling
missing_hours = df_hourly["demand_mw"].isna().sum()
print(f"Missing hours before fill: {missing_hours}")

# Linear interpolation for missing hours
df_hourly["demand_mw"] = df_hourly["demand_mw"].interpolate(method="linear")

# Confirm no remaining nulls
remaining = df_hourly["demand_mw"].isna().sum()
print(f"Missing hours after fill: {remaining}")

# Reset index
df_hourly = df_hourly.reset_index()

print(f"\nShape: {df_hourly.shape}")
print(df_hourly.head())
print(f"\nDate range: {df_hourly['datetime'].min()} → {df_hourly['datetime'].max()}")

# Save
output_path = CLEANED_DATA_DIR / "energy_cleaned.parquet"
df_hourly.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")