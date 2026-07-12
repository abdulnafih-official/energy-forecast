import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DATA_DIR, COMBINED_DATA_DIR

v2_data = "January 2024- June 2025.xlsx"

all_dfs = []

for file in os.listdir(RAW_DATA_DIR):
    if not file.endswith(".xlsx"):
        continue

    if file == v2_data:
        df_month = pd.read_excel(
            RAW_DATA_DIR / file,
            sheet_name="Report",
            header=0,
            engine="calamine"
        )
        df_month = df_month[["Timestamp", "Demand (MW)"]].rename(columns={
            "Timestamp": "datetime",
            "Demand (MW)": "demand_mw"
        })
        df_month["datetime"] = pd.to_datetime(df_month["datetime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
        df_month = df_month.dropna(subset=["datetime"])
    else:
        df_month = pd.read_excel(
            RAW_DATA_DIR / file,
            sheet_name="Sheet1",
            header=1,
            engine="calamine"
        )
        df_month = df_month[["Unnamed: 0", "NLDC_DEMAND|P"]].rename(columns={
            "Unnamed: 0": "datetime",
            "NLDC_DEMAND|P": "demand_mw"
        })
        df_month["datetime"] = pd.to_datetime(df_month["datetime"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        df_month = df_month.dropna(subset=["datetime"])

    all_dfs.append(df_month)
    print(f"✅ {file}")

df = pd.concat(all_dfs, ignore_index=True)

df.sort_values("datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nShape: {df.shape}")
print(df.head())
print(f"\nDate range: {df['datetime'].min()} → {df['datetime'].max()}")

output_path = COMBINED_DATA_DIR / "energy_combined.parquet"
df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")