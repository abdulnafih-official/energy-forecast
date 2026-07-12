# Energy Demand Forecasting

End-to-end energy demand forecasting app using NLDC/SCADA data.

## Stack
- Models: XGBoost, LightGBM, Prophet, Random Forest, Ensemble
- Backend: Python
- Frontend: Streamlit
- Cloud: AWS S3 + Lambda (ETL)

## Setup
```bash
python -m venv energy-forecast-venv
energy-forecast-venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Pipeline
1. `etl/convert.py` — raw xlsx → parquet
2. `src/clean.py` — resample to hourly, interpolate gaps
3. `src/features.py` — build model features
4. `src/train.py` — train models + calibrate
5. `app.py` — Streamlit app
