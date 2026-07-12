"""Project configuration and constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
COMBINED_DATA_DIR = PROCESSED_DATA_DIR / "combined"
CLEANED_DATA_DIR = PROCESSED_DATA_DIR / "cleaned"
