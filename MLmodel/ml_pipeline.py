from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

from src.data_cleaning import step_02_clean_data
from src.feature_engineering import step_03_engineer_features
from src.good_road_filter import step_04_filter_good_road
from src.data_loading import step_01_load_data
from src.data_scaling import step_05_data_scaling
from src.model_training import step_06_model_training


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "Data"
SUPPORTED_EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
# Baseline calibration constants
BASELINE_QUANTILE = 0.40
MIN_FEATURES_REQUIRED = 2


def resolve_input_path(input_path: Path | None) -> Path:
    if input_path is None:
        return DEFAULT_DATA_DIR

    if not isinstance(input_path, Path):
        raise TypeError(f"input_path must be a Path object, got {type(input_path)}")

    if input_path.is_absolute():
        raise ValueError(
            "Absolute paths are not allowed for security reasons. "
            "Use relative paths from the repository root."
        )

    if ".." in input_path.parts:
        raise ValueError(
            "Path traversal (..) is not allowed for security reasons."
        )

    resolved = REPO_ROOT / input_path
    if not resolved.exists():
        raise FileNotFoundError(f"Input path not found: {resolved}")

    if resolved.is_dir():
        return resolved

    if resolved.is_file() and resolved.suffix.lower() in SUPPORTED_EXCEL_SUFFIXES:
        return resolved

    raise ValueError(
        "Input path must be a directory containing Excel files or an Excel file (.xlsx/.xlsm)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Loads Excel data for ML pipeline. "
            "By default reads from repository Data directory."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Optional relative path from repository root. "
            "Can be a directory containing Excel files (recommended) or a single Excel file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input_path)
    print()
    print("=== ML Pipeline Started ===")
    print()
    print("------------------------------------------------------------")
    
    try:
        filtered_dataframe = step_01_load_data(input_path)
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {e}") from e
    
    try:
        cleaned_dataframe = step_02_clean_data(filtered_dataframe)
    except Exception as e:
        raise RuntimeError(f"Data cleaning failed: {e}") from e
    
    try:
        # Feature engineering on ALL clean data
        engineered_dataframe = step_03_engineer_features(cleaned_dataframe)
    except Exception as e:
        raise RuntimeError(f"Feature engineering failed: {e}") from e

    try:
        # Good road filtering on engineered data (only needed features)
        good_road_dataframe = step_04_filter_good_road(
            engineered_dataframe,
            baseline_quantile=BASELINE_QUANTILE,
            min_features_required=MIN_FEATURES_REQUIRED,
        )
    except Exception as e:
        raise RuntimeError(f"Good road filtering failed: {e}") from e

    try:
        # Data scaling using good road baseline
        good_road_scaled, scaler = step_05_data_scaling(good_road_dataframe)
    except Exception as e:
        raise RuntimeError(f"Data scaling failed: {e}") from e

    try:
        # Scale all road data using fitted scaler
        all_road_scaled_data = scaler.transform(engineered_dataframe)
        all_road_scaled = pd.DataFrame(
            all_road_scaled_data, 
            columns=engineered_dataframe.columns, 
            index=engineered_dataframe.index
        )
        
        # Model training and anomaly detection
        anomaly_results, model = step_06_model_training(
            good_road_scaled,
            all_road_scaled,
            engineered_dataframe,
        )
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}") from e
    
    print()
    print("------------------------------------------------------------")
    print()
    print("=== ML Pipeline Finished ===")
    print()


if __name__ == "__main__":
    main()
