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


DEFAULT_DATA_FILE_PATH = (
     Path(__file__).resolve().parent.parent
     / "Data"
     / "Paallystettyjen_teiden_lahtotiedot_ominaisuus_kuntotiedot_100m_L145695.xlsx"
 )


def resolve_file_path(file_path: Path) -> Path:
    if not isinstance(file_path, Path):
        raise TypeError(f"file_path must be a Path object, got {type(file_path)}")
    
    # Prevent absolute paths for security reasons
    if file_path.is_absolute():
        raise ValueError(
            "Absolute paths are not allowed for security reasons. "
            "Use relative paths from the repository root."
        )
    
    # Check for potentially dangerous path components
    if ".." in file_path.parts:
        raise ValueError(
            "Path traversal (..) is not allowed for security reasons."
        )
    
    if file_path == DEFAULT_DATA_FILE_PATH:
        return file_path

    if file_path.parent == Path("."):
        return DEFAULT_DATA_FILE_PATH.parent / file_path.name

    if file_path.parent == Path("Data"):
        return DEFAULT_DATA_FILE_PATH.parent / file_path.name

    if file_path.parent == Path("data"):
        return DEFAULT_DATA_FILE_PATH.parent / file_path.name

    if file_path.parent != Path("."):
        raise ValueError(
            "Only a file name or Data/<file name> is supported. Place the Excel file in the repository Data directory."
        )

    return DEFAULT_DATA_FILE_PATH.parent / file_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loads an Excel file from the repository Data directory and prints the column names and the first data row."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_FILE_PATH,
        help="Excel file name located in the repository Data directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = resolve_file_path(args.file_path)
    print()
    print("=== ML Pipeline Started ===")
    print()
    print("------------------------------------------------------------")
    
    try:
        filtered_dataframe = step_01_load_data(file_path)
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
        good_road_dataframe = step_04_filter_good_road(engineered_dataframe)
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
        anomaly_results, model = step_06_model_training(good_road_scaled, all_road_scaled)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}") from e
    
    print()
    print("------------------------------------------------------------")
    print()
    print("=== ML Pipeline Finished ===")
    print()


if __name__ == "__main__":
    main()
