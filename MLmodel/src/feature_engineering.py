from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "MLmodel" / "MLfiles"


def step_03_engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Check if input is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame, got {type(dataframe).__name__}"
        )

    selected_features = [
        "vertical_acceleration",
        "lateral_acceleration",
        "longitudinal_acceleration",
    ]

    # Map Finnish column names to English
    column_mapping = {
        "pys_kiiht": "vertical_acceleration",
        "siv_kiiht": "lateral_acceleration",
        "nyo_kiiht": "longitudinal_acceleration",
    }

    # Create mapped dataframe
    mapped_dataframe = dataframe.rename(columns=column_mapping)

    # Check if required columns exist in the mapped dataframe
    missing_features = [
        col for col in selected_features if col not in mapped_dataframe.columns
    ]
    if missing_features:
        raise ValueError(
            f"Required features are missing: {missing_features}. "
            f"Available columns: {list(mapped_dataframe.columns)}"
        )

    engineered_dataframe = mapped_dataframe.loc[:, selected_features].copy()

    # Print preview of engineered features
    print()
    print("------------------------------------------------------------")
    print("Feature engineering selected features.")
    print()
    print(engineered_dataframe.head().to_string())
    print()

    # Save feature metadata
    feature_metadata = {
        "pipeline_version": "0.1",
        "features": selected_features,
        "feature_count": len(selected_features),
        "data_rows": len(engineered_dataframe),
        "feature_engineering_date": pd.Timestamp.now().isoformat(),
        "description": "Road condition acceleration measurements for anomaly detection",
    }

    # Create output directory if it doesn't exist
    metadata_path = OUTPUT_DIR / "feature_metadata.json"
    metadata_path.parent.mkdir(exist_ok=True)

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=2)

    print(f"Feature metadata saved to: {metadata_path}")

    return engineered_dataframe
