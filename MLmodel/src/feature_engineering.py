from __future__ import annotations

import pandas as pd


def step_04_engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Check if input is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame, got {type(dataframe).__name__}"
        )
    
    selected_features = [
        "Pysty_kiiht",
        "Sivuheilahdus_kiiht",
        "Nyökkimis_kiiht",
    ]
    
    # Check if required columns exist in the dataframe
    missing_features = [col for col in selected_features if col not in dataframe.columns]
    if missing_features:
        raise ValueError(
            f"Required features are missing: {missing_features}. "
            f"Available columns: {list(dataframe.columns)}"
        )
    
    engineered_dataframe = dataframe.loc[:, selected_features].copy()

    print()
    print("------------------------------------------------------------")
    print("The data has been prepared for feature engineering.")
    print()
    print(engineered_dataframe.head(5).to_string())

    return engineered_dataframe
