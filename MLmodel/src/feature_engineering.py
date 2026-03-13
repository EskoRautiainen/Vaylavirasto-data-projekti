from __future__ import annotations

import pandas as pd


def step_04_engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    selected_features = [
        "Pysty_kiiht",
        "Sivuheilahdus_kiiht",
        "Nyökkimis_kiiht",
        "Yhdistetty_kiiht_rms"

    ]
    engineered_dataframe = dataframe.loc[:, selected_features].copy()

    print()
    print("------------------------------------------------------------")
    print("The data has been prepared for feature engineering.")
    print()
    print(engineered_dataframe.head(5).to_string())

    # Save results to Excel
    output_path = "MLmodel/output/engineered_features.xlsx"
    engineered_dataframe.to_excel(output_path, index=False)

    print()
    print(f"Feature engineered data saved to: {output_path}")

    return engineered_dataframe
