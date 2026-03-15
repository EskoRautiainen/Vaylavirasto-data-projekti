from __future__ import annotations

import pandas as pd


# Good road filtering criteria based on 75th percentile values for all speed data
# These values are derived from all road measurements (speed-agnostic approach)
GOOD_ROAD_CRITERIA = {
    "Pysty_kiiht": 0.13,
    "Sivuheilahdus_kiiht": 3.0,
    "Nyökkimis_kiiht": 6.0,
}


def step_03_filter_good_road(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataframe to retain only road segments that represent good road surface conditions.

    This function uses speed-agnostic thresholds based on 75th percentile values from all speed data,
    providing a broader baseline for anomaly detection across different speed limits.

    Args:
        dataframe: Input DataFrame containing road measurement data

    Returns:
        DataFrame containing only good road segments

    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If required columns are missing or no data remains after filtering
    """
    # Check if input is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame, got {type(dataframe).__name__}"
        )

    # Check for empty dataframe
    if dataframe.empty:
        print("Warning: Empty dataframe provided for good road filtering!")
        return dataframe.copy()

    # Check if required columns exist in the dataframe
    required_columns = list(GOOD_ROAD_CRITERIA.keys())
    missing_columns = [col for col in required_columns if col not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            f"Required columns for good road filtering are missing: {missing_columns}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    # Calculate filtering statistics before filtering for accurate reporting
    original_rows = len(dataframe)

    # Apply filtering: keep only rows that meet ALL criteria
    # Use direct boolean indexing for better performance
    mask = (dataframe[required_columns[0]] <= GOOD_ROAD_CRITERIA[required_columns[0]])
    for col in required_columns[1:]:
        mask &= (dataframe[col] <= GOOD_ROAD_CRITERIA[col])

    filtered_dataframe = dataframe[mask].copy()

    # Calculate filtering statistics
    filtered_rows = len(filtered_dataframe)
    removed_rows = original_rows - filtered_rows
    retention_rate = (filtered_rows / original_rows) * 100 if original_rows > 0 else 0

    print()
    print("------------------------------------------------------------")
    print("Good road filtering applied.")
    print()
    print("Filtering criteria (25th percentile thresholds for all speed data):")
    for column, threshold in GOOD_ROAD_CRITERIA.items():
        print(f"  {column} ≤ {threshold}")
    print()
    print(f"Original rows: {original_rows}")
    print(f"Rows after filtering: {filtered_rows}")
    print(f"Removed rows: {removed_rows}")
    print(f"Data retention rate: {retention_rate:.1f}%")
    print()
    if filtered_rows > 0:
        print("Preview of filtered good road data:")
        print(filtered_dataframe.head(5).to_string())
    else:
        print("Warning: No data remaining after good road filtering!")
    print()

    # Output validation: ensure data remains
    if filtered_dataframe.empty:
        raise ValueError(
            "Good road filtering removed all rows. No data remains for feature engineering."
        )

    return filtered_dataframe
