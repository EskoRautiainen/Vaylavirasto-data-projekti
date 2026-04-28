from __future__ import annotations

import pandas as pd


# -------------------------
# INDEX SUMMARY FORMATTING
# -------------------------
def _format_index_summary(indices: list[int], sample_size: int = 20) -> str:
    if not indices:
        return "none"
    if len(indices) <= sample_size:
        return f"{indices} (total: {len(indices)})"
    preview = ", ".join(str(i) for i in indices[:sample_size])
    return f"[{preview}, ...] (total: {len(indices)})"


# -------------------------
# DATA CLEANING
# -------------------------
def step_02_clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Check if input is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame, got {type(dataframe).__name__}"
        )
    
    # Check for empty dataframe
    if dataframe.empty:
        print("Warning: Empty dataframe provided for data cleaning!")
        return dataframe.copy()
    
    # Phase 1: Remove rows with missing values
    rows_with_missing_values = dataframe.index[dataframe.isna().any(axis=1)].tolist()
    dataframe_without_missing = dataframe.drop(index=rows_with_missing_values)
    
    # Phase 2: Remove rows with non-numeric values
    # Supports both comma (,) and period (.) as decimal separators
    def convert_with_comma_and_dot(value):
        if isinstance(value, str):
            # If comma found, replace it with period
            if ',' in value:
                value = value.replace(',', '.')
            # Convert to numeric
            try:
                return float(value)
            except ValueError:
                return None  # Return None if conversion fails
        return value
    
    # Use custom conversion
    converted_dataframe = dataframe_without_missing.map(convert_with_comma_and_dot)
    numeric_dataframe = pd.DataFrame(converted_dataframe, columns=dataframe_without_missing.columns)
    
    rows_with_non_numeric_values = numeric_dataframe.index[numeric_dataframe.isna().any(axis=1)].tolist()
    dataframe_with_numeric_only = numeric_dataframe.drop(index=rows_with_non_numeric_values)
    
    # Phase 3: Remove rows with negative values
    rows_with_negative_values = dataframe_with_numeric_only.index[
        (dataframe_with_numeric_only < 0).any(axis=1)
    ].tolist()
    cleaned_dataframe = dataframe_with_numeric_only.drop(index=rows_with_negative_values).reset_index(drop=True)
    
    # Calculate statistics
    original_rows = len(dataframe)
    cleaned_rows = len(cleaned_dataframe)
    
    # Adjust reporting to match three-phase process
    missing_mask = dataframe.isna().any(axis=1)
    non_numeric_mask = numeric_dataframe.isna().any(axis=1) & ~missing_mask.reindex(numeric_dataframe.index, fill_value=False)
    negative_mask = (dataframe_with_numeric_only < 0).any(axis=1)
    
    rows_with_missing_values = dataframe.index[missing_mask].tolist()
    rows_with_non_numeric_values = dataframe_without_missing.index[non_numeric_mask].tolist()
    rows_with_negative_values = dataframe_with_numeric_only.index[negative_mask].tolist()

    print()
    print("------------------------------------------------------------")
    print("Data cleaning statistics")
    print()
    
    # Phase 1: Missing values
    print(
        "Phase 1 - Removed rows with missing values: "
        f"{_format_index_summary(rows_with_missing_values)}"
    )
    
    # Phase 2: Non-numeric values
    print(
        "Phase 2 - Removed rows with non-numeric values: "
        f"{_format_index_summary(rows_with_non_numeric_values)}"
    )
    
    # Phase 3: Negative values
    print(
        "Phase 3 - Removed rows with negative values: "
        f"{_format_index_summary(rows_with_negative_values)}"
    )
    
    print(f"Rows remaining after cleaning: {cleaned_rows}")

    # Output validation: ensure data remains
    if cleaned_dataframe.empty:
        raise ValueError(
            "Data cleaning removed all rows. No data remains for further processing."
        )

    return cleaned_dataframe
