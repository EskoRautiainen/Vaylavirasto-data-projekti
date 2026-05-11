from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "MLmodel" / "MLfiles"


REQUIRED_COLUMNS = [
    "pys_kiiht",
    "siv_kiiht",
    "nyo_kiiht",
]

FEATURE_LABELS = {
    "pys_kiiht": "Vertical acceleration",
    "siv_kiiht": "Lateral acceleration",
    "nyo_kiiht": "Longitudinal acceleration",
}
ABSOLUTE_DEFAULT_CRITERIA = {
    "pys_kiiht": 0.08,
    "siv_kiiht": 2.0,
    "nyo_kiiht": 4.0,
}


# -------------------------
# DYNAMIC CRITERIA COMPUTATION
# -------------------------
def _compute_dynamic_criteria(
    dataframe: pd.DataFrame, required_columns: list[str], baseline_quantile: float
) -> dict[str, float]:
    criteria: dict[str, float] = {}
    for column in required_columns:
        criteria[column] = float(dataframe[column].quantile(baseline_quantile))
    return criteria


# -------------------------
# BASELINE METADATA SAVING
# -------------------------
def _save_baseline_metadata(
    criteria: dict[str, float],
    baseline_quantile: float | None,
    min_features_required: int,
    original_rows: int,
    filtered_rows: int,
    criteria_mode: str,
) -> None:
    metadata = {
        "baseline_quantile": baseline_quantile,
        "min_features_required": min_features_required,
        "criteria": criteria,
        "criteria_mode": criteria_mode,
        "original_rows": original_rows,
        "filtered_rows": filtered_rows,
        "retention_rate": (filtered_rows / original_rows) if original_rows else 0.0,
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    output_path = OUTPUT_DIR / "baseline_criteria.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Baseline criteria saved to: {output_path}")


# -------------------------
# GOOD ROAD FILTERING
# -------------------------
def step_04_filter_good_road(
    dataframe: pd.DataFrame,
    baseline_quantile: float = 0.25,
    min_features_required: int = 3,
    absolute_criteria: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Filters the dataframe to retain only road segments that represent good road surface conditions.

    This function computes speed-agnostic dynamic thresholds from current data
    using a quantile value, then filters rows that meet all criteria.

    Args:
        dataframe: Input DataFrame containing road measurement data
        baseline_quantile: Quantile used to compute dynamic baseline thresholds
        min_features_required: Minimum count of features that must satisfy thresholds

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

    if not (0.0 < baseline_quantile <= 1.0):
        raise ValueError(
            f"baseline_quantile must be in range (0.0, 1.0], got {baseline_quantile}"
        )

    # Check if required columns exist in the dataframe
    required_columns = REQUIRED_COLUMNS
    missing_columns = [col for col in required_columns if col not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            f"Required columns for good road filtering are missing: {missing_columns}. "
            f"Available columns: {list(dataframe.columns)}"
        )
    if not (1 <= min_features_required <= len(required_columns)):
        raise ValueError(
            f"min_features_required must be in range [1, {len(required_columns)}], "
            f"got {min_features_required}"
        )

    # Calculate filtering statistics before filtering for accurate reporting
    original_rows = len(dataframe)

    criteria_mode = "dynamic"
    if absolute_criteria is not None:
        missing_criteria = [col for col in required_columns if col not in absolute_criteria]
        if missing_criteria:
            raise ValueError(
                "absolute_criteria is missing required keys: "
                f"{missing_criteria}"
            )
        criteria = {col: float(absolute_criteria[col]) for col in required_columns}
        criteria_mode = "absolute"
    else:
        # Compute dynamic filtering criteria from current data
        criteria = _compute_dynamic_criteria(dataframe, required_columns, baseline_quantile)

    # Apply filtering: keep rows that satisfy at least min_features_required criteria.
    # This is more robust when no external reference set exists.
    threshold_hits = pd.DataFrame(
        {
            col: (dataframe[col] <= criteria[col]).astype(int)
            for col in required_columns
        }
    )
    mask = threshold_hits.sum(axis=1) >= min_features_required

    filtered_dataframe = dataframe[mask].copy()

    # Calculate filtering statistics
    filtered_rows = len(filtered_dataframe)
    removed_rows = original_rows - filtered_rows
    retention_rate = (filtered_rows / original_rows) * 100 if original_rows > 0 else 0

    print()
    print("------------------------------------------------------------")
    print("Good road filtering applied.")
    print()
    if criteria_mode == "absolute":
        print("Good-road selection thresholds (fixed absolute thresholds):")
    else:
        percentile_text = int(baseline_quantile * 100)
        print(
            "Good-road selection thresholds "
            f"(computed from current data, lower bound at the {percentile_text}th percentile):"
        )
    print(
        f"Selection rule: at least {min_features_required}/{len(required_columns)} "
        "features must be at or below their threshold."
    )
    for column, threshold in criteria.items():
        label = FEATURE_LABELS.get(column, column)
        print(f"  {label} ({column}) <= {threshold}")
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
    
    # Output validation: ensure data remains
    if filtered_dataframe.empty:
        raise ValueError(
            "Good road filtering removed all rows. No data remains for feature engineering."
        )

    _save_baseline_metadata(
        criteria,
        baseline_quantile if criteria_mode == "dynamic" else None,
        min_features_required,
        original_rows,
        filtered_rows,
        criteria_mode,
    )

    return filtered_dataframe
