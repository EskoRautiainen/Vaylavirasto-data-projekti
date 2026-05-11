from __future__ import annotations

import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "MLmodel" / "MLfiles"
ZERO_TOLERANCE = 1e-9


# -------------------------
# DATA SCALING
# -------------------------
def step_05_data_scaling(good_road_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, RobustScaler]:
    """
    Scales features using RobustScaler for ML model compatibility.
    
    This function fits RobustScaler on the good road baseline data and returns
    both the scaled data and the fitted scaler for consistent transformation
    of all road data.
    
    Args:
        good_road_dataframe: DataFrame containing good road baseline data
        
    Returns:
        Tuple of (scaled_dataframe, scaler) where:
        - scaled_dataframe: DataFrame with scaled features
        - scaler: Fitted RobustScaler object for transforming other data
        
    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If input dataframe is empty
    """
    # Check if input is a pandas DataFrame
    if not isinstance(good_road_dataframe, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame, got {type(good_road_dataframe).__name__}"
        )
    
    # Check for empty dataframe
    if good_road_dataframe.empty:
        raise ValueError("Input dataframe is empty for scaling!")
    
    # Initialize RobustScaler (median + IQR), less sensitive to outliers.
    scaler = RobustScaler()
    
    # Fit and transform the good road data
    scaled_data = scaler.fit_transform(good_road_dataframe)
    
    # Create DataFrame with scaled data
    scaled_dataframe = pd.DataFrame(
        scaled_data, 
        columns=good_road_dataframe.columns, 
        index=good_road_dataframe.index
    )
    
    # Print scaling statistics
    print()
    print("------------------------------------------------------------")
    print("Data scaling completed (RobustScaler)")
    print()
    print("Baseline scaling summary (good roads):")

    medians_dict = scaled_dataframe.median().to_dict()
    q25_dict = scaled_dataframe.quantile(0.25).to_dict()
    q75_dict = scaled_dataframe.quantile(0.75).to_dict()

    print("  Scaled medians (target is ~0):")
    for feature, value in medians_dict.items():
        print(f"    {feature}: {value:.6f}")

    print("  Scaled Q1/Q3 (spread around baseline):")
    for feature in scaled_dataframe.columns:
        print(f"    {feature}: Q1={q25_dict[feature]:.6f}, Q3={q75_dict[feature]:.6f}")
    
    print(f"  Scaled rows: {len(scaled_dataframe)}")
    print()
    print("Quick interpretation:")
    print("  0 = baseline center. Negative = below baseline. Positive = above baseline.")
    print("  Q1/Q3 = where the middle 50% of good-road baseline data lies after scaling.")
    print("  Values are not limited to [-1, 1]; larger absolute values are possible.")
    nonzero_medians = [
        feature for feature, value in medians_dict.items() if abs(value) > ZERO_TOLERANCE
    ]
    if nonzero_medians:
        print("  Scaling status: CHECK - some scaled medians are not near 0.")
        print(f"  Features to inspect: {', '.join(nonzero_medians)}")
    else:
        print("  Scaling status: OK - all scaled medians are near 0.")
    
    # Save scaler for future use
    scaler_path = OUTPUT_DIR / "scaler.pkl"
    scaler_path.parent.mkdir(exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to: {scaler_path}")
    
    return scaled_dataframe, scaler
