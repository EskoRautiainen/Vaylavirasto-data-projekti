from __future__ import annotations

import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def step_05_data_scaling(good_road_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scales features using StandardScaler for ML model compatibility.
    
    This function fits StandardScaler on the good road baseline data and returns
    both the scaled data and the fitted scaler for consistent transformation
    of all road data.
    
    Args:
        good_road_dataframe: DataFrame containing good road baseline data
        
    Returns:
        Tuple of (scaled_dataframe, scaler) where:
        - scaled_dataframe: DataFrame with scaled features
        - scaler: Fitted StandardScaler object for transforming other data
        
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
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
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
    print("Data scaling applied using StandardScaler")
    print()
    print("Baseline statistics (good roads):")
    
    means_dict = scaled_dataframe.mean().round(3).to_dict()
    std_dict = scaled_dataframe.std().round(3).to_dict()
    
    print("  Feature means:")
    for feature, value in means_dict.items():
        print(f"    {feature}: {value}")
    
    print("  Feature std:")
    for feature, value in std_dict.items():
        print(f"    {feature}: {value}")
    
    print(f"  Scaled rows: {len(scaled_dataframe)}")
    
    # Save scaler for future use
    scaler_path = Path("MLfiles/scaler.pkl")
    scaler_path.parent.mkdir(exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to: {scaler_path}")
    
    return scaled_dataframe, scaler
