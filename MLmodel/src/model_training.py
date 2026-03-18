from __future__ import annotations

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest


def step_06_model_training(good_road_scaled: pd.DataFrame, all_road_scaled: pd.DataFrame) -> tuple[pd.DataFrame, IsolationForest]:
    """
    Trains Isolation Forest model for anomaly detection.
    
    This function trains an Isolation Forest model on good road baseline data
    and makes predictions on all road data to identify anomalies.
    
    Args:
        good_road_scaled: Scaled good road baseline data
        all_road_scaled: Scaled all road data
        
    Returns:
        Tuple of (results_dataframe, trained_model) where:
        - results_dataframe: DataFrame with anomaly predictions and scores
        - trained_model: Fitted IsolationForest model
        
    Raises:
        TypeError: If inputs are not pandas DataFrames
        ValueError: If input dataframes are empty
    """
    # Check inputs
    if not isinstance(good_road_scaled, pd.DataFrame) or not isinstance(all_road_scaled, pd.DataFrame):
        raise TypeError("Both inputs must be pandas DataFrames")
    
    if good_road_scaled.empty or all_road_scaled.empty:
        raise ValueError("Input dataframes cannot be empty")
    
    # Model parameters
    contamination = 'auto'    # Let algorithm determine contamination
    n_estimators = 200
    random_state = 42
    
    # Initialize model
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state
    )
    
    # Get feature names
    features = good_road_scaled.columns.tolist()
    
    # Train model on good roads baseline
    model.fit(good_road_scaled[features])
    
    # Make predictions on all roads
    predictions = model.predict(all_road_scaled[features])
    scores = model.decision_function(all_road_scaled[features])
    
    # Create results dataframe
    results = all_road_scaled.copy()
    results['anomaly_prediction'] = predictions
    results['anomaly_score'] = scores
    
    # Add anomaly classification
    results['anomaly_type'] = results['anomaly_prediction'].map({
        1: 'Normal',
        -1: 'Anomaly'
    })
    
    # Add percentile-based categorization
    def categorize_by_fixed_thresholds(scores):
        """Categorize anomaly scores using fixed thresholds based on data distribution."""
        
        def categorize(score):
            if score <= -0.15:      # Vahvat poikkeamat
                return 'Critical'
            elif score <= -0.08:    # Keskivahvat poikkeamat
                return 'Poor'
            elif score <= -0.03:    # Keskinkertaiset poikkeamat
                return 'Fair'
            elif score <= 0.02:     # Lähellä normaalia
                return 'Good'
            else:                  # Hyvät tiet
                return 'Excellent'
        
        return scores.apply(categorize)
    
    results['anomaly_category'] = categorize_by_fixed_thresholds(results['anomaly_score'])
    
    # Add priority scoring
    priority_mapping = {
        'Critical': 1,
        'Poor': 2,
        'Fair': 3,
        'Good': 4,
        'Excellent': 5
    }
    results['priority_score'] = results['anomaly_category'].map(priority_mapping)
    
    # Calculate statistics
    anomaly_count = sum(predictions == -1)
    normal_count = sum(predictions == 1)
    total_count = len(predictions)
    
    # Print results
    print()
    print("------------------------------------------------------------")
    print("Model training completed using Isolation Forest")
    print()
    print("Model parameters:")
    print(f"  Contamination: {contamination} (auto-determined)")
    print(f"  N_estimators: {n_estimators}")
    print(f"  Random state: {random_state}")
    print("  Features:")
    for feature in features:
        print(f"    {feature}")
    print()
    print("Training results:")
    print(f"  Normal roads: {normal_count} ({normal_count/total_count*100:.1f}%)")
    print(f"  Anomalous roads: {anomaly_count} ({anomaly_count/total_count*100:.1f}%)")
    print(f"  Total roads: {total_count}")
    print(f"  Actual contamination rate: {anomaly_count/total_count:.4f}")
    print()
    print("Anomaly category distribution:")
    category_counts = results['anomaly_category'].value_counts()
    for category in ['Critical', 'Poor', 'Fair', 'Good', 'Excellent']:
        count = category_counts.get(category, 0)
        percentage = (count / total_count) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    print()
    
    # Show fixed thresholds
    print("Fixed thresholds:")
    print(f"  Critical (≤ -0.15): Vahvat poikkeamat")
    print(f"  Poor (-0.15 to -0.08): Keskivahvat poikkeamat")
    print(f"  Fair (-0.08 to -0.03): Keskinkertaiset poikkeamat")
    print(f"  Good (-0.03 to 0.02): Lähellä normaalia")
    print(f"  Excellent (> 0.02): Hyvät tiet")
    print()
    
    # Save model
    model_path = Path("MLfiles/anomaly_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save results
    results_path = Path("MLfiles/anomaly_results.xlsx")
    results.to_excel(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    return results, model
