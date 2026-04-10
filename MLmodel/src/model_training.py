from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "MLmodel" / "MLfiles"
GOOD_ROAD_NORMAL_SCORE_QUANTILE = 0.05


def step_06_model_training(
    good_road_scaled: pd.DataFrame,
    all_road_scaled: pd.DataFrame,
    all_road_unscaled: pd.DataFrame,
) -> tuple[pd.DataFrame, IsolationForest]:
    """
    Trains Isolation Forest model for anomaly detection.

    This function trains an Isolation Forest model on good road baseline data
    and makes predictions on all road data to identify anomalies.

    Args:
        good_road_scaled: Scaled good road baseline data
        all_road_scaled: Scaled all road data
        all_road_unscaled: Unscaled all road data (same features/order as all_road_scaled)

    Returns:
        Tuple of (results_dataframe, trained_model) where:
        - results_dataframe: DataFrame with anomaly predictions and scores
        - trained_model: Fitted IsolationForest model

    Raises:
        TypeError: If inputs are not pandas DataFrames
        ValueError: If input dataframes are empty
    """
    # Check inputs
    if (
        not isinstance(good_road_scaled, pd.DataFrame)
        or not isinstance(all_road_scaled, pd.DataFrame)
        or not isinstance(all_road_unscaled, pd.DataFrame)
    ):
        raise TypeError("All inputs must be pandas DataFrames")

    if good_road_scaled.empty or all_road_scaled.empty or all_road_unscaled.empty:
        raise ValueError("Input dataframes cannot be empty")
    if len(all_road_scaled) != len(all_road_unscaled):
        raise ValueError(
            "all_road_scaled and all_road_unscaled row counts must match"
        )

    # Model parameters (calibrated defaults)
    contamination = 0.10
    n_estimators = 300
    max_samples = 0.5
    max_features = 1.0
    bootstrap = False
    n_jobs = 1
    random_state = 42

    # Initialize model
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    # Get feature names
    features = good_road_scaled.columns.tolist()

    # Train model on good roads baseline
    model.fit(good_road_scaled[features])

    # Score good-road baseline and all roads.
    # Threshold is derived from the lower tail of good-road score distribution.
    good_road_scores = model.decision_function(good_road_scaled[features])
    scores = model.decision_function(all_road_scaled[features])
    normal_threshold = float(
        pd.Series(good_road_scores).quantile(GOOD_ROAD_NORMAL_SCORE_QUANTILE)
    )
    predictions = np.where(scores < normal_threshold, -1, 1).astype(int)
    raw_model_predictions = model.predict(all_road_scaled[features])

    # Create results dataframe
    results = all_road_scaled.copy()
    results["vertical_acceleration_raw"] = all_road_unscaled["vertical_acceleration"].values
    results["lateral_acceleration_raw"] = all_road_unscaled["lateral_acceleration"].values
    results["longitudinal_acceleration_raw"] = all_road_unscaled[
        "longitudinal_acceleration"
    ].values
    results["anomaly_prediction"] = predictions
    results["model_prediction_raw"] = raw_model_predictions
    results["anomaly_score"] = scores

    # Add anomaly classification
    results["anomaly_type"] = results["anomaly_prediction"].map({1: "Normal", -1: "Anomaly"})

    # Dynamic severity categories based on score distributions from current run.
    categories = pd.Series(index=results.index, dtype="object")
    anomaly_mask = results["anomaly_prediction"] == -1
    normal_mask = ~anomaly_mask

    anomaly_q25 = anomaly_q50 = np.nan
    if anomaly_mask.any():
        anomaly_scores = results.loc[anomaly_mask, "anomaly_score"]
        anomaly_q25 = float(anomaly_scores.quantile(0.25))
        anomaly_q50 = float(anomaly_scores.quantile(0.50))
        categories.loc[anomaly_mask & (results["anomaly_score"] <= anomaly_q25)] = "Critical"
        categories.loc[
            anomaly_mask
            & (results["anomaly_score"] > anomaly_q25)
            & (results["anomaly_score"] <= anomaly_q50)
        ] = "Poor"
        categories.loc[anomaly_mask & (results["anomaly_score"] > anomaly_q50)] = "Fair"

    normal_q75 = np.nan
    if normal_mask.any():
        normal_scores = results.loc[normal_mask, "anomaly_score"]
        normal_q75 = float(normal_scores.quantile(0.75))
        categories.loc[normal_mask & (results["anomaly_score"] < normal_q75)] = "Good"
        categories.loc[normal_mask & (results["anomaly_score"] >= normal_q75)] = "Excellent"

    results["anomaly_category"] = categories.fillna("Good")

    # Add priority scoring
    priority_mapping = {
        "Critical": 1,
        "Poor": 2,
        "Fair": 3,
        "Good": 4,
        "Excellent": 5,
    }
    results["priority_score"] = results["anomaly_category"].map(priority_mapping)

    # Calculate statistics
    anomaly_count = int((predictions == -1).sum())
    normal_count = int((predictions == 1).sum())
    total_count = len(predictions)
    raw_model_mismatch = int((results["anomaly_prediction"] != results["model_prediction_raw"]).sum())

    # Print results
    print()
    print("------------------------------------------------------------")
    print("Model training completed using Isolation Forest")
    print()
    print("Model parameters:")
    print(f"  Contamination: {contamination}")
    print(f"  N_estimators: {n_estimators}")
    print(f"  Max samples: {max_samples}")
    print(f"  Max features: {max_features}")
    print(f"  Bootstrap: {bootstrap}")
    print(f"  N jobs: {n_jobs}")
    print(f"  Random state: {random_state}")
    print(
        "  Good-road normal threshold quantile: "
        f"{GOOD_ROAD_NORMAL_SCORE_QUANTILE}"
    )
    print(f"  Derived anomaly threshold (score): {normal_threshold:.6f}")
    print("  Features:")
    for feature in features:
        print(f"    {feature}")
    print()
    print("Training results:")
    print(f"  Normal roads: {normal_count} ({normal_count / total_count * 100:.1f}%)")
    print(f"  Anomalous roads: {anomaly_count} ({anomaly_count / total_count * 100:.1f}%)")
    print(f"  Total roads: {total_count}")
    print(f"  Actual contamination rate: {anomaly_count / total_count:.4f}")
    print(f"  Difference vs raw model prediction rows: {raw_model_mismatch}")
    print()
    print("Anomaly category distribution:")
    category_counts = results["anomaly_category"].value_counts()
    for category in ["Critical", "Poor", "Fair", "Good", "Excellent"]:
        count = int(category_counts.get(category, 0))
        percentage = (count / total_count) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    print()

    print("Dynamic thresholds (from current run):")
    print(f"  Good/Anomaly score boundary: {normal_threshold:.6f}")
    if anomaly_mask.any():
        print(f"  Critical/Poor anomaly boundary: {anomaly_q25:.6f}")
        print(f"  Poor/Fair anomaly boundary: {anomaly_q50:.6f}")
    if normal_mask.any():
        print(f"  Good/Excellent normal boundary: {normal_q75:.6f}")
    print()

    # Save model
    model_path = OUTPUT_DIR / "anomaly_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save results
    results_path = OUTPUT_DIR / "anomaly_results.xlsx"
    results.to_excel(results_path, index=False)
    print(f"Results saved to: {results_path}")

    return results, model
