from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from shared.baseline_model import BaselineDistanceModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "MLmodel" / "MLfiles"

# -------------------------
# MODEL TRAINING
# -------------------------
def step_06_model_training(
    good_road_scaled: pd.DataFrame,
) -> BaselineDistanceModel:
    """
    Trains one-class baseline distance model on good-road baseline data.

    Args:
        good_road_scaled: Scaled good road baseline data

    Returns:
        trained_model: Fitted BaselineDistanceModel

    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If input dataframe is empty
    """
    if not isinstance(good_road_scaled, pd.DataFrame):
        raise TypeError("good_road_scaled must be a pandas DataFrame")
    if good_road_scaled.empty:
        raise ValueError("good_road_scaled cannot be empty")

    threshold_quantile = 0.99
    regularization = 1e-6

    features = good_road_scaled.columns.tolist()

    print("------------------------------------------------------------")
    print("Model training started. This may take a while, please wait...")
    print()

    # Use only upward deviation from good baseline center.
    X = good_road_scaled.loc[:, features].to_numpy(dtype=float, copy=False)
    Z = np.clip(X, a_min=0.0, a_max=None)

    mean_vec = Z.mean(axis=0)
    cov = np.cov(Z, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov += np.eye(cov.shape[0]) * regularization
    inv_cov = np.linalg.pinv(cov)

    diff = Z - mean_vec
    d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    threshold = float(np.quantile(d2, threshold_quantile))

    model = BaselineDistanceModel(
        mean_=mean_vec,
        inv_cov_=inv_cov,
        threshold_=threshold,
        feature_names_=features,
        threshold_quantile_=threshold_quantile,
    )

    # Training-only score summary on good-road baseline.
    good_road_scores = model.decision_function(good_road_scaled[features])
    good_road_anomaly_scores = -good_road_scores

    print("Model training completed using baseline distance model")
    print()
    print("Model parameters:")
    print(f"  Threshold quantile (baseline): {threshold_quantile}")
    print(f"  Covariance regularization: {regularization}")
    print(f"  Distance threshold (d2): {threshold:.6f}")
    print("  Features:")
    for feature in features:
        print(f"    {feature}")
    print()

    print("Training summary (good-road baseline only):")
    print("  Dataset used: good_road_scaled")
    print(f"  Training rows (good_road_scaled): {len(good_road_scaled)}")
    print(
        "  Good-road normality score min/median/max: "
        f"{float(np.min(good_road_scores)):.6f} / "
        f"{float(np.median(good_road_scores)):.6f} / "
        f"{float(np.max(good_road_scores)):.6f}"
    )
    print(
        "  Good-road anomaly score min/median/max: "
        f"{float(np.min(good_road_anomaly_scores)):.6f} / "
        f"{float(np.median(good_road_anomaly_scores)):.6f} / "
        f"{float(np.max(good_road_anomaly_scores)):.6f}"
    )
    print()

    model_path = OUTPUT_DIR / "anomaly_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    return model
