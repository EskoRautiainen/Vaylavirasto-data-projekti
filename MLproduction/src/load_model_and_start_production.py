import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
model_path = BASE_DIR / "MLmodel" / "MLfiles" / "anomaly_model.pkl"
scaler_path = BASE_DIR / "MLmodel" / "MLfiles" / "scaler.pkl"

# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------
def step_04_load_artifacts():
    """
    Loads the trained anomaly detection model and preprocessing scaler.

    Returns:
        tuple:
            model: Trained anomaly detection model loaded from disk
            scaler: Fitted scaler used for feature normalization
    """
    model = joblib.load(model_path)                             # Reconstructs trained model in memory
    scaler = joblib.load(scaler_path)                           # Reconstructs scaler in memory
    return model, scaler

# -----------------------------------------------------------------------------
# PRODUCTION
# -----------------------------------------------------------------------------
def step_05_run_production(features_df, model, scaler):
    """
    Runs inference on engineered features using a trained model and scaler.
    Args:
        features_df (pd.DataFrame): Engineered feature set
        model: Trained anomaly detection model
        scaler: Pre-fitted scaler

    Returns:
        predictions (np.ndarray): Model predictions
        scores (np.ndarray): Anomaly scores
    """
    scaled = scaler.transform(features_df)                      # Transform data using training statistics

    predictions = model.predict(scaled)                         # Get anomaly predictions
    scores = model.decision_function(scaled)                    # Get anomaly scores

    return predictions, scores
