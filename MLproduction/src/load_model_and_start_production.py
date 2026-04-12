import joblib
import pandas as pd
import xlsxwriter

# -------------------------
# LOAD MODEL
# -------------------------
def step_04_load_artifacts():
    model = joblib.load('./MLmodel/MLfiles/anomaly_model.pkl')
    scaler = joblib.load('./MLmodel/MLfiles/scaler.pkl')
    return model, scaler

# -------------------------
# PRODUCTION
# -------------------------
def step_05_run_production(features_df, model, scaler):
    scaled = scaler.transform(features_df)

    predictions = model.predict(scaled)
    scores = model.decision_function(scaled)

    return predictions, scores

