import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLproduction.src.data_loading                      import step_01_load_data
from MLmodel.src.data_cleaning                          import step_02_clean_data
from MLmodel.src.feature_engineering                    import step_03_engineer_features
from MLproduction.src.load_model_and_start_production   import step_04_load_artifacts
from MLproduction.src.load_model_and_start_production   import step_05_run_production
from MLproduction.src.build_results                     import step_06_build_results
from MLproduction.src.build_excel_table                 import step_07_excel_colours


# DEFINE FILEPATH
file_path = './Data/Paallystettyjen_teiden_lahtotiedot_ominaisuus_kuntotiedot_100m_L145695.xlsx'


# DEFINE SELECTED FEATURES
selected_features = [
    'Karttapvm',
    'Elinvoimakeskus',
    'Ajorata',
    'Kaista',
    'Aosa',
    'Aet',
    'Losa',
    'Let',
    'Pituus',
    'Ura_max',
    'Harjanne',
    'Kaltevuus',
    'Rms_mega_oik',
    'Delta',
    "Yhdistetty_kiiht_rms",
    'Päällyste_pak'
    ]




# -------------------------
# MAIN PIPELINE
# -------------------------
def run_pipeline(file_path):


# LOAD DATA
    try:
        df = step_01_load_data(file_path)
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {e}") from e
    metadata = df[selected_features].copy() # Load 18 columns

# CLEAN DATA
    ml_features = df[['Pysty_kiiht', 'Sivuheilahdus_kiiht', 'Nyökkimis_kiiht']].copy() # Load 3 columns for ML
    try:
        cleaned_features = step_02_clean_data(ml_features)
    except Exception as e:
        raise RuntimeError(f"Data cleaning failed: {e}") from e
    metadata_cleaned = metadata.loc[cleaned_features.index]

# FEATURE ENGINEERING
    
    try:
        engineered_features = step_03_engineer_features(cleaned_features)
    except Exception as e:
        raise RuntimeError(f"Feature engineering failed: {e}") from e

# USE PREVIOUSLY TRAINED MODEL

    try:
        model, scaler = step_04_load_artifacts()
    except Exception as e:
        raise RuntimeError(f"Model and scaler loading failed: {e}") from e

    try:
        predictions, scores = step_05_run_production(engineered_features, model, scaler)
    except Exception as e:
        raise RuntimeError(f"Production (prediction) failed: {e}") from e

# BUILD RESULTS

    try:
        results = step_06_build_results(metadata_cleaned, engineered_features, predictions, scores)
    except Exception as e:
        raise RuntimeError(f"Building results failed: {e}") from e


# CUSTOMISE EXCEL AND SAVE RESULTS

    try:
        step_07_excel_colours(results, "./MLproduction/MLfiles/production_results_coloured.xlsx")
    except Exception as e:
        raise RuntimeError(f"Excel formatting and saving failed: {e}") from e
    print("Production complete.")


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    import sys
    run_pipeline(file_path)