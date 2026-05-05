from pathlib import Path
from MLproduction.src.data_loading                      import step_01_load_data
from MLproduction.src.data_cleaning                     import step_02_clean_data
from MLproduction.src.feature_engineering               import step_03_engineer_features
from MLproduction.src.load_model_and_start_production   import step_04_load_artifacts
from MLproduction.src.load_model_and_start_production   import step_05_run_production
from MLproduction.src.build_results                     import step_06_build_results
from MLproduction.src.build_excel_table                 import step_07_excel_colours

# ----------------------------------------------------------------------------------------------------
#   DEFINE CONSTANTS
# ----------------------------------------------------------------------------------------------------
# FILE PATH
REPO_ROOT = Path(__file__).resolve().parent.parent
file_path = REPO_ROOT / "Data"
output_path = REPO_ROOT / "MLproduction" / "MLfiles" / "production_results_coloured.xlsx"

# SELECTED FEATURES
selected_features = [
    'karttapvm',
    'tie',
    'kaista',
    'ajorata',
    'aosa',
    'aet',
    'losa',
    'let',
    'pituus',
    'mittausaika',
    'kevat_kesa',
    'ura_max',
    'harjanne_ka',
    'kaltevuus',
    'rms_mega_oik',
    'delta',
    'yhd_kiiht',    # Metadata = df[selected_features] contains yhd_kiiht,
                    # while df = step_01_load_data(file_path) contains yhd_kiiht, pys_kiiht, siv_kiiht, nyo_kiiht.
    'tl332_paapak'
    ]


# ----------------------------------------------------------------------------------------------------
#   MAIN PIPELINE
# ----------------------------------------------------------------------------------------------------
def run_pipeline():
    """
    Executes the full ML production pipeline:
    1. Load raw data
    2. Clean ML features
    3. Engineer features
    4. Load trained model and scaler
    5. Generate predictions and scores
    6. Build results dataset
    7. Export formatted Excel file
    """

# LOAD DATA
    try:
        df = step_01_load_data(file_path)
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {e}") from e
    metadata = df[selected_features].copy() # Store only selected features

# CLEAN DATA
    ml_features = df[['pys_kiiht', 'siv_kiiht', 'nyo_kiiht']].copy() # Load 3 columns for ML
    try:
        cleaned_features = step_02_clean_data(ml_features) # Clean bad rows
    except Exception as e:
        raise RuntimeError(f"Data cleaning failed: {e}") from e
    metadata_cleaned = metadata.loc[cleaned_features.index] # Merge with metadata - No bad rows

# FEATURE ENGINEERING
# - Prepare data for machine learning
# - Dataframe input validation
# - Feature selection
# - Feature renaming
    try:
        engineered_features = step_03_engineer_features(cleaned_features)
    except Exception as e:
        raise RuntimeError(f"Feature engineering failed: {e}") from e

# USE PREVIOUSLY TRAINED MODEL
    try:
        model, scaler = step_04_load_artifacts() # Load scaler and model
    except Exception as e:
        raise RuntimeError(f"Model and scaler loading failed: {e}") from e

    try:
        predictions, scores = step_05_run_production(engineered_features, model, scaler) # Run production and get predictions
    except Exception as e:
        raise RuntimeError(f"Production (prediction) failed: {e}") from e 

    print("metadata_cleaned:", len(metadata_cleaned))
    print("engineered:", len(engineered_features))
    print("predictions:", len(predictions))
    print("scores:", len(scores))

# BUILD RESULTS
    try:
        results = step_06_build_results(metadata_cleaned, engineered_features, predictions, scores) # Build and categorize results
    except Exception as e:
        raise RuntimeError(f"Building results failed: {e}") from e

# CUSTOMISE EXCEL AND SAVE RESULTS
    try:
        step_07_excel_colours(results, output_path) # Save colour-coded results to Excel
    except Exception as e:
        raise RuntimeError(f"Excel formatting and saving failed: {e}") from e
    print("Production complete.")


# ----------------------------------------------------------------------------------------------------
#   ENTRY POINT
# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()