# ML Pipeline - Road Condition Anomaly Detection

## Overview

This production pipeline processes road condition data from Excel, applies a pre-trained ML model, and outputs anomaly predictions with formatted Excel results.

The pipeline is inference-only (no training). It uses a saved model and scaler to detect anomalies in acceleration data.


## 1.0 How to run

### Creating and activating virtual environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/Scripts/activate

# Install dependencies
pip install -r MLmodel/requirements.txt

# Run the pipeline
python ml_pipeline.py "Paallystettyjen_teiden_lahtotiedot_ominaisuus_kuntotiedot_100m_L145695.xlsx"
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r MLmodel/requirements.txt

# Run the pipeline
python ml_pipeline.py "Paallystettyjen_teiden_lahtotiedot_ominaisuus_kuntotiedot_100m_L145695.xlsx"
```

### Using the virtual environment

1. **Activation:** Always activate the environment before running the pipeline
2. **Deactivation:** Type `deactivate` to exit the virtual environment
3. **Dependencies:** All required packages are listed in `requirements.txt`
4. **Recreation:** If you need to recreate the environment, delete the `venv` folder and repeat the setup process

The virtual environment ensures that all dependencies are isolated and consistent across different machines.

Run the program from the `Vaylavirasto-data-projekti` directory:

```bash
python production_pipeline.py
```

The input must be an Excel file name with the extension `.xlsx` or `.xlsm`.

The Excel file must be located in the repository root `Data` directory. Supported input formats are a plain file name or `Data/<file name>`. Full paths and other relative paths are not supported.

## 2.0 Pipeline execution order

The ML pipeline executes the steps in the following order:

1. `step_01_load_data()` - loads and filters the data from Excel worksheets
2. `step_02_clean_data()` - removes invalid rows (missing values, non-numeric values, negative values)
3. `step_03_engineer_features()` - selects final ML features and renames them to English
4. `step_04_load_artifacts()` - load previously trained model and scaler
5. `step_05_run_production` - run predictions
6. `step_06_build_results()` - combine results
7. `step_07_excel_colours()` - format and save to excel

Each step includes comprehensive error handling and data validation. If any step removes all data, the pipeline stops execution with a descriptive error message.



## 3.1 Data Loading
Reads Excel file (.xlsx / .xlsm)
Requires sheet:
Raportti 10m MALLI
Filters required columns
Outputs a DataFrame with selected features

## 3.2 Data Cleaning
Removes:
Missing values
Non-numeric values
Invalid rows
Keeps only valid ML input data

## 3.3 Feature Engineering
Selects acceleration features:
Pysty_kiiht
Sivuheilahdus_kiiht
Nyökkimis_kiiht
Renames to model-ready format

## 3.4 Load Model
```model = joblib.load('anomaly_model.pkl')```
```scaler = joblib.load('scaler.pkl')```
Loads pre-trained Isolation Forest model
Loads fitted scaler

## 3.5 Prediction (Production)
```scaled = scaler.transform(features)```
```predictions = model.predict(scaled)```
```scores = model.decision_function(scaled)```
Applies scaling
Outputs:
predictions (1 = normal, -1 = anomaly)
scores (anomaly strength)

## 3.6 Build Results

Adds:
anomaly_prediction
anomaly_score
anomaly_type (Normal / Anomaly)
anomaly_category:
Category	    Score
Critical	    ≤ -0.15
Poor	        ≤ -0.08
Fair	        ≤ -0.03
Good	        ≤ 0.02
Excellent	    > 0.02
priority_score  (1 = highest priority)


## 3.7 Excel Output

Saves results to:

MLproduction/MLfiles/production_results_coloured.xlsx
Adds:
Color scales (green → red)
Highlighted feature columns
Ratio columns (vs combined acceleration)

## 4.0 Output

Main File
production_results_coloured.xlsx

Contains:
Original metadata
Engineered features
Predictions & scores
Categorization & priority

## 5.0 Error Handling

Each step is wrapped in try/except:

Fails fast with clear messages:
"Data loading failed"
"Data cleaning failed"
"Production failed"
Prevents silent errors in production