# ML Production Pipeline - Road Condition Anomaly Detection

## Overview

This repository contains a production inference pipeline for road condition anomaly detection.

The system processes raw Excel measurement files, cleans and validates acceleration data, applies a pre-trained anomaly detection model, and exports categorized results into a formatted Excel workbook.

The pipeline is inference-only:

- No model training occurs during execution
- A previously trained model and scaler are loaded from disk
- Predictions and anomaly scores are generated for incoming production data

Output Excel-file is saved to **/output**

## How to run?

### Creating and activating virtual environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/Scripts/activate

# Install dependencies
pip install -r MLmodel/requirements.txt

# Run the production pipeline
python -m MLproduction.production_pipeline
```

```bash
python -m venv venv source venv/Scripts/activate pip install -r MLmodel/requirements.txt python -m MLproduction.production_pipeline
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r MLmodel/requirements.txt

# Run the production pipeline
python -m MLproduction.production_pipeline
```

```bash
python3 -m venv venv source venv/bin/activate pip install -r MLmodel/requirements.txt python -m MLproduction.production_pipeline
```

### Using the virtual environment

1. **Activation:** Always activate the environment before running the pipeline
2. **Deactivation:** Type `deactivate` to exit the virtual environment
3. **Dependencies:** All required packages are listed in `requirements.txt`
4. **Recreation:** If you need to recreate the environment, delete the `venv` folder and repeat the setup process

The virtual environment ensures that all dependencies are isolated and consistent across different machines.

Run the program from the `Vaylavirasto-data-projekti` directory:

## Pipeline Architecture

The pipeline consists of seven sequential stages:

```text
Step    Module                  Purpose
1.	    Data Loading	        Discover and read Excel source files
2.	    Data Cleaning	        Remove invalid ML rows
3.	    Feature Selection	    Select model input features
4.	    Artifact Loading	    Load trained scaler and model
5.	    Production Inference	Generate predictions and anomaly scores
6.	    Result Assembly         Combine metadata and predictions
7.	    Excel Export	        Save formatted production results
```

## Directory Structure
```text
MLproduction/
├── production_pipeline.py
├── src/
│   ├── data_loading.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── load_model_and_start_production.py
│   ├── build_results.py
│   └── build_excel_table.py
│
MLmodel/
├── MLfiles/
│   ├── anomaly_model.pkl
│   ├── scaler.pkl
│   └── feature_metadata.json
|
|── output
│   └── production_results_coloured.xlsx
```

## Step 1 — Data Loading
### Purpose
Reads Excel measurement files and extracts valid source data.

### Parameters
- User does not give additional parameters → use /SourceData folder.
- User gives folder in parameters: → use it.​
- User gives a file name in parameters:​ → use its parent-folder.

### Example
```bash
python -m MLproduction.production_pipeline
python -m MLproduction.production_pipeline Data2027
python -m MLproduction.production_pipeline tiet-2-3-9.xlsx
```

### Responsibilities
- Discover .xlsx and .xlsm files
- Ignore temporary Excel lock files (~$)
- Dynamically locate valid worksheets
- Validate required columns
- Normalize column naming

Filter rows where:
**pituus** is == 10

### Required Columns
- pys_kiiht
- siv_kiiht
- nyo_kiiht
- yhd_kiiht
- pituus

Returns a combined pandas DataFrame containing all valid rows across all source files.

### Required Columns - Details
```text
Column	        Type	        Unit	    Description
pys_kiiht	    float	        m/s²	    Vertical acceleration (or equivalent sensor unit)
siv_kiiht	    float	        m/s²	    Lateral acceleration
nyo_kiiht	    float	        m/s²	    Longitudinal / pitch acceleration
yhd_kiiht	    float	        m/s²	    Combined acceleration metric
pituus	        int / float	    meters	    Segment length filter (must equal 10)

### Console feedback
Log in console how many rows where dropped from each loaded file.

Show in console 5 rows of data.

## Step 2 — Data Cleaning
### Purpose
Ensure ML input data is numerically valid.

### Cleaning Stages
Phase 1 — Missing Values
Removes rows containing NaN values.

Phase 2 — Non-Numeric Values
Converts comma decimals:
1,23 → 1.23
Removes rows containing invalid numeric values.

Phase 3 — Negative Values
Removes rows containing negative acceleration values.

### Output
A cleaned DataFrame suitable for ML inference.

### Console feedback
Log in console how many rows were removed during each phase.


## Step 3 — Feature Engineering
### Purpose
- Prepare features for machine learning.
- Save relevant info to MLmodel/MLfiles/feature.metadata.json.
- feature_metadata.json may help with troubleshooting.

### Selected Features
- pys_kiiht
- siv_kiiht
- nyo_kiiht

### Notes
- This stage performs feature selection only.
- Features are not transformed.

### Console feedback
Log in console 5 example rows of data

## Step 4 — Model Loading
### Purpose
Load previously trained ML artifacts.

### Loaded Files
- anomaly_model.pkl
- scaler.pkl

### anomaly_model.pkl
- Model is taught on "good" road data.
- Contains anomaly scoring logic.
- Without it, no predictions can be made.

### scaler.pkl
- Transforms data into the same scale used during training.
- Without it, model assumption breaks and results become unstable.

### Notes
The scaler **must match**:
- feature order
- feature count
- training preprocessing


## Step 5 — Production Inference
### Purpose
Applies the trained preprocessing pipeline and model inference to generate anomaly predictions and scores from engineered features

### Scaling
scaled = scaler.transform(features)

### Prediction
- predictions = model.predict(scaled)
- scores = model.decision_function(scaled)

### Outputs
```text
Field                   Description
predictions             1 = normal, -1 = anomaly
scores                  anomaly confidence score
```

### Console feedback
Log in console the amount of rows in the following variables:
- metadata_cleaned
- engineered
- scored

Allows user to confirm that no data was lost during pipeline steps.

## Step 6 — Result Assembly
### Purpose
Combine metadata, ML outputs, and prioritization.

### Generated Fields
```text
Column                      Description
anomaly_prediction	        Raw model output
anomaly_score	            Rounded anomaly score
anomaly_type	            Normal / Anomaly
anomaly_category	        Priority category
priority_score	            Numeric urgency
```

### Categorization
Current implementation uses percentile-based ranking.

Percentile --- Category
- 0–4%	Critical
- 4–8%	Poor
- 8–40%	Fair
- 40–80%	Good
- 80–100%	Excellent

## Step 7 — Excel Export
### Purpose
Export production-ready results workbook.

### Features
Green → Yellow → Red scales

Colour-coded formatting is applied to: 
- Ride-values
- Ride-ratios

Blue highlighting is applied to:
- ura_max
- harjanne_ka
- kaltevuus
- rms_mega_oik
- delta
- tl332_paapak
- **anomaly_score**

### Ratio Columns
- pysty_vs_yhdistetty
- sivu_vs_yhdistetty
- nyökkimis_vs_yhdistetty
Creates a visual aid, where user can glance, which type of ride-variable is increasing the total yhd_ride.


## Error Handling Strategy
Each pipeline stage:

- validates inputs
- raises descriptive exceptions
- stops execution immediately on failure

This prevents silent corruption of production results.

### Important Development Notes
Metadata and ML features rely on shared row indices.

### Avoid:
- reset_index(drop=True)
- reset_index may be used if index synchronization is explicitly handled or stable row ID's are added.

## Model Compatibility

The scaler and model must always match:
- feature order
- preprocessing logic
- feature count

Changing feature selection requires retraining artifacts.


### Future Improvement Recommendations
Add stable row ID's to dataframes. Merging different files can cause trouble in the future.

Change data loading logic so that file name parameter loads only the chosen file. Currently it loads up the parent folder.
