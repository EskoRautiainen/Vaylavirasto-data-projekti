# ML Pipeline - Road Condition Anomaly Detection

## Overview

This repository contains a production inference pipeline for road condition anomaly detection.

The system processes raw Excel measurement files, cleans and validates acceleration data, applies a pre-trained anomaly detection model, and exports categorized results into a formatted Excel workbook.

The pipeline is inference-only:

- No model training occurs during execution
- A previously trained model and scaler are loaded from disk
- Predictions and anomaly scores are generated for incoming production data

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

# Run the pipeline
python -m MLproduction.production_pipeline
```

```bash
# Run everything at once
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

# Run the pipeline
python -m MLproduction.production_pipeline
```

```bash
# Run everything at once
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

Step --- Module --- Purpose
1.	Data Loading	        Discover and read Excel source files
2.	Data Cleaning	        Remove invalid ML rows
3.	Feature Selection	    Select model input features
4.	Artifact Loading	    Load trained scaler and model
5.	Production Inference	Generate predictions and anomaly scores
6.	Result Assembly         Combine metadata and predictions
7.	Excel Export	        Save formatted production results


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

### Responsibilities
- Discover .xlsx and .xlsm files
- Ignore temporary Excel lock files (~$)
- Dynamically locate valid worksheets
- Validate required columns
- Normalize column naming

Filter rows where:
pituus == 10

### Required Columns
- pys_kiiht
- siv_kiiht
- nyo_kiiht
- yhd_kiiht
- pituus

Returns a combined pandas DataFrame containing all valid rows across all source files.

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

## Step 3 — Feature Selection
### Purpose
Select model-required acceleration features.

### Selected Features
pys_kiiht
siv_kiiht
nyo_kiiht

### Notes
No feature transformation currently occurs.
This stage performs feature selection only.

## Step 4 — Model Loading
### Purpose
Load previously trained ML artifacts.

### Loaded Files
anomaly_model.pkl
scaler.pkl
Notes

The scaler **must match**:
- feature order
- feature count
- training preprocessing

## Step 5 — Production Inference
### Purpose
Generate anomaly predictions.

### Processing

### Scaling
scaled = scaler.transform(features)

### Prediction
predictions = model.predict(scaled)
scores = model.decision_function(scaled)

### Outputs
Field --- Description
- predictions         1 = normal, -1 = anomaly
- scores	            anomaly confidence score

## Step 6 — Result Assembly
### Purpose
Combine metadata, ML outputs, and prioritization.

### Generated Fields
Column --- Description
- anomaly_prediction	    Raw model output
- anomaly_score	            Rounded anomaly score
- anomaly_type	            Normal / Anomaly
- anomaly_category	        Priority category
- priority_score	        Numeric urgency


**Categorization**
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
Conditional Formatting
Green → Yellow → Red scales
Applied to acceleration ratios
Applied to anomaly indicators

### Ratio Columns
pysty_vs_yhdistetty
sivu_vs_yhdistetty
nyökkimis_vs_yhdistetty

### Highlighted Columns
Selected metadata columns receive blue highlighting for readability.

### Error Handling Strategy
Each pipeline stage:

- validates inputs
- raises descriptive exceptions
- stops execution immediately on failure

This prevents silent corruption of production results.

### Important Development Notes**
Index Integrity
Metadata and ML features rely on shared row indices.

### Avoid:
reset_index(drop=True)
unless index synchronization is explicitly handled.

## Model Compatibility

The scaler and model must always match:
- feature order
- preprocessing logic
- feature count

**Changing feature selection requires retraining artifacts.**


### Future Improvement Recommendations
High Priority
Add schema validation
Add model versioning
Add logging framework
Add unit tests
Add row-level unique identifiers
Medium Priority
Convert prints → structured logging
Add configuration file
Add CLI interface
Add performance metrics
Long-Term
Model registry
Drift detection
Batch processing reports
Automated retraining pipeline
Add stable row ID's to dataframes. Merging different files can cause trouble in the future.