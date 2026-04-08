# ML Pipeline for Road Condition Anomaly Detection

## Overview

This pipeline reads road measurement Excel files, builds a "very good road" baseline,
trains an Isolation Forest model, and produces anomaly scores + categories for all rows.

Pipeline steps:
1. Load Excel data and required columns (`pys_kiiht`, `siv_kiiht`, `nyo_kiiht`, `yhd_kiiht`, `pituus`)
2. Keep rows where `pituus == 10`
3. Clean data (missing, non-numeric, negative values)
4. Engineer features (`vertical/lateral/longitudinal_acceleration`)
5. Build dynamic good-road baseline by quantile thresholding
6. Scale baseline and all data with `StandardScaler`
7. Train `IsolationForest` and score all rows

## Run

From `MLmodel` directory:

```bash
python ml_pipeline.py
```

Optional input path (relative to repository root):

```bash
python ml_pipeline.py Data
python ml_pipeline.py Data/some_file.xlsx
```

## Input Handling

- The loader scans all worksheets and tries header rows `0` and `1`.
- Worksheet names are not hardcoded.
- A worksheet is accepted when all required columns are found.

## Baseline Logic

"Good road" baseline is computed dynamically from current run data using quantile thresholds
for these features:

- `vertical_acceleration`
- `lateral_acceleration`
- `longitudinal_acceleration`

The quantile value is an internal constant in `ml_pipeline.py`:

- `BASELINE_QUANTILE = 0.15`

This is intentionally strict to model only very good road conditions.

Used baseline criteria are saved to:

- `MLmodel/MLfiles/baseline_criteria.json`

## Model Training

Current Isolation Forest settings (`src/model_training.py`):

- `contamination='auto'`
- `n_estimators=300`
- `max_samples=0.5`
- `max_features=1.0`
- `bootstrap=False`
- `random_state=42`
- `n_jobs=1`

A holdout split from good-road baseline is used for sanity check:

- `Good-road holdout anomaly rate`

## Categories

Category assignment is consistent with `anomaly_prediction`:

- If prediction is anomaly (`-1`): `Critical`, `Poor`, `Fair`
- If prediction is normal (`1`): `Good`, `Excellent`

## Outputs

All outputs are saved to absolute project path:

- `MLmodel/MLfiles/feature_metadata.json`
- `MLmodel/MLfiles/baseline_criteria.json`
- `MLmodel/MLfiles/scaler.pkl`
- `MLmodel/MLfiles/anomaly_model.pkl`
- `MLmodel/MLfiles/anomaly_results.xlsx`

## Dependencies

Install from:

```bash
pip install -r requirements.txt
```
