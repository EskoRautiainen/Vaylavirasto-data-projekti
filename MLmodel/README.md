# ML Pipeline for Road Condition Anomaly Detection

## Overview

This pipeline reads road measurement data from Excel files, builds a fixed-threshold good-road baseline, trains a one-class baseline distance model, and exports anomaly results to Excel.

Pipeline steps:
1. Load Excel data with required columns (`pys_kiiht`, `siv_kiiht`, `nyo_kiiht`, `yhd_kiiht`, `pituus`).
2. Keep only rows where `pituus == 10`.
3. Clean data (remove missing, non-numeric, and negative values).
4. Select model features (`pys_kiiht`, `siv_kiiht`, `nyo_kiiht`).
5. Build good-road baseline using fixed absolute thresholds.
6. Fit `RobustScaler` on baseline data and reuse it for all rows.
7. Train one-class baseline distance model on scaled baseline.
8. Run inference on all rows and export formatted Excel results.

## Run

Run from repository root (`Vaylavirasto-data-projekti`) in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python MLmodel/ml_pipeline.py
python MLmodel/ml_pipeline.py Data
python MLmodel/ml_pipeline.py Data/some_file.xlsx
```

`input_path` is optional and must be relative to repository root.

Current code behavior:
- If `input_path` is a directory, all Excel files in that directory are scanned.
- If `input_path` is an Excel file, all Excel files in that file's parent directory are scanned.

## Input Requirements

- Input files must be `.xlsx` or `.xlsm`.
- At least one worksheet must contain:
  - `pys_kiiht`
  - `siv_kiiht`
  - `nyo_kiiht`
  - `yhd_kiiht`
  - `pituus`
- Loader scans all worksheets and tries header rows `0` and `1`.
- Temporary lock files (`~$...`) are ignored.
- Only rows with `pituus == 10` are included.
- Absolute paths and `..` path traversal are rejected.

## Good-Road Baseline Logic

Fixed absolute thresholds:
- `pys_kiiht <= 0.08`
- `siv_kiiht <= 2.0`
- `nyo_kiiht <= 4.0`
- `MIN_FEATURES_REQUIRED = 3`

Selection rule:
- A row is kept in baseline only if all `3/3` features meet thresholds.

Baseline metadata:
- `MLmodel/MLfiles/baseline_criteria.json`

## Scaling

`RobustScaler` is fitted on good-road baseline and then used for:
- good-road baseline (training)
- all-road data (inference)

Saved scaler:
- `MLmodel/MLfiles/scaler.pkl`

## Model Training

Model (`MLmodel/src/model_training.py`):
- Squared Mahalanobis distance in scaled feature space
- Upward-only deviation (`clip(x, 0, +inf)`)
- Threshold quantile: `0.99`
- Covariance regularization: `1e-6`

Saved model:
- `MLmodel/MLfiles/anomaly_model.pkl`

## Model Testing / Excel Output

`step_07_model_testing`:
- Runs inference on all scaled rows
- `anomaly_score = -decision_function` (higher = worse)
- Builds `anomaly_score_log = log1p(anomaly_score - min(anomaly_score))`
- Builds rank-based `anomaly_category` from `anomaly_score_log` quantiles (5/20/50/80%)
- Builds `mismatch_flag` for contradictory extremes:
  - high acceleration rank + low anomaly score rank
  - low acceleration rank + high anomaly score rank

Excel output:
- Path: `MLmodel/MLfiles/anomaly_results.xlsx`
- Sheet `Data`:
  - sorted by `anomaly_score` descending
  - color scales on raw accelerations and `anomaly_score`
  - `mismatch_flag` row highlight
  - filter + freeze panes
- Sheet `QA_summary`:
  - correlations (Spearman/Pearson) vs `anomaly_score`
  - category counts and category means
  - top 20 worst rows by `anomaly_score`

## Outputs

All outputs are written to `MLmodel/MLfiles`:
- `feature_metadata.json`
- `baseline_criteria.json`
- `scaler.pkl`
- `anomaly_model.pkl`
- `anomaly_results.xlsx`

## Result Columns (`anomaly_results.xlsx` -> `Data`)

Core columns include:
- scaled feature columns (`pys_kiiht`, `siv_kiiht`, `nyo_kiiht`)
- raw columns:
  - `pys_kiiht_raw`
  - `siv_kiiht_raw`
  - `nyo_kiiht_raw`
- `anomaly_prediction` (`-1` anomaly, `1` normal)
- `anomaly_score` (higher = more anomalous / worse road)
- `anomaly_score_log`
- `anomaly_type` (`Anomaly` / `Normal`)
- `anomaly_category` (`Excellent`, `Good`, `Fair`, `Poor`, `Critical`)
- `mismatch_flag` (`0` / `1`)

## Inference with Trained ML Model

Run training first so `MLmodel/MLfiles` contains:
- `anomaly_model.pkl`
- `scaler.pkl`
- `feature_metadata.json`
- `baseline_criteria.json`

Run inference from repository root after activating the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python
```

Then run the full code block below in the Python session.

### Inference Code (End-to-End)

```python
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import sys

ARTIFACT_DIR = Path("MLmodel/MLfiles")
MLMODEL_PACKAGE_DIR = Path("MLmodel").resolve()
INPUT_EXCEL = Path("Data/new_data.xlsx")

# 1) Load feature contract from training run
with open(ARTIFACT_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
    feature_meta = json.load(f)
features = feature_meta["features"]  # ["pys_kiiht", "siv_kiiht", "nyo_kiiht"]

# 2) Load input and keep exact feature order
df_in = pd.read_excel(INPUT_EXCEL)
X = df_in.loc[:, features].copy()

# 3) Apply same cleaning logic as training (drop NA, numeric conversion, non-negative)
for col in features:
    X[col] = pd.to_numeric(
        X[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
valid_mask = X.notna().all(axis=1) & (X[features] >= 0).all(axis=1)
X_valid = X.loc[valid_mask].copy()

# 4) Load scaler and model
scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
if str(MLMODEL_PACKAGE_DIR) not in sys.path:
    sys.path.append(str(MLMODEL_PACKAGE_DIR))
import src.model_training  # noqa: F401
model = joblib.load(ARTIFACT_DIR / "anomaly_model.pkl")

# 5) Inference
X_scaled = scaler.transform(X_valid)
raw_prediction = model.predict(X_scaled)            # -1 anomaly, 1 normal
normality_score = model.decision_function(X_scaled) # larger = more normal
anomaly_score = -normality_score                    # larger = worse
anomaly_score_log = np.log1p(anomaly_score - anomaly_score.min())

# 6) Relative prioritization categories (same logic as pipeline)
score_rank = pd.Series(anomaly_score_log).rank(method="average", pct=True)
conditions = [
    score_rank <= 0.05,
    score_rank <= 0.20,
    score_rank <= 0.50,
    score_rank <= 0.80,
]
choices = ["Excellent", "Good", "Fair", "Poor"]
anomaly_category = np.select(conditions, choices, default="Critical")

# 7) Optional QA flag for contradictory extremes
accel_sum = X_valid["pys_kiiht"] + X_valid["siv_kiiht"] + X_valid["nyo_kiiht"]
accel_rank = accel_sum.rank(method="average", pct=True)
score_rank_raw = pd.Series(anomaly_score, index=X_valid.index).rank(method="average", pct=True)
mismatch_flag = (
    ((accel_rank >= 0.90) & (score_rank_raw <= 0.10))
    | ((accel_rank <= 0.10) & (score_rank_raw >= 0.90))
).astype(int)

# 8) Build output table for valid rows
result = df_in.loc[valid_mask].copy()
result["anomaly_prediction"] = raw_prediction
result["anomaly_score"] = anomaly_score
result["anomaly_score_log"] = anomaly_score_log
result["anomaly_category"] = anomaly_category
result["mismatch_flag"] = mismatch_flag.values
result["anomaly_type"] = result["anomaly_prediction"].map({1: "Normal", -1: "Anomaly"})
result = result.sort_values("anomaly_score", ascending=False)
```

## How to Handle `anomaly_score`

- `anomaly_score` is the primary severity metric. Higher value means worse road condition.
- Use `anomaly_score` for sorting and top-N inspection.
- Do not use one fixed numeric threshold across different datasets without calibration, because score scale can vary by data distribution.
- Use relative thresholds (percentiles/ranks) for stable operations.

Recommended handling:
1. Sort by `anomaly_score` descending and inspect top rows first.
2. Use rank-based bins for reporting:
   - bottom 5%: `Excellent`
   - 5-20%: `Good`
   - 20-50%: `Fair`
   - 50-80%: `Poor`
   - top 20%: `Critical`
3. Use `anomaly_score_log` for charts and dashboards to compress very large score ranges.
4. Track `mismatch_flag == 1` rows as QA cases for manual review.

## Dependencies

Install dependencies:

```bash
pip install -r MLmodel/requirements.txt
```

## Notes

- This is unsupervised anomaly detection.
- Keep feature names and order exactly the same at inference time as during training.
