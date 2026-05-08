# ML Pipeline for Road Condition Anomaly Detection

This folder contains the end-to-end ML pipeline for detecting road-condition anomalies from Excel measurement files.

## Quick Start

Run from repository root (`Vaylavirasto-data-projekti`):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r MLmodel/requirements.txt

# Use default input location
python MLmodel/ml_pipeline.py

# Scan all Excel files under a directory
python MLmodel/ml_pipeline.py SourceData

# If a file path is given (or path does not resolve to an existing directory),
# pipeline scans all Excel files in that path's parent directory
python MLmodel/ml_pipeline.py SourceData/some_file.xlsx
```

`input_path` is optional, must be relative to repository root, and must not contain absolute paths or `..`.

## What the Pipeline Does

1. Loads Excel sheets with required columns:
   - `pys_kiiht`, `siv_kiiht`, `nyo_kiiht`, `yhd_kiiht`, `pituus`
2. Keeps only rows where `pituus == 10`.
3. Cleans model features (`pys_kiiht`, `siv_kiiht`, `nyo_kiiht`):
   - converts to numeric (supports decimal commas)
   - drops missing/non-numeric values
   - drops negative values
4. Builds a fixed-threshold "good road" baseline.
5. Fits `RobustScaler` on baseline rows.
6. Trains one-class baseline-distance anomaly model.
7. Scores all valid rows and exports formatted Excel results.

## Input Requirements

- File types: `.xlsx` or `.xlsm`
- Loader behavior:
  - scans all worksheets
  - tries header rows `0` and `1`
  - ignores temporary lock files (`~$...`)
- At least one worksheet must include all required columns.

## Baseline Criteria

Row is accepted into baseline only if all 3 model features pass:

- `pys_kiiht <= 0.08`
- `siv_kiiht <= 2.0`
- `nyo_kiiht <= 4.0`

Criteria metadata is written to:
- `MLmodel/MLfiles/baseline_criteria.json`

## Model and Scoring

Implementation: `MLmodel/src/model_training.py`

- Distance: squared Mahalanobis in scaled feature space
- Deviation handling: upward-only (`clip(x, 0, +inf)`)
- Threshold quantile: `0.99`
- Covariance regularization: `1e-6`

Scoring outputs:
- `anomaly_prediction`: `-1` anomaly, `1` normal
- `anomaly_score`: `-decision_function` (higher = worse)
- `anomaly_score_log`: `log1p(anomaly_score - min(anomaly_score))`

## Output Files

All artifacts are written to `MLmodel/MLfiles`:

- `feature_metadata.json`
- `baseline_criteria.json`
- `scaler.pkl`
- `anomaly_model.pkl`
- `anomaly_results.xlsx`

## Excel Output

`anomaly_results.xlsx` contains:

- `Data` sheet:
  - rows sorted by `anomaly_score` descending
  - conditional formatting for raw accelerations and anomaly score
  - `mismatch_flag` highlight
  - filters and freeze panes
- `QA_summary` sheet:
  - Spearman and Pearson correlations vs `anomaly_score`
  - category counts and means
  - top-20 worst rows

Main columns in `Data` include:
- `pys_kiiht`, `siv_kiiht`, `nyo_kiiht` (scaled features)
- `pys_kiiht_raw`, `siv_kiiht_raw`, `nyo_kiiht_raw`
- `anomaly_prediction`, `anomaly_type`
- `anomaly_score`, `anomaly_score_log`
- `anomaly_category` (`Excellent`, `Good`, `Fair`, `Poor`, `Critical`)
- `mismatch_flag`

## How to Use `anomaly_score`

- Use `anomaly_score` as the primary severity metric.
- Higher score means more anomalous (worse road condition).
- Prefer ranking/percentiles over a single global numeric threshold.

Recommended:
1. Sort descending by `anomaly_score`.
2. Prioritize top-N rows for inspection.
3. Use `anomaly_score_log` in visualizations to compress wide score ranges.
4. Review rows where `mismatch_flag == 1` as QA candidates.

## Inference with Saved Artifacts

Train once first so these exist in `MLmodel/MLfiles`:
- `anomaly_model.pkl`
- `scaler.pkl`
- `feature_metadata.json`
- `baseline_criteria.json`

Then run inference in a Python session from repository root:

```python
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import sys

ARTIFACT_DIR = Path("MLmodel/MLfiles")
MLMODEL_PACKAGE_DIR = Path("MLmodel").resolve()
INPUT_EXCEL = Path("SourceData/new_data.xlsx")

# 1) Load feature contract from training
with open(ARTIFACT_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
    feature_meta = json.load(f)
features = feature_meta["features"]  # ["pys_kiiht", "siv_kiiht", "nyo_kiiht"]

# 2) Load and align features
df_in = pd.read_excel(INPUT_EXCEL)
X = df_in.loc[:, features].copy()

# 3) Apply training-compatible cleaning
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

# 5) Score
X_scaled = scaler.transform(X_valid)
raw_prediction = model.predict(X_scaled)            # -1 anomaly, 1 normal
normality_score = model.decision_function(X_scaled) # larger = more normal
anomaly_score = -normality_score                    # larger = worse
anomaly_score_log = np.log1p(anomaly_score - anomaly_score.min())

# 6) Rank-based categories
score_rank = pd.Series(anomaly_score_log).rank(method="average", pct=True)
conditions = [
    score_rank <= 0.05,
    score_rank <= 0.20,
    score_rank <= 0.50,
    score_rank <= 0.80,
]
choices = ["Excellent", "Good", "Fair", "Poor"]
anomaly_category = np.select(conditions, choices, default="Critical")

# 7) Optional QA mismatch flag
accel_sum = X_valid["pys_kiiht"] + X_valid["siv_kiiht"] + X_valid["nyo_kiiht"]
accel_rank = accel_sum.rank(method="average", pct=True)
score_rank_raw = pd.Series(anomaly_score, index=X_valid.index).rank(method="average", pct=True)
mismatch_flag = (
    ((accel_rank >= 0.90) & (score_rank_raw <= 0.10))
    | ((accel_rank <= 0.10) & (score_rank_raw >= 0.90))
).astype(int)

# 8) Build result
result = df_in.loc[valid_mask].copy()
result["anomaly_prediction"] = raw_prediction
result["anomaly_score"] = anomaly_score
result["anomaly_score_log"] = anomaly_score_log
result["anomaly_category"] = anomaly_category
result["mismatch_flag"] = mismatch_flag.values
result["anomaly_type"] = result["anomaly_prediction"].map({1: "Normal", -1: "Anomaly"})
result = result.sort_values("anomaly_score", ascending=False)
```

## Notes

- Method type: unsupervised anomaly detection.
- In inference, feature names and order must exactly match training metadata.
