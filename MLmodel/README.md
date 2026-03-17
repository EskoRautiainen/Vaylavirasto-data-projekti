# ML Pipeline for Road Condition Anomaly Detection

## Overview

This ML pipeline analyzes road condition data using acceleration measurements to detect anomalies and categorize road conditions. The pipeline uses Isolation Forest algorithm with a goal-oriented approach to identify roads that don't meet quality standards.

## 1.0 How to run

### Creating and activating virtual environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

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
pip install -r requirements.txt

# Run the pipeline
python ml_pipeline.py "Paallystettyjen_teiden_lahtotiedot_ominaisuus_kuntotiedot_100m_L145695.xlsx"
```

### Using the virtual environment

1. **Activation:** Always activate the environment before running the pipeline
2. **Deactivation:** Type `deactivate` to exit the virtual environment
3. **Dependencies:** All required packages are listed in `requirements.txt`
4. **Recreation:** If you need to recreate the environment, delete the `venv` folder and repeat the setup process

The virtual environment ensures that all dependencies are isolated and consistent across different machines.

Run the program from the `MLmodel` directory:

```bash
python ml_pipeline.py "Paallystettyjen_teiden_lahtotiedot_ominaisuus_kuntotiedot_100m_L145695.xlsx"
```

```bash
python ml_pipeline.py "Data/Paallystettyjen_teiden_lahtotiedot_ominaisuus_kuntotiedot_100m_L145695.xlsx"
```

The input must be an Excel file name with the extension `.xlsx` or `.xlsm`.

The Excel file must be located in the repository root `Data` directory. Supported input formats are a plain file name or `Data/<file name>`. Full paths and other relative paths are not supported.

## 2.0 Pipeline execution order

The ML pipeline executes the steps in the following order:

1. `step_01_load_data()` - loads and filters the data from Excel worksheets
2. `step_02_clean_data()` - removes invalid rows (missing values, non-numeric values, negative values)
3. `step_03_engineer_features()` - selects final ML features and renames them to English
4. `step_04_filter_good_road()` - filters to good road conditions using 25th percentile thresholds
5. `step_05_data_scaling()` - scales features using StandardScaler fitted on good road baseline
6. `step_06_model_training()` - trains Isolation Forest and detects anomalies with categorization

Each step includes comprehensive error handling and data validation. If any step removes all data, the pipeline stops execution with a descriptive error message.

## 2.1 step_01_load_data()

The `step_01_load_data()` function loads the configured Excel worksheets using worksheet-specific header row settings. If any required worksheet is missing or does not contain data rows, the function raises an error.

It first prints the first data row in the format `Worksheet | Column: value` so the loaded worksheet content can be validated before further processing.

After the validation output, the function validates the `Raportti 10m MALLI` worksheet and filters its data to the selected ML feature columns:

- `Pysty_kiiht` (vertical acceleration)
- `Sivuheilahdus_kiiht` (lateral acceleration)
- `Nyökkimis_kiiht` (longitudinal acceleration)

Both required worksheets must exist in the Excel file:

- `Raportti 100m` - read for user validation and future use
- `Raportti 10m MALLI` - used as the primary data source for ML features

The `Raportti 100m` worksheet is read to validate that both worksheets are accessible and to display the first data row to the user. Currently, this data is not used in the ML processing but may be utilized in future development.

The `Raportti 10m MALLI` worksheet must contain all three selected ML feature columns. If any of these columns are missing, the function raises an error.

The original larger worksheet data is not kept after filtering. Only the filtered dataset from the `Raportti 10m MALLI` worksheet is retained in memory and returned from the function.

Finally, the function prints a five-row table preview of the filtered dataset.

## 2.2 step_02_clean_data()

The `step_02_clean_data()` function receives the filtered `DataFrame` returned by `step_01_load_data()` and removes invalid rows in three distinct phases.

### Phase 1: Missing values
The function first identifies and removes rows that contain at least one missing value (`NaN`). This ensures that all subsequent processing works with complete data.

### Phase 2: Non-numeric values
The function then converts all values to numeric format, supporting both comma (`,`) and period (`.`) as decimal separators. Rows containing values that cannot be converted to numbers (such as text strings or invalid formats) are removed.

**Supported numeric formats:**
- `"1,23"` → `1.23` (Finnish format)
- `"1.23"` → `1.23` (English format)
- `"123"` → `123.0` (integers)

### Phase 3: Negative values
Finally, the function identifies and removes any rows that contain negative values, as acceleration measurements should not be negative in this context.

The function prints detailed statistics for each phase, including:
- the index values of removed rows for each specific phase
- the number of rows remaining after all cleaning phases

Finally, the function returns the cleaned `DataFrame` with its index reset.

## 2.3 step_03_engineer_features()

The `step_03_engineer_features()` function receives the cleaned `DataFrame` returned by `step_02_clean_data()` and performs feature selection and renaming to prepare the final feature set for the ML model.

### Feature Selection and Renaming
The function selects and renames the three primary acceleration measurement features:

| Original Name | English Name | Description |
|--------------|--------------|-------------|
| `Pysty_kiiht` | `vertical_acceleration` | Vertical bouncing/up-down movement |
| `Sivuheilahdus_kiiht` | `lateral_acceleration` | Side-to-side swaying movement |
| `Nyökkimis_kiiht` | `longitudinal_acceleration` | Forward-backward nodding movement |

### Feature Metadata
The function saves feature metadata to `MLfiles/feature_metadata.json` including:
- Pipeline version
- Feature names and descriptions
- Data row count
- Processing date
- Feature count

### Feature Engineering Foundation
This function serves as the foundation for future feature engineering capabilities. While the current implementation focuses on feature selection and renaming, this is the designated location where new feature creation will be implemented in future development.

The function prints a five-row table preview of the engineered feature set.

Finally, the function returns the engineered `DataFrame` with the selected features, ready for subsequent ML processing steps.

## 2.4 step_04_filter_good_road()

The `step_04_filter_good_road()` function receives the engineered `DataFrame` returned by `step_03_engineer_features()` and filters it to retain only road segments that represent good road surface conditions.

This function uses speed-agnostic thresholds based on 25th percentile values from all speed data, providing a comprehensive baseline for anomaly detection across different speed limits.

The function applies the following filtering criteria based on the 25th percentile values of all speed data:

- `vertical_acceleration ≤ 0.05`
- `lateral_acceleration ≤ 1.0`
- `longitudinal_acceleration ≤ 3.0`

### Why these values?

Think of this like setting quality standards for road smoothness. We analyzed thousands of road measurements from all speed limits to find what makes a "perfectly smooth" road.

#### The simple explanation:

**We studied all speed data** (40 km/h, 60 km/h, 80 km/h, 100 km/h, etc.) to create a comprehensive baseline that works for any road. This speed-agnostic approach provides more data for better ML training.

#### How we found the perfect road standards:

1. **Collected data**: We gathered 1576 road measurements from all speed limits
2. **Found the best 18.8%**: We looked for the smoothest 25th percentile of all road measurements
3. **Set the standards**: These roads became our "perfect road" baseline

#### What the numbers mean in plain language:

| Measurement | What it measures | Perfect road limit | Everyday meaning |
|-------------|------------------|-------------------|------------------|
| `vertical_acceleration ≤ 0.05` | Vertical bouncing | ≤ 0.05 | Car doesn't bounce up and down |
| `lateral_acceleration ≤ 1.0` | Side-to-side swaying | ≤ 1.0 | Car doesn't sway left and right |
| `longitudinal_acceleration ≤ 3.0` | Forward-backward nodding | ≤ 3.0 | Car doesn't nod forward/backward |

#### Why this matters for finding problems:

**Perfect baseline makes problems obvious:**
- When we know what a "perfect road" feels like (these limits)
- Any road that's worse than perfect is easy to spot
- It's like having a ruler - you can measure exactly how bad a problem is

#### The results in simple numbers:
- **18.8% of all roads** meet these perfect standards (297 rows)
- **81.2% of roads** have some level of problems
- This gives us a very clear "perfect vs. imperfect" comparison

#### Real-world benefit:
When the car drives on a road that's worse than these perfect standards, our system immediately knows: *"This road has problems"* - because we have scientifically proven what "perfect" actually means across all speed limits.

The function prints filtering statistics including the number of rows before and after filtering and displays a five-row preview of the filtered good road dataset.

Finally, the function returns the filtered `DataFrame` containing only good road segments.

## 2.5 step_05_data_scaling()

The `step_05_data_scaling()` function receives the good road filtered `DataFrame` returned by `step_04_filter_good_road()` and applies feature scaling using StandardScaler.

### Scaling Process

1. **Baseline Fitting**: The StandardScaler is fitted only on the good road baseline data (18.8% of roads that meet quality standards)
2. **Data Transformation**: The fitted scaler transforms all road data to ensure consistent scaling
3. **Model Persistence**: The fitted scaler is saved to `MLfiles/scaler.pkl` for future use

### Why Scale on Good Roads Only?

The scaling is performed using only the good road baseline because:
- **Goal-oriented approach**: We want to measure how far roads deviate from the quality standard
- **Consistent baseline**: All measurements are relative to the "perfect road" standard
- **Better anomaly detection**: Deviations from the good baseline are more meaningful

### Scaling Statistics

The function prints baseline statistics including:
- Feature means (should be ~0.0 after scaling)
- Feature standard deviations (should be ~1.0 after scaling)
- Number of scaled rows

The function returns:
- `good_road_scaled`: Scaled good road baseline data
- `scaler`: Fitted StandardScaler object

## 2.6 step_06_model_training()

The `step_06_model_training()` function receives the scaled good road baseline and all road data, then trains an Isolation Forest model for anomaly detection and categorization.

### Model Configuration

The function uses Isolation Forest with the following parameters:
- **Contamination**: 'auto' - lets the algorithm determine the optimal contamination rate based on data
- **N_estimators**: 200 - number of trees in the forest
- **Random state**: 42 - ensures reproducible results

### Training Process

1. **Model Training**: The Isolation Forest is trained on the good road baseline data
2. **Prediction**: The trained model makes predictions on all road data
3. **Scoring**: Anomaly scores are calculated for all road segments
4. **Categorization**: Fixed-threshold categorization is applied to anomaly scores

### Anomaly Categories

The function applies fixed threshold categorization based on anomaly scores:

| Category | Score Range | Description | Priority |
|----------|-------------|-------------|----------|
| **Critical** | ≤ -0.15 | Strong anomalies - immediate repair needed | 1 |
| **Poor** | -0.15 to -0.08 | Moderate anomalies - planned repair | 2 |
| **Fair** | -0.08 to -0.03 | Mild anomalies - monitoring | 3 |
| **Good** | -0.03 to 0.02 | Near normal - no action needed | 4 |
| **Excellent** | > 0.02 | Better than normal - examples for others | 5 |

### Output Files

The function saves two output files:
1. **Model**: `MLfiles/anomaly_model.pkl` - trained Isolation Forest model
2. **Results**: `MLfiles/anomaly_results.xlsx` - complete results with all data and predictions

### Results DataFrame

The results DataFrame contains 8 columns:
1. `vertical_acceleration` - Scaled vertical acceleration
2. `lateral_acceleration` - Scaled lateral acceleration  
3. `longitudinal_acceleration` - Scaled longitudinal acceleration
4. `anomaly_prediction` - Binary prediction (1=Normal, -1=Anomaly)
5. `anomaly_score` - Anomaly score (-0.254 to 0.0457)
6. `anomaly_type` - Human-readable type ('Normal'/'Anomaly')
7. `anomaly_category` - 5-tier categorization (Critical/ Poor/ Fair/ Good/ Excellent)
8. `priority_score` - Priority for maintenance (1=highest, 5=lowest)

### Statistics Output

The function prints comprehensive statistics:
- Model parameters and configuration
- Training results (normal vs anomalous roads)
- Category distribution with percentages
- Fixed threshold explanations

### Example Output

```
Anomaly category distribution:
  Critical: 11 (0.7%)
  Poor: 17 (1.1%)
  Fair: 1263 (80.1%)
  Good: 56 (3.6%)
  Excellent: 229 (14.5%)
```

## 3.0 Output Files

The pipeline generates the following output files in the `MLfiles/` directory:

### 3.1 anomaly_model.pkl
- **Content**: Trained Isolation Forest model
- **Purpose**: Can be loaded for future predictions without retraining
- **Format**: Pickle file

### 3.2 anomaly_results.xlsx
- **Content**: Complete results with 1576 rows and 8 columns
- **Purpose**: Primary output for analysis and decision-making
- **Format**: Excel file with all predictions and categorizations

### 3.3 feature_metadata.json
- **Content**: Pipeline metadata and feature information
- **Purpose**: Documentation and version tracking
- **Format**: JSON file

### 3.4 scaler.pkl
- **Content**: Fitted StandardScaler object
- **Purpose**: Ensures consistent scaling for future predictions
- **Format**: Pickle file

All output files are excluded from version control via `.gitignore`.

## 5.0 Model Inference

### Using the Trained Model for New Data

The saved model and scaler can be used to process new road data without retraining:

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Load saved artifacts
model = joblib.load('MLfiles/anomaly_model.pkl')
scaler = joblib.load('MLfiles/scaler.pkl')

# Prepare new data (must have same feature structure)
new_data = pd.DataFrame({
    'vertical_acceleration': [...],
    'lateral_acceleration': [...],
    'longitudinal_acceleration': [...]
})

# Scale the data using the fitted scaler
scaled_data = scaler.transform(new_data)

# Make predictions
predictions = model.predict(scaled_data)
scores = model.decision_function(scaled_data)

# Add categorization
def categorize_by_fixed_thresholds(scores):
    def categorize(score):
        if score <= -0.15:
            return 'Critical'
        elif score <= -0.08:
            return 'Poor'
        elif score <= -0.03:
            return 'Fair'
        elif score <= 0.02:
            return 'Good'
        else:
            return 'Excellent'
    return [categorize(s) for s in scores]

# Create results DataFrame
results = new_data.copy()
results['anomaly_prediction'] = predictions
results['anomaly_score'] = scores
results['anomaly_type'] = ['Normal' if p == 1 else 'Anomaly' for p in predictions]
results['anomaly_category'] = categorize_by_fixed_thresholds(scores)
results['priority_score'] = results['anomaly_category'].map({
    'Critical': 1, 'Poor': 2, 'Fair': 3, 'Good': 4, 'Excellent': 5
})
```

### Important Notes for Inference

1. **Feature Order**: New data must have features in the same order as training data
2. **Data Quality**: Ensure new data goes through the same cleaning process as training data
3. **Scaling**: Always use the saved scaler - never fit a new one on inference data
4. **Model Version**: Track model versions to ensure compatibility with inference code

## 6.0 Error handling and data validation

The pipeline includes comprehensive error handling and data validation at each step:

### Input validation
Each function validates that the input is a pandas DataFrame and handles empty dataframes appropriately with warning messages.

### Output validation
Functions that can remove data rows (data cleaning and good road filtering) include output validation checks. If all data is removed during processing, the function raises a `ValueError` with a descriptive message, stopping the pipeline execution.

### Exception types
- `TypeError`: Used for incorrect input types (e.g., non-DataFrame inputs)
- `ValueError`: Used for data content issues (e.g., missing columns, empty results)
- `FileNotFoundError`: Used for missing data files

### Security features
- Path traversal protection prevents access to files outside the Data directory
- Absolute paths are not allowed for security reasons
- Only relative paths within the repository structure are supported

