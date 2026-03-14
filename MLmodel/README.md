# README


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
3. `step_03_filter_good_road()` - filters to good road conditions using 80 km/h specific thresholds
4. `step_04_engineer_features()` - selects final ML features and removes redundant columns

Each step includes comprehensive error handling and data validation. If any step removes all data, the pipeline stops execution with a descriptive error message.

## 2.1 step_01_load_data()

The `step_01_load_data()` function loads the configured Excel worksheets using worksheet-specific header row settings. If any required worksheet is missing or does not contain data rows, the function raises an error.

It first prints the first data row in the format `Worksheet | Column: value` so the loaded worksheet content can be validated before further processing.

After the validation output, the function validates the `Raportti 10m MALLI` worksheet and filters its data to the selected ML feature columns:

- `Pysty_kiiht`
- `Sivuheilahdus_kiiht`
- `Nyökkimis_kiiht`
- `Yhdistetty_kiiht_rms`

Both required worksheets must exist in the Excel file:

- `Raportti 100m` - read for user validation and future use
- `Raportti 10m MALLI` - used as the primary data source for ML features

The `Raportti 100m` worksheet is read to validate that both worksheets are accessible and to display the first data row to the user. Currently, this data is not used in the ML processing but may be utilized in future development.

The `Raportti 10m MALLI` worksheet must contain all four selected ML feature columns. If any of these columns are missing, the function raises an error.

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

## 2.3 step_03_filter_good_road()

The `step_03_filter_good_road()` function receives the cleaned `DataFrame` returned by `step_02_clean_data()` and filters it to retain only road segments that represent good road surface conditions.

This function uses speed-agnostic thresholds based on 25th percentile values from all speed data, providing a broader baseline for anomaly detection across different speed limits.

The function applies the following filtering criteria based on the 25th percentile values of all speed data:

- `Pysty_kiiht ≤ 0.05`
- `Sivuheilahdus_kiiht ≤ 1.0`
- `Nyökkimis_kiiht ≤ 3.0`

### Why these values?

Think of this like setting quality standards for road smoothness. We analyzed thousands of road measurements from all speed limits to find what makes a "perfectly smooth" road.

#### The simple explanation:

**We studied all speed data** (40 km/h, 60 km/h, 80 km/h, 100 km/h, etc.) to create a comprehensive baseline that works for any road. This speed-agnostic approach provides more data for better ML training.

#### How we found the perfect road standards:

1. **Collected data**: We gathered 1576 road measurements from all speed limits
2. **Found the best 25%**: We looked for the smoothest 25% of all road measurements
3. **Set the standards**: These roads became our "perfect road" baseline

#### What the numbers mean in plain language:

| Measurement | What it measures | Perfect road limit | Everyday meaning |
|-------------|------------------|-------------------|------------------|
| `Pysty_kiiht ≤ 0.05` | Vertical bouncing | ≤ 0.05 | Car doesn't bounce up and down |
| `Sivuheilahdus_kiiht ≤ 1.0` | Side-to-side swaying | ≤ 1.0 | Car doesn't sway left and right |
| `Nyökkimis_kiiht ≤ 3.0` | Forward-backward nodding | ≤ 3.0 | Car doesn't nod forward/backward |

#### Why this matters for finding problems:

**Perfect baseline makes problems obvious:**
- When we know what a "perfect road" feels like (these limits)
- Any road that's worse than perfect is easy to spot
- It's like having a ruler - you can measure exactly how bad a problem is

#### The results in simple numbers:
- **18.8% of all roads** meet these perfect standards (297 rows)
- **81.2% of roads** have some level of problems
- This gives us a very clear "perfect vs. imperfect" comparison with 2.25x more data than the previous 80 km/h specific approach

#### Real-world benefit:
When the car drives on a road that's worse than these perfect standards, our system immediately knows: *"This road has problems"* - because we have scientifically proven what "perfect" actually means across all speed limits.

The function prints filtering statistics including the number of rows before and after filtering and displays a five-row preview of the filtered good road dataset.

Finally, the function returns the filtered `DataFrame` containing only good road segments.

## 2.4 step_04_engineer_features()

The `step_04_engineer_features()` function receives the good road filtered `DataFrame` returned by `step_03_filter_good_road()` and performs feature selection to prepare the final feature set for the ML model.

### Feature Selection Process
The function selects only the three primary acceleration measurement features:

- `Pysty_kiiht` - vertical acceleration
- `Sivuheilahdus_kiiht` - side-to-side swaying acceleration  
- `Nyökkimis_kiiht` - forward-backward nodding acceleration

### Removal of Redundant Feature
The function removes the `Yhdistetty_kiiht_rms` (combined RMS acceleration) column because it is **redundant** with the three individual acceleration measurements. The combined RMS value can be mathematically derived from the individual components, so removing it eliminates redundant information and simplifies the model without losing predictive power.

### Feature Engineering Foundation
This function serves as the foundation for future feature engineering capabilities. While the current implementation focuses on feature selection, this is the designated location where new feature creation will be implemented in future development.

### Future Processing Steps
Note that **normalization/scaling** and **dataset splitting** will be performed in later pipeline stages, not in this function.

The function prints a five-row table preview of the engineered feature set.

Finally, the function returns the engineered `DataFrame` with the selected features, ready for subsequent ML processing steps.

## 3. Error handling and data validation

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
