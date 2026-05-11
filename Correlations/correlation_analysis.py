from pathlib import Path
import pandas as pd

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "output"

FEATURES = [
    "ura_max",
    "harjanne_ka",
    "rms_mega_oik",
    "delta",
    "tl332_paapak"
]

TARGET = "yhd_kiiht"

# Combine everything into one list
ALL_COLUMNS = FEATURES + [TARGET]

# --------------------------------------------------
# DEBUG INFO
# --------------------------------------------------
print("-------- DEBUG INFO --------")
print("Script location:", Path(__file__).resolve())
print()
print("Looking for files in:", DATA_DIR)
print()
print("Exists:", DATA_DIR.exists())
print()
print("Files found:", list(DATA_DIR.glob("*")))
print()

# --------------------------------------------------
# LOAD FILE
# --------------------------------------------------
def load_data():
    # Finds all Excel files
    files = list(DATA_DIR.glob("*.xls*"))

    if not files:
        raise FileNotFoundError(f"No Excel files found in {DATA_DIR}")

    file_path = files[0]
    print(f"\nReading file: {file_path.name}")

    # Load Excel into a pandas dataframe
    return pd.read_excel(file_path)


# --------------------------------------------------
# CLEAN NUMERIC DATA
# --------------------------------------------------
def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ALL_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                # Convert everything to string (commas to dots)
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --------------------------------------------------
# CORRELATION VS TARGET
# --------------------------------------------------
def compute_correlations(df: pd.DataFrame):
    correlations = {}

    for col in FEATURES:
        # Computes Pearson correlation ( -1 -> 1 )
        corr_value = df[col].corr(df[TARGET])
        correlations[col] = corr_value

    return correlations


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    df = load_data()

    # Validate columns
    missing = [col for col in ALL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = clean_numeric(df)

    # Keep only relevant columns and drop NaN rows
    df = df[ALL_COLUMNS].dropna()

    print("\nData sample:")
    print(df.head(), "\n")

# --------------------------------------------------
# CORRELATION
# --------------------------------------------------
    correlations = compute_correlations(df)

    print("--------------------------------------------------")
    print(f"Correlation vs {TARGET} (Pearson):")
    print("--------------------------------------------------")

    # Sort by absolute strength
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for col, value in sorted_corr:
        print(f"{col:20} -> {value:.3f}")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()