from __future__ import annotations
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------------------------------------
#   DEFINE CONSTANTS
# ----------------------------------------------------------------------------------------------------
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
OUTPUT_COLUMNS = [
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
    'pys_kiiht',
    'siv_kiiht',
    'nyo_kiiht',
    'yhd_kiiht',
    'tl332_paapak'
]

# ----------------------------------------------------------------------------------------------------
#   FILE DISCOVERY
# ----------------------------------------------------------------------------------------------------
def _discover_excel_files(anchor_path: Path) -> list[Path]: # Find ALL Excel files in a folder
    if anchor_path.exists() and anchor_path.is_dir():
        data_dir = anchor_path                              # If user gives folder -> use it
    else:                                                   
        data_dir = anchor_path.parent                       # If user gives file -> use parent folder

    if not data_dir.exists() or not data_dir.is_dir():      # Check if folder exists
        raise FileNotFoundError(
            f"Data directory not found from input path: {anchor_path}"
        )

    excel_files = sorted(
        path
        for path in data_dir.iterdir()                      # Loops through all Excel files
        if path.is_file()                                   # Ignore folders
        and path.suffix.lower() in EXCEL_SUFFIXES           # Filter only Excel files
        and not path.name.startswith("~$")                  # Ignores Excel temporary lock files
    )
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in directory: {data_dir}")  # If no Excels found, raise error
    return excel_files                                      # Return list of Excel files


# ----------------------------------------------------------------------------------------------------
#   NORMALIZE COLUMNS
# ----------------------------------------------------------------------------------------------------
def _column_map(columns: pd.Index) -> dict[str, str]:
    mapped: dict[str, str] = {}                             # Create empty dictionary
    for col in columns:                                                         
        if isinstance(col, str):                            # Ensure column is string
            mapped[col.strip().lower()] = col               # Remove spaces and lowercase
    return mapped


# ----------------------------------------------------------------------------------------------------
#   CONFIRM TARGET COLUMNS EXIST
# ----------------------------------------------------------------------------------------------------
def _resolve_feature_columns(dataframe: pd.DataFrame) -> dict[str, str]:
    cols = _column_map(dataframe.columns)
    resolved = {
        "pys_kiiht": cols.get("pys_kiiht"),
        "siv_kiiht": cols.get("siv_kiiht"),
        "nyo_kiiht": cols.get("nyo_kiiht"),
        "yhd_kiiht": cols.get("yhd_kiiht"),
        "pituus": cols.get("pituus"),
    }
    if any(value is None for value in resolved.values()):   # Validate if all required columns exist
        return {}
    return resolved  # type: ignore[return-value]


# ----------------------------------------------------------------------------------------------------
#   FIND CORRECT SHEET IN EXCEL
# ----------------------------------------------------------------------------------------------------
def _read_matching_sheet(file_path: Path) -> tuple[pd.DataFrame, str, int]:
    with pd.ExcelFile(file_path) as excel_file:             # Open file without loading all data
        sheet_names = excel_file.sheet_names                # Get sheet names

    # Process every worksheet and detect required columns dynamically.
    # Do not rely on worksheet names because source files may vary.
    for sheet_name in sheet_names:
        for header_row in (0, 1):                           # Try different header positions
            dataframe = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row) # Load sheets into DataFrame
            if dataframe.empty:
                continue
            resolved = _resolve_feature_columns(dataframe)
            if resolved:
                renamed_dataframe = dataframe.rename(columns={
                    resolved["pys_kiiht"]: "pys_kiiht",
                    resolved["siv_kiiht"]: "siv_kiiht",
                    resolved["nyo_kiiht"]: "nyo_kiiht",
                    resolved["yhd_kiiht"]: "yhd_kiiht",
                    resolved["pituus"]: "pituus",
            })

            return renamed_dataframe, sheet_name, header_row

    raise ValueError(
        f"No worksheet with required columns found in file: {file_path.name}. "
        "Required columns: pys_kiiht, siv_kiiht, nyo_kiiht, yhd_kiiht, pituus"
    )


# ----------------------------------------------------------------------------------------------------
#   KEEP ONLY ROWS WHERE PITUUS IS 10
# ----------------------------------------------------------------------------------------------------
def _filter_pituus(dataframe: pd.DataFrame) -> pd.DataFrame:
    pituus_numeric = pd.to_numeric(
        dataframe["pituus"].astype(str).str.replace(",", ".", regex=False), # Handle decimal format
        errors="coerce",                                                    # Invalid values become NaN
    )
    mask_valid = pituus_numeric == 10
    mask_removed = ~mask_valid

    removed_count = mask_removed.sum()
    print(f"Removed {removed_count} rows where 'pituus' is not 10")

    return dataframe.loc[mask_valid].copy()

# ----------------------------------------------------------------------------------------------------
#   MAIN LOADER - ENTRY POINT
# ----------------------------------------------------------------------------------------------------
def step_01_load_data(file_path: Path) -> pd.DataFrame:
    validated_path = Path(file_path)
    excel_files = _discover_excel_files(validated_path)
    print("------------------------------------------------------------")
    print()
    print("Reading Excel files. Please wait...")
    print()

    filtered_frames: list[pd.DataFrame] = []
    total_rows_picked = 0

    for excel_file in excel_files:
        source_dataframe, sheet_name, _ = _read_matching_sheet(excel_file)
        filtered_source = _filter_pituus(source_dataframe)
        rows_picked = len(filtered_source)
        total_rows_picked += rows_picked

        print(
            f"{excel_file.name} | sheet name: {sheet_name} | rows_picked: {rows_picked}"
        )

        if rows_picked > 0:
            filtered_frames.append(filtered_source)

    if not filtered_frames:
        raise ValueError(
            "No rows remained after applying filter pituus == 10 across all Excel files."
        )

    filtered_dataframe = pd.concat(filtered_frames, ignore_index=True)

    print()
    print("------------------------------------------------------------")
    print("Data loading summary")
    print(f"files_read: {len(excel_files)} | total_rows_picked: {total_rows_picked}")
    print()
    print(filtered_dataframe.head(5).to_string())

    return filtered_dataframe