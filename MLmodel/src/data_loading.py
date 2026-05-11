from __future__ import annotations

from pathlib import Path

import pandas as pd


EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
OUTPUT_COLUMNS = [
    "pys_kiiht",
    "siv_kiiht",
    "nyo_kiiht",
    "yhd_kiiht",
]


# -------------------------
# EXCEL FILE DISCOVERY
# -------------------------
def _discover_excel_files(anchor_path: Path) -> list[Path]:
    if anchor_path.exists() and anchor_path.is_dir():
        data_dir = anchor_path
    else:
        data_dir = anchor_path.parent

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found from input path: {anchor_path}"
        )

    excel_files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in EXCEL_SUFFIXES
        and not path.name.startswith("~$")
    )
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in directory: {data_dir}")
    return excel_files


# -------------------------
# COLUMN NAME NORMALIZATION
# -------------------------
def _column_map(columns: pd.Index) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for col in columns:
        if isinstance(col, str):
            mapped[col.strip().lower()] = col
    return mapped


# -------------------------
# REQUIRED FEATURE RESOLUTION
# -------------------------
def _resolve_feature_columns(dataframe: pd.DataFrame) -> dict[str, str]:
    cols = _column_map(dataframe.columns)
    resolved = {
        "pys_kiiht": cols.get("pys_kiiht"),
        "siv_kiiht": cols.get("siv_kiiht"),
        "nyo_kiiht": cols.get("nyo_kiiht"),
        "yhd_kiiht": cols.get("yhd_kiiht"),
        "pituus": cols.get("pituus"),
    }
    if any(value is None for value in resolved.values()):
        return {}
    return resolved  # type: ignore[return-value]


# -------------------------
# MATCHING SHEET READING
# -------------------------
def _read_matching_sheets(file_path: Path) -> list[tuple[pd.DataFrame, str, int]]:
    with pd.ExcelFile(file_path) as excel_file:
        sheet_names = excel_file.sheet_names

    # Process every worksheet and detect required columns dynamically.
    # Do not rely on worksheet names because source files may vary.
    matches: list[tuple[pd.DataFrame, str, int]] = []
    for sheet_name in sheet_names:
        for header_row in (0, 1):
            try:
                dataframe = pd.read_excel(
                    file_path, sheet_name=sheet_name, header=header_row
                )
            except Exception:
                continue
            if dataframe.empty:
                continue
            resolved = _resolve_feature_columns(dataframe)
            if resolved:
                filtered = dataframe.loc[
                    :,
                    [
                        resolved["pys_kiiht"],
                        resolved["siv_kiiht"],
                        resolved["nyo_kiiht"],
                        resolved["yhd_kiiht"],
                        resolved["pituus"],
                    ],
                ].copy()
                filtered.columns = OUTPUT_COLUMNS + ["pituus"]
                matches.append((filtered, sheet_name, header_row))
                break

    if not matches:
        raise ValueError(
            f"No worksheet with required columns found in file: {file_path.name}. "
            "Required columns: pys_kiiht, siv_kiiht, nyo_kiiht, yhd_kiiht, pituus"
        )

    return matches


# -------------------------
# PITUUS FILTERING
# -------------------------
def _filter_pituus_10(dataframe: pd.DataFrame) -> pd.DataFrame:
    pituus_numeric = pd.to_numeric(
        dataframe["pituus"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return dataframe.loc[pituus_numeric == 10, OUTPUT_COLUMNS].copy()


# -------------------------
# DATA LOADING
# -------------------------
def step_01_load_data(file_path: Path) -> pd.DataFrame:
    validated_path = Path(file_path)
    excel_files = _discover_excel_files(validated_path)

    print("Reading Excel files. Please wait...")
    print()

    filtered_frames: list[pd.DataFrame] = []
    total_rows_picked = 0

    for excel_file in excel_files:
        sheet_matches = _read_matching_sheets(excel_file)
        file_rows_picked = 0
        included_sheet_names: list[str] = []

        for source_dataframe, sheet_name, _ in sheet_matches:
            filtered_source = _filter_pituus_10(source_dataframe)
            rows_picked = len(filtered_source)
            if rows_picked > 0:
                filtered_frames.append(filtered_source)
                file_rows_picked += rows_picked
                included_sheet_names.append(sheet_name)

        total_rows_picked += file_rows_picked
        sheets_display = ", ".join(sorted(set(included_sheet_names))) if included_sheet_names else "none"
        print(
            f"{excel_file.name} | sheets_included: {sheets_display} | rows_picked: {file_rows_picked}"
        )

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
