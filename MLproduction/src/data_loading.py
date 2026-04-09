from __future__ import annotations

from pathlib import Path

import pandas as pd


def step_01_load_data(file_path: Path) -> pd.DataFrame:
    validated_path = Path(file_path)

    if not validated_path.exists():
        raise FileNotFoundError(f"File not found: {validated_path}")

    if validated_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError(
            "Invalid file type. Please provide an Excel file with the extension .xlsx or .xlsm."
        )

    sheet_headers = {
        "Raportti 100m": 1,
        "Raportti 10m MALLI": 0,
    }
    selected_features = [
        'Karttapvm',
        'Elinvoimakeskus',
        'Ajorata',
        'Kaista',
        'Aosa',
        'Aet',
        'Losa',
        'Let',
        'Pituus',
        'Ura_max',
        'Harjanne',
        'Kaltevuus',
        'Rms_mega_oik',
        'Delta',
        "Pysty_kiiht",
        "Sivuheilahdus_kiiht",
        "Nyökkimis_kiiht",
        "Yhdistetty_kiiht_rms",
        'Päällyste_pak'
    ]
    filtered_dataframe: pd.DataFrame | None = None

    with pd.ExcelFile(validated_path) as excel_file:
        available_sheet_names = excel_file.sheet_names
    missing_sheet_names = [
        sheet_name for sheet_name in sheet_headers if sheet_name not in available_sheet_names
    ]

    if missing_sheet_names:
        raise ValueError(
            "The Excel file is missing required worksheet(s): "
            f"{', '.join(missing_sheet_names)}. "
            f"Available worksheets: {', '.join(available_sheet_names)}"
        )

    print("Reading selected worksheets. Please wait...")
    print()

    for sheet_name, header_row in sheet_headers.items():
        dataframe = pd.read_excel(validated_path, header=header_row, sheet_name=sheet_name)
        if dataframe.empty:
            raise ValueError(
                "The following required worksheet does not contain data rows: "
                f"{sheet_name}"
            )

        first_row = dataframe.iloc[0]
        for column_name, value in first_row.items():
            print(f"{sheet_name} | {column_name}: {value}")

        if sheet_name == "Raportti 10m MALLI":
            missing_features = [
                feature for feature in selected_features if feature not in dataframe.columns
            ]
            if missing_features:
                raise ValueError(
                    "Required worksheet 'Raportti 10m MALLI' is missing selected ML feature columns: "
                    f"{', '.join(missing_features)}. "
                    f"Selected ML features: {', '.join(selected_features)}"
                )

            filtered_dataframe = dataframe.loc[:, selected_features].copy()
        del dataframe

    if filtered_dataframe is None:
        raise ValueError(
            "The required worksheet 'Raportti 10m MALLI' did not produce usable ML feature data."
        )

    print()
    print("------------------------------------------------------------")
    print("The data has been filtered to the selected ML features.")
    print()
    print(filtered_dataframe.head(5).to_string())

    return filtered_dataframe
