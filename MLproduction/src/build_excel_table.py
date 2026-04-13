from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------------------------------------
# EXCEL COLOR-CODING
# ----------------------------------------------------------------------------------------------------
def step_07_excel_colours(feature_df, output_path="./MLproduction/production_results_coloured.xlsx"):
    output_path = Path(output_path)

    # Replace zeros 0.0 with 0.01. Don't want to divide by zero.
    denominator = feature_df["yhd_kiiht"].replace(0, 0.01)

    # Calculate ratios. See, which type of acceleration is most dominant.
    feature_df["Pysty_vs_yhdistetty"] = (feature_df["vertical_acceleration"] / denominator).round(3)
    feature_df["Sivu_vs_yhdistetty"] = (feature_df["lateral_acceleration"] / denominator).round(3)
    feature_df["Nyökkimis_vs_yhdistetty"] = (feature_df["longitudinal_acceleration"] / denominator).round(3)

    # Move 'yhd_kiiht' column to the rightmost position
    cols = list(feature_df.columns)
    if "yhd_kiiht" in cols:
        cols.append(cols.pop(cols.index("yhd_kiiht")))
        feature_df = feature_df[cols]


    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # Write dataframe
        feature_df.to_excel(writer, sheet_name="Data", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Data"]

        last_row = len(feature_df)  # number of rows
        vs_columns = ["Pysty_vs_yhdistetty", "Sivu_vs_yhdistetty", "Nyökkimis_vs_yhdistetty"]

        for col_name in vs_columns:
            if col_name in feature_df.columns:
                col_idx = feature_df.columns.get_loc(col_name)
                worksheet.conditional_format(1, col_idx, last_row, col_idx, {
                    'type': '3_color_scale',
                    'min_type': 'min',
                    'mid_type': 'percentile',
                    'mid_value': 50,
                    'max_type': 'max',
                    'min_color': "#00FF33",   # green
                    'mid_color': "#FFDB4B",   # yellow
                    'max_color': "#FF001E",   # red
                })

        # Add blue fill to specific columns for readability
        blue_fill = workbook.add_format({'bg_color': '#ADD8E6'})  # light blue
        for col_name in ['ura_max', 'harjanne', 'rms_mega_oik', 'delta']:
            if col_name in feature_df.columns:
                col_idx = feature_df.columns.get_loc(col_name)
                # Apply solid fill from row 1 (first data row) to last_row
                worksheet.set_column(col_idx, col_idx, None, blue_fill)
                worksheet.conditional_format(1, col_idx, last_row, col_idx, {
                    'type': 'no_blanks',
                    'format': blue_fill
                })

        print(f"Colours added successfully to {output_path}!")