# python -m pip install openpyxl
# pip install matlab --upgrade
# pip install scikit-learn --upgrade
# pip install ipython -- upgrade
# pip install xlsxwriter -- upgrade

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os
import xlsxwriter

# ==============================
# PARAMETERS
# ==============================
good_roads_file_path = "output/engineered_features.xlsx"
all_roads_file_path = "output/all_roads.xlsx"
output_excel_path = "output/anomaly_score.xlsx"
contamination = 0.12     # expected fraction of anomalies
n_estimators = 200
random_state = 42

# ==============================
# LOAD DATA
# ==============================
good_roads_dataframe = pd.read_excel(good_roads_file_path)
all_roads_dataframe = pd.read_excel(all_roads_file_path)

# ==============================
# ADDS NEW AVARAGE COLUMN
# ==============================
good_roads_dataframe ["Summattu_kiiht"] = (
    good_roads_dataframe["Pysty_kiiht"] +
    good_roads_dataframe["Sivuheilahdus_kiiht"] +
    good_roads_dataframe["Nyökkimis_kiiht"]
)

all_roads_dataframe["Summattu_kiiht"] = (
    all_roads_dataframe["Pysty_kiiht"] +
    all_roads_dataframe["Sivuheilahdus_kiiht"] +
    all_roads_dataframe["Nyökkimis_kiiht"]
)

# ==============================
# DEFINE FEATURES
# ==============================

features = [
    "Pysty_kiiht",
    "Sivuheilahdus_kiiht",
    "Nyökkimis_kiiht",
    "Yhdistetty_kiiht_rms",
    "Summattu_kiiht"
]

# ==============================
#  DEFINE MODEL
# ==============================
model = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    random_state=random_state
)

# ==============================
# SCALE FEATURES & TRAIN
# ==============================
scaler = StandardScaler()

X_train = scaler.fit_transform(good_roads_dataframe[features])
X_all = scaler.transform(all_roads_dataframe[features])

model.fit(X_train)

# ==============================
# COMPUTE ANOMALY SCORES
# ==============================
all_roads_dataframe["Poikkeamapisteet"] = model.decision_function(X_all).round(3)


all_roads_dataframe["Poikk_voimakkuus"] = pd.cut(
    all_roads_dataframe["Poikkeamapisteet"],
    bins=[-1, -0.132, -0.05, 0, 1],
    labels=["Erittäin vahva", "Vahva", "Heikko", "Normaali"]
)

# ==============================
# COMPUTE RATIOS VS YHDISTETTY_kiiht_rms
# ==============================
feature_df = all_roads_dataframe.copy()
feature_df["Pysty_vs_yhdistetty"] = (feature_df["Pysty_kiiht"] / feature_df["Yhdistetty_kiiht_rms"].replace(0, pd.NA)).round(3)
feature_df["Sivu_vs_yhdistetty"] = (feature_df["Sivuheilahdus_kiiht"] / feature_df["Yhdistetty_kiiht_rms"].replace(0, pd.NA)).round(3)
feature_df["Nyökkimis_vs_yhdistetty"] = (feature_df["Nyökkimis_kiiht"] / feature_df["Yhdistetty_kiiht_rms"].replace(0, pd.NA)).round(3)


# ==============================
# SAVE RESULTS
# ==============================
# Excel with anomaly score
all_roads_dataframe.to_excel(output_excel_path, index=False)

# Excel with ratios (plain)
feature_excel_path = "output/anomaly_score_and_additional_info.xlsx"
feature_df.to_excel(feature_excel_path, index=False)

# Excel with ratios + colors
feature_colored_excel_path = "output/anomaly_score_and_additional_info_colored.xlsx"

# ==============================
# SAVE RESULTS WITH 3-COLOR HEATMAP
# ==============================
feature_colored_excel_path = "output/anomaly_score_and_additional_info_colored.xlsx"

with pd.ExcelWriter(feature_colored_excel_path, engine="xlsxwriter") as writer:
    # Write dataframe
    feature_df.to_excel(writer, sheet_name="Data", index=False)

    workbook = writer.book
    worksheet = writer.sheets["Data"]

    last_row = len(feature_df)  # number of rows
    vs_columns = ["Pysty_vs_yhdistetty", "Sivu_vs_yhdistetty", "Nyökkimis_vs_yhdistetty"]

    for col_name in vs_columns:
        col_idx = feature_df.columns.get_loc(col_name)

        # Apply 3-color scale ONLY to this column
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
        print("Colours added succesfully!")



# ==============================
# SUMMARY
# ==============================
feature_summary_df = feature_df.describe().transpose().reset_index().rename(columns={"index": "ominaisuus"})

print(f"Tallennettu {len(feature_df)} riviä tiedostoon: {feature_excel_path}")

print(feature_df.head(10))
print(feature_summary_df)

