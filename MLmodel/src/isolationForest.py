# python -m pip install openpyxl
# pip install matlab --upgrade
# pip install scikit-learn --upgrade
# pip install ipython -- upgrade

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from IPython.display import display
from sklearn.model_selection import train_test_split

# ==============================
# PARAMETERS
# ==============================
file_path = "MLmodel/output/engineered_features.xlsx"
output_excel_path = "MLmodel/output/simple_anomaly_score.xlsx"
contamination = 0.07     # expected fraction of anomalies
n_estimators = 200
random_state = 42

# ==============================
# LOAD DATA
# ==============================
dataframe = pd.read_excel(file_path)

# ==============================
# AVARAGE - ADDS NEW COLUMN "Summattu_kiiht"
# ==============================
dataframe ["Summattu_kiiht"] = (
dataframe["Pysty_kiiht"] +
dataframe["Sivuheilahdus_kiiht"] +
dataframe["Nyökkimis_kiiht"]
)

# ==============================
# FEATURE SCALING
# ==============================
#features_for_scaling = ["Pysty_kiiht", "Sivuheilahdus_kiiht", "Nyökkimis_kiiht", "Yhdistetty_kiiht_rms", "Summattu_kiiht"]
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(dataframe[features_for_scaling])
#scaled_df = pd.DataFrame(X_scaled, columns=features_for_scaling)

# ==============================
# TRAIN ISOLATION FOREST ON GOOD ROADS (BOTTOM 25% FOR PER VARIABLE)
# ==============================
threshold = dataframe["Summattu_kiiht"].quantile(0.7)
good_roads = dataframe[dataframe["Summattu_kiiht"] <= threshold]
train_data, val_data = train_test_split(good_roads, test_size=0.2, random_state=random_state)

model = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    random_state=random_state
)
model.fit(train_data[["Pysty_kiiht", "Sivuheilahdus_kiiht", "Nyökkimis_kiiht", "Yhdistetty_kiiht_rms"]])

# ==============================
# COMPUTE ANOMALY SCORES
# ==============================
dataframe["Poikkeamapisteet"] = model.decision_function(
    dataframe[["Pysty_kiiht", "Sivuheilahdus_kiiht", "Nyökkimis_kiiht", "Yhdistetty_kiiht_rms"]]
).round(3)

# ==============================
# COMPUTE RATIOS VS YHDISTETTY_kiiht_rms
# ==============================
feature_df = dataframe.copy()
feature_df["Pysty_vs_yhdistetty"] = (feature_df["Pysty_kiiht"] / feature_df["Yhdistetty_kiiht_rms"].replace(0, pd.NA)).round(3)
feature_df["Sivu_vs_yhdistetty"] = (feature_df["Sivuheilahdus_kiiht"] / feature_df["Yhdistetty_kiiht_rms"].replace(0, pd.NA)).round(3)
feature_df["Nyokkimis_vs_yhdistetty"] = (feature_df["Nyökkimis_kiiht"] / feature_df["Yhdistetty_kiiht_rms"].replace(0, pd.NA)).round(3)

# ==============================
# SAVE RESULTS
# ==============================
# Excel with anomaly score
dataframe.to_excel(output_excel_path, index=False)

# Excel with ratios
feature_excel_path = "MLmodel/output/anomaly_score_and_ride_vs_yhdistetty_rms.xlsx"
feature_df.to_excel(feature_excel_path, index=False)

# ==============================
# SUMMARY
# ==============================
feature_summary_df = feature_df.describe().transpose().reset_index().rename(columns={"index": "ominaisuus"})

print(f"Tallennettu {len(feature_df)} riviä tiedostoon: {feature_excel_path}")

print(feature_df.head(10))
print(feature_summary_df)

