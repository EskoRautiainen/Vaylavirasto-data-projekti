# ML-putki tieolosuhteiden poikkeamien tunnistukseen

## Yleiskuvaus

Tämä putki lukee tieaineiston Excel-tiedostoista, muodostaa dynaamisen hyvän tien vertailuaineiston, kouluttaa Isolation Forest -mallin ja tallentaa poikkeamatulokset vakavuusluokkineen.

Putken vaiheet:
1. Ladataan Excel-data vaadituilla sarakkeilla (`pys_kiiht`, `siv_kiiht`, `nyo_kiiht`, `yhd_kiiht`, `pituus`)
2. Pidetään vain rivit, joilla `pituus == 10`
3. Puhdistetaan data (puuttuvat, ei-numeeriset ja negatiiviset arvot)
4. Muodostetaan piirteet (`vertical/lateral/longitudinal_acceleration`)
5. Muodostetaan dynaaminen hyvän tien vertailuaineisto nykyisestä ajodatasta
6. Skaloidaan data `RobustScaler`-skaalaajalla (sovitus hyvän tien vertailuaineistoon)
7. Koulutetaan `IsolationForest`, pisteytetään kaikki rivit ja luokitellaan poikkeamat

## Ajo

Aja komennot repositorion juuresta (`vaylavirasto-data-projekti`).

Aktivoi ensin repositorion juuressa oleva virtuaaliympäristö:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
python MLmodel/ml_pipeline.py
python MLmodel/ml_pipeline.py Data
python MLmodel/ml_pipeline.py Data/some_file.xlsx
```

Valinnainen `input_path` annetaan suhteessa repositorion juureen.

## Input-vaatimukset

- Input-tiedostojen tulee olla Excel-muotoa: `.xlsx` tai `.xlsm`.
- Ainakin yhdestä välilehdestä täytyy löytyä sarakkeet:
  `pys_kiiht`, `siv_kiiht`, `nyo_kiiht`, `yhd_kiiht`, `pituus`.
- Lataus käy läpi kaikki välilehdet ja yrittää otsakerivejä `0` ja `1`.
- Väliaikaiset lukitustiedostot (`~$...`) ohitetaan.
- Vain rivit, joilla `pituus == 10`, otetaan mukaan.
- Absoluuttiset input-polut hylätään.
- Polkunavigointi (`..`) hylätään.

## Vertailuaineiston logiikka

Hyvän tien vertailuaineisto muodostetaan dynaamisesti nykyisestä ajodatasta käyttäen arvoja:

- `BASELINE_QUANTILE = 0.40` (`ml_pipeline.py`)
- `MIN_FEATURES_REQUIRED = 2` (`ml_pipeline.py`)

Vaaditut mallipiirteet:
- `vertical_acceleration`
- `lateral_acceleration`
- `longitudinal_acceleration`

Vertailuaineiston metadata tallennetaan:
- `MLmodel/MLfiles/baseline_criteria.json`

## Skaalaus

Skaalauksessa käytetään `RobustScaler`-skaalaajaa (`src/data_scaling.py`), joka sovitetaan hyvän tien vertailuaineistoon. Sama skaalain käytetään sen jälkeen kaiken datan muunnokseen.

Skaalain tallennetaan:
- `MLmodel/MLfiles/scaler.pkl`

## Mallin koulutus

Nykyiset Isolation Forest -parametrit (`src/model_training.py`):
- `contamination=0.10`
- `n_estimators=300`
- `max_samples=0.5`
- `max_features=1.0`
- `bootstrap=False`
- `random_state=42`
- `n_jobs=1`

Päättelylogiikka:
- Malli koulutetaan skaalatulla hyvän tien vertailuaineistolla.
- `decision_function`-pisteet lasketaan vertailuaineistolle ja kaikille riveille.
- Poikkeamakynnys johdetaan hyvän tien pisteistä:
  `GOOD_ROAD_NORMAL_SCORE_QUANTILE = 0.05`.
- Lopullinen `anomaly_prediction` muodostetaan tästä johdetusta kynnyksestä.
- Raakamallin ennuste tallennetaan myös sarakkeeseen `model_prediction_raw`.
- Käytännössä suuremmat kiihtyvyysarvot lisäävät poikkeamaennusteen todennäköisyyttä.
  Yksittäisiä poikkeuksia voi silti esiintyä, koska päätös perustuu kolmen piirteen yhteisvaikutukseen eikä yhteen arvoon.

## Mitä ajo tuottaa

Kaikki tuotokset tallennetaan hakemistoon `MLmodel/MLfiles`:

- `feature_metadata.json`: piirteet ja putken metadata.
- `baseline_criteria.json`: dynaamiset vertailurajat ja säilymisprosentti.
- `scaler.pkl`: sovitettu `RobustScaler`.
- `anomaly_model.pkl`: koulutettu `IsolationForest`-malli.
- `anomaly_results.xlsx`: rivikohtaiset ennusteet, pisteet ja luokat.

## Tulosten tulkinta

`anomaly_results.xlsx`-tiedoston keskeiset sarakkeet:

- `anomaly_prediction`: `-1` = poikkeama, `1` = normaali.
- `anomaly_score`: pienempi arvo = poikkeavampi rivi.
- `anomaly_category`: `Critical`, `Poor`, `Fair`, `Good`, `Excellent`.
- `priority_score`: numeerinen prioriteetti (`Critical=1 ... Excellent=5`).
- `model_prediction_raw`: raaka `IsolationForest.predict` -tulos ennen mukautettua kynnyslogiikkaa.

Käytännön tulkinta:
- Suuremmat kiihtyvyysarvot ohjautuvat yleensä poikkeamaluokkiin.
- Luokittelu on monimuuttujainen, joten yksittäisiä sarakekohtaisia poikkeuksia voi olla.

## Inference Script (Model Usage)

Example inference flow for new data using saved model + scaler:

```python
from pathlib import Path
import joblib
import pandas as pd

FEATURES = [
    "vertical_acceleration",
    "lateral_acceleration",
    "longitudinal_acceleration",
]

model = joblib.load(Path("MLmodel/MLfiles/anomaly_model.pkl"))
scaler = joblib.load(Path("MLmodel/MLfiles/scaler.pkl"))

# df_raw must already contain engineered features with these exact names
missing = [c for c in FEATURES if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing required features: {missing}")

# Enforce exact feature order before scaling and prediction
X = df_raw.loc[:, FEATURES].copy()
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES, index=X.index)

scores = model.decision_function(X_scaled_df)
raw_pred = model.predict(X_scaled_df)  # -1 anomaly, 1 normal
```

Huomiot:
- Piirteiden nimet ja järjestys pitää säilyttää samoina kuin koulutuksessa.
- Skaalaa data aina tallennetulla `scaler.pkl`-tiedostolla ennen mallikutsuja.

## Riippuvuudet

Asenna riippuvuudet:

```bash
pip install -r requirements.txt
```

## Mallin toimivuuden testaus (poimintoja tulostiedostosta)

Tiedosto `MLmodel/MLfiles/anomaly_results.xlsx` soveltuu mallin käyttäytymisen validointiin (onko tulos looginen), mutta ei yksin riitä tarkkuuden arviointiin ilman ground truth -labeleita.

Poimintoja nykyisestä tulosajosta:
- Rivejä yhteensä: `1,042,431`
- `anomaly_prediction = -1`: `479,379` (45.99 %)
- `anomaly_prediction = 1`: `563,052` (54.01 %)
- `model_prediction_raw` ja lopullinen `anomaly_prediction` erosivat `78,776` rivillä (7.56 %), koska lopullinen päätös käyttää erillistä score-kynnyslogiikkaa.
- Kategoriat jakautuivat: `Critical` 11.50 %, `Poor` 11.50 %, `Fair` 22.99 %, `Good` 40.44 %, `Excellent` 13.57 %.

Mitä tämän perusteella voi päätellä:
- Luokittelu toimii johdonmukaisesti: suuremmat kiihtyvyydet painottuvat alempiin laatuluokkiin (`Critical/Poor/Fair`) ja pienemmät `Good/Excellent`-luokkiin.
- Score-jakauma erottaa normaalit ja poikkeamat selkeästi.

Mitä tämän perusteella ei voi päätellä:
- Precision/recall/F1-arvoja ei voi arvioida luotettavasti ilman erillistä käsin labeloitua testidataa.
