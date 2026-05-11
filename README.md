# Vaylavirasto-data-projekti

Vaylavirasto-data-projekti is a data processing and machine learning pipeline built for analysing Finnish road condition data provided by Väylävirasto (https://vayla.fi/etusivu). The system combines Digiroad road network data with road measurement Excel datasets to detect anomalies in road conditions using a custom-built machine learning pipeline.

The solution is composed of four main parts:
MLmodel: trains a machine learning model that learns what “good road condition” looks like and outputs model artifacts (e.g. scaler and anomaly model).
MLproduction: runs inference on new road measurement data and produces scored results and prioritised anomaly classifications in Excel format.
GeoPackage generator: merges the processed results with Digiroad geospatial data and produces a GeoPackage file for map-based visualization (e.g. QGIS), enabling spatial analysis of road condition anomalies.
Correlation analysis: Use Pearson correlation to calculate correlations between yhd_kiiht and other road metadata.

The project was developed as part of a data and AI project course at Tampere University of Applied Sciences (https://www.tuni.fi/fi/tutustu-meihin/tamk).

## Usage

### Required files
1. Digiroad for the whole of Finland (Geopackage R)
2. Excel files with road condition data

Digiroad file can be downloaded from [here](https://aineistot.vayla.fi/spa/ava/Tie/Digiroad/Aineistojulkaisut/latest/) <br>
Digiroad file needs to be extracted into [Root] -> [Data] -> [Digiroad] <br>
When these steps are done, the code will do the rest and automatically search for the files and output the end result into <br>
[Root] -> [output] <br>

### Launching MLmodel - training pipeline
1. Install python 3.13+
2. Create virtual environment ```python -m venv venv```
3. Activate virtual environment ```venv\Scripts\activate```
4. Install dependencies ```pip install -r MLmodel/requirements.txt```
5. Run training pipeline ```python MLmodel/ml_pipeline.py```

### Launching MLproduction - production pipeline
1. Install python 3.13+
2. Create virtual environment ```python -m venv venv```
3. Activate virtual environment ```venv\Scripts\activate```
4. Install dependencies ```pip install -r MLmodel/requirements.txt```
   > MLproduction uses MLmodel dependencies
6. Run production pipeline ```python -m MLproduction/production_pipeline.py```

### Launching Geopackage Generator
1. Install python 3.13+
2. Create virtual environment ```python -m venv venv```
3. Activate virtual environment ```venv\Scripts\activate```
4. Install dependencies ```pip install -r GeopackageGenerator/requirements.txt```
5. Launch jupyter ```jupyter notebook```
6. Open geogenerator file
7. Select Kernel -> venv
8. Run all

### Launching Correlation Analysis
1. Install python 3.13+
2. Create virtual environment ```python -m venv venv```
3. Activate virtual environment ```venv\Scripts\activate```
4. Install dependencies ```pip install -r MLmodel/requirements.txt```
5. Run program ```python Correlations/correlation_analysis.py```

## Contributing

Not accepting outside contributions during development.

## Authors and acknowledgment

Programmers:
- Esko Rautiainen - [Github](https://github.com/EskoRautiainen)
- Aleksi Malminen - [Github](https://github.com/AleksiMal)
- Kimmo Vuori - [Github](LINK)
- Eeli Kemppainen

Supporting us are our teachers:
- Anne-Mari Stenbacka
- Jere Käpyaho

Project provided by:
- [Tampere University of Applied Sciences](https://www.tuni.fi/fi/tutustu-meihin/tamk)
- [Väylävirasto](https://vayla.fi/etusivu)
