# Vaylavirasto-data-projekti

This program is made to use open road data together with road measurment data provided by [Väylävirasto](https://vayla.fi/etusivu) to create a visual representation of the Finnish road network condition. Along side with the visual element, the program outputs a geopackage with the data stored in side it.
Custom made machine learning is used to process the data to find anomalies within the road condition data.

## Usage

### Required files
1. Digiroad for the whole of finland (Geopackage R)
2. Excel files with road condition data

Digiroad file can be downloaded from [here](https://aineistot.vayla.fi/spa/ava/Tie/Digiroad/Aineistojulkaisut/latest/) <br>
Digiroad file needs to be extracted into [Root] -> [Data] -> [Digiroad] <br>
When these steps are done, the code will do the rest and automatically search for the files and output the end result into <br>
[Root] -> [output] <br>

### Launching Geopackage Generator
1. Install python 3.13+
2. Create virtual enviorment ```python -m venv venv```
3. Activate virtual enviorment ```venv\Scripts\activate```
4. Install dependencies ```pip install -r GeopackageGenerator/requirements.txt```
5. Launch jupyter ```jupyter notebook```
6. Open geogenerator file
7. Select Kernel -> venv
8. Run all

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
