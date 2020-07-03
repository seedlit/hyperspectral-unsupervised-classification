# Unsupervised classification of hyperspectral remote sensing imagery
This repo produces classified raster and shapefile output from input Hyperspectral image.

## Features
- The input could be a directory containing indvidual bands as GeoTIFF rasters. It will stack all the bands together.
If the input is a single GeoTIFF raster containing all the bands, use `img_to_array` function to get all the bands as array.
- Uses K-means clustering algorithm.
- Option to perform min-max feature scaling.
- Option to perform Principal Component Analysis (PCA).
- Classified output raster is georeferenced wrt to the input GeoTIFF.
- Produces corresponding shapefiles as well. Shapefile is generated separately for each class. The shapefiles also contain 'Area' field.

## Data and Results
The hyperspectral data that I used is from Nasa's Earth Observing-1. 
There are three AOIs (Area Of Interests) - agricultural, cloudy, and LULC (Land Use Land Cover).
For each AOI, there are 242 spectral bands each belonging to different wavelength range.
The input data can be found [here](https://drive.google.com/drive/folders/1RkcIBpgFQfwIAZuBSgrIdVJhaEvcUrRE?usp=sharing). The results with performing feature scaling can be found [here](https://drive.google.com/drive/folders/119ILfRvftV7TQeCoqbTo0n6loZFbv5gx?usp=sharing). The results without performing feature scaling can be found [here](https://drive.google.com/drive/folders/19MhLs9Ep3lRIw5GP6lN2f7XzjsrYTAJS?usp=sharing).

### Sample Images
Classified crops output <br />
<p align="center">
<img src="https://github.com/seedlit/hyperspectral-unsupervised-classification/blob/master/images/crops_cluster.png?raw" height = "600" width = "970">
</p>

Crops shapefiles with RGB basemap <br />
<p align="center">
<img src="https://github.com/seedlit/hyperspectral-unsupervised-classification/blob/master/images/crops_shp.png?raw" height = "600" width = "970">
</p>

Water body shapefile with RGB basemap <br />
<p align="center">
<img src="https://github.com/seedlit/hyperspectral-unsupervised-classification/blob/master/images/crop_water_shp.png?raw" height = "600" width = "970">
</p>

Input hyperspectral imagery <br />
<p align="center">
<img src="https://github.com/seedlit/hyperspectral-unsupervised-classification/blob/master/images/clouds_without_shp.png?raw" height = "600" width = "600">
</p>

Output clouds shapefile <br />
<p align="center">
<img src="https://github.com/seedlit/hyperspectral-unsupervised-classification/blob/master/images/clouds_shp.png?raw" height = "600" width = "600">
</p>

Simplified shapefile <br />
<p align="center">
<img src="https://github.com/seedlit/hyperspectral-unsupervised-classification/blob/master/images/simplified_Shp.png?raw" height = "400" width = "600">
</p>
