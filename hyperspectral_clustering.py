"""
hyperspectral_clustering.py
Clustering using hyperspectral remote sensing data.
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
        www.namanji.wixsite.com/naman/
"""
#---------------------------------------------------------------------------------------------------
import os
import gdal, ogr, osr
import numpy as np
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
import osgeo.gdalnumeric as gdn
from sklearn.cluster import KMeans
import seaborn as sns; sns.set(style = "ticks", color_codes = True)
import pandas as pd


# defining functions
def return_tif_as_array(tif_path):
    # NOTE: Use this function instead of np.array(gdal.Open(tif_path).GetRasterBand(1).ReadAsArray())
    # to avoid session crashing or segmention fault.    
    a = gdal.Open(tif_path)
    b = a.GetRasterBand(1)
    c = b.ReadAsArray()
    d = np.array(c)
    return d


def stack_bands(bands_dir, normalize = False):  
    """
    Returns a 3D numpy array of all the tifs in the 'bands_dir'.
    Input:
        bands_dir: {string} path of the directory containing tif images.
        normalize: {boolean} whether to perform feature scaling.
    Output:
        band_np: {numpy array} 3D array with all tifs bands stacked depthwise.
    """
    i = 0
    for band in os.listdir(bands_dir):
        if band.endswith('.tif') or band.endswith('.TIF'):
            temp = return_tif_as_array(os.path.join(bands_dir, band))
            if i == 0:                
                band_np = np.empty((temp.shape[0], temp.shape[1]))
            if normalize:
                    temp = temp/(np.max(temp) - np.min(temp))
            band_np = np.dstack((band_np, temp))        
        i += 1    
    band_np = band_np[:,:,1:]        
    print('Shape of the stacked array is ', band_np.shape)
    return band_np


def save_array_as_geotif(array, source_tif_path, out_path):  
    """
    Generates a geotiff raster from the input numpy array (height * width * depth)
    Input:
        array: {numpy array} numpy array to be saved as geotiff
        source_tif_path: {string} path to the geotiff from which projection and geotransformation information will be extracted.
    Output:
        out_path: {string} path to the generated Geotiff raster        
    """
    if len(array.shape) > 2:
        height, width, depth = array.shape
    else:
        height, width = array.shape
        depth = 1
    source_tif = gdal.Open(source_tif_path)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(out_path, width, height, depth, gdal.GDT_Float32)
    if depth != 1:
        for i in range(depth):
            dataset.GetRasterBand(i+1).WriteArray(array[:,:,i])
    else:
        dataset.GetRasterBand(1).WriteArray(array)
    geotrans = source_tif.GetGeoTransform()  
    proj = source_tif.GetProjection()     
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset = None

def perform_pca(input_array):
    """
    Performs Principal Component Analysis and returns the principal components and explained variance.
    """
    pca = PCA(n_components = 'mle', svd_solver = 'full')
    principalComponents = pca.fit_transform(input_array)
    ev=pca.explained_variance_ratio_
    return principalComponents, ev

def visualize_pca_variance_ratio(ev):
    # Visualization of explained variance ratio
    plt.plot(np.cumsum(ev))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

def max_min(tif_path):
    """
    Returns the max and min DN value present in the input tif_path.
    """
    np_array = return_tif_as_array(tif_path)
    max_value = np.max(np_array)
    min_value = np.min(np_array)
    print('max_value ', max_value)
    print('min_value ', min_value)
    return max_value, min_value


def img_to_array(tif_path, dtype = 'float32'):   
    """
    Returns numpy array of the input tif_path.
    """     
    file  = gdal.Open(tif_path)
    bands = [file.GetRasterBand(i) for i in range(1, file.RasterCount + 1)]
    arr = np.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype)    
    arr = np.transpose(arr, [1, 2, 0])  # Reordering dimensions, so that channels are last
    return arr

def cluster(img_array, n_samples=20000, n_clusters=6):
    print('img array shape ', img_array.shape)      
    band_names = []
    for i in range(img_array.shape[-1]):
        band_names.append(str(i))
    # sample random subset of images
    imagesamples = []
    for i in range(n_samples):
        xr = np.random.randint(0, img_array.shape[1]-1)
        yr = np.random.randint(0, img_array.shape[0]-1)
        imagesamples.append(img_array[yr,xr,:])
    # convert to pandas dataframe
    imagessamplesDF=pd.DataFrame(imagesamples, columns = band_names)
    # make pairs plot (each band vs. each band)
    # seaborn_params_p = {'alpha': 0.15, 's': 20, 'edgecolor': 'k'}
    #pp1=sns.pairplot(imagessamplesDF, plot_kws = seaborn_params_p)#, hist_kws=seaborn_params_h)

    # fit kmeans to samples:
    print('clustering now')
    KMmodel = KMeans(n_clusters = n_clusters) 
    KMmodel.fit(imagessamplesDF)
    KM_train = list(KMmodel.predict(imagessamplesDF)) 
    i=0
    for k in KM_train:
        KM_train[i] = str(k) 
        i=i+1
    imagessamplesDF2 = imagessamplesDF
    imagessamplesDF2['group'] = KM_train
    # pair plots with clusters coloured:
    # pp2=sns.pairplot(imagessamplesDF,vars=band_names, hue='group',plot_kws = seaborn_params_p)
    # pp2._legend.remove()
    #  make the clustered image
    imageclustered=np.empty((img_array.shape[0],img_array.shape[1]))
    i=0
    for row in img_array:
        temp = KMmodel.predict(row) #.astype(np.float32))
        imageclustered[i,:]=temp
        i=i+1
    return imageclustered

def visualize_cluster(image_array):   
    # plot the map of the clustered data
    print('plotting now')
    colour_map = 'terrain'
    plt.figure(figsize=(80,64))
    plt.imshow(image_array, cmap=colour_map)


def polygonize_raster(img_path, outShp):
    """
    Generates a shapefile (outShp) from the input image (img_path).
    """ 
    try:        
        sourceRaster = gdal.Open(img_path)
        band = sourceRaster.GetRasterBand(1)                
        driver = ogr.GetDriverByName("ESRI Shapefile")        
        # If shapefile already exist, delete it
        if os.path.exists(outShp):
            driver.DeleteDataSource(outShp)
        outDatasource = driver.CreateDataSource(outShp)
        # get proj from raster
        srs = osr.SpatialReference()
        srs.ImportFromWkt(sourceRaster.GetProjectionRef())
        # create layer with proj
        outLayer = outDatasource.CreateLayer(outShp, srs)
        # Add class column to shapefile
        newField = ogr.FieldDefn('DN', ogr.OFTInteger)
        outLayer.CreateField(newField)
        gdal.Polygonize(band, None, outLayer, 0, [], callback=None)
        outDatasource.Destroy()
        sourceRaster = None        
    except Exception as e:
        print('gdal Polygonize Error: ' + str(e))

def simplify_geom(shp_path, tolerance_value = 10): 
    """
    Simplifies geometries of the shapefile (shp_path) while preserving topology using 'tolerance_value'.
    """ 
    shp_file = ogr.Open(shp_path, update=1)    
    lyr = shp_file.GetLayerByIndex(0)    
    i = 0                        
    while i < len(lyr):        
        j = lyr[i].Clone()        
        geom = j.GetGeometryRef()        
        j.SetGeometry(geom.SimplifyPreserveTopology(tolerance_value))                
        lyr.SetFeature(j)
        i += 1
    shp_file.Destroy()    

def add_area_field(shp_path):
    """
    Adds a field named 'Area' to the attribute table of 'shp_path' populated by areas corresponding
    to each geometry.
    """ 
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp_path, 1)
    layer = dataSource.GetLayer()
    new_field = ogr.FieldDefn("Area", ogr.OFTReal)
    new_field.SetWidth(32)
    new_field.SetPrecision(8)
    layer.CreateField(new_field)
    for feature in layer:                
        area = feature.GetGeometryRef().Area()                    
        feature.SetField("Area", area)
        layer.SetFeature(feature)
    dataSource = None 

def split_shp(in_shp_path, new_shp_path, DN_value = 2):
    """
    Generates a new shapefile ('new_shp_path') from the 'in_shp_path' consiting of geometries 
    where the DN value is equal to 'DN_value'.
    """ 
    cmd = 'ogr2ogr -f "ESRI Shapefile" -where "DN={}" {} {}'.format(DN_value, new_shp_path, shp_path)
    os.system(cmd)

def read_colab_error_logs():
    import json
    with open("/var/log/colab-jupyter.log", "r") as fo:
      for line in fo:
        print(json.loads(line)['msg'])


# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":    
    
    # input paths
    bands_dir = ''                  # directory containing all the bands tifs
    source_tif_path = ''            # path to one of the band tifs. Used to extract geotransformation information

    # out paths
    stacked_tif_path = ''           # path where the stacked tiff will be generated.
    cluster_raster_path = ''        # path where clustered tiff will be generated.
    cluster_shp_path = ''           # path where the clustered shapefile will be generated.
    
    print('stacking')
    bands_np_array = stack_bands(bands_dir, normalize = True)
    save_array_as_geotif(bands_np_array, source_tif_path, stacked_tif_path)
    # print('stacked array shape ', bands_np_array.shape)
    
    # clustering        
    print('clustering')
    cluster_array = cluster(bands_np_array, n_clusters = 5)
    # visualize_cluster(cluster_array)
    print('saving cluster')
    save_array_as_geotif(cluster_array, source_tif_path, out_path)        

    # converting clustered raster to shapefile   
    cluster_array = return_tif_as_array(cluster_raster_path)
    num_classes = len(np.unique(cluster_array))
    print('classes ', num_classes)
    polygonize_raster(cluster_raster_path, cluster_shp_path)
    for i in range(num_classes):        
        new_shp_path = cluster_shp_path.replace('.shp', '_{}.shp'.format(i))
        split_shp(cluster_shp_path, new_shp_path, i)
        simplify_geom(new_shp_path, tolerance_value = 30)
        add_area_field(new_shp_path)    
