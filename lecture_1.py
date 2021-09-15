# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:26:30 2020

@author: user
"""



import pandas as pd
from PIL import Image
import numpy as np
#import shapefile

pd.options.display.max_rows = 9
pd.options.display.max_columns = 9

path = 'C:/Users/user/Documents/B.A. Governance Sem.6/Heidelberg Master/machine learning geo/1/'


''' csv '''
#worldcities = pd.read_csv(path+'worldcities.csv')
#
#print(worldcities)


''' tiff '''
#tiff = Image.open(path+'Westeros_dtm.tif')
#tiff.show()
#
#tiff_array = np.array(tiff)
#
#print(tiff_array)



''' npy '''
#npy = np.load(path+'population_germany.npy')
#print(npy)
#print(pd.DataFrame(npy).T)
#pd.DataFrame(npy).T.set_index(0).plot()



''' shx NEEDS ENVIRONMENT OR PIP INSTALL '''

#shx = shapefile.Reader(path+'Locations.shx')
#first feature of the shapefile
#feature = shape.shapeRecords()[0]
#first = feature.shape.__geo_interface__  
#print first # (GeoJSON format)
#{'type': 'LineString', 'coordinates': ((0.0, 0.0), (25.0, 10.0), (50.0, 50.0))}
#



'''  dbf '''









import importlib
def imp_and_version(package):
    try:    
        p = importlib.import_module(package)
        v = "(None)"
        try:
            v = p.__version__
        except Exception as e:
            pass
        print(package, v)
    except Exception as e:
        print(package, "FAILED:", e)


for p in ['numpy', 'scipy', 'pandas', 'sklearn', 'skimage',
'tensorflow', 'torch', 'gdal', 'matplotlib',
'matplotlib.pyplot','PyQt5.QtCore']:
    imp_and_version(p)


import joblib
from sklearn import datasets
california_housing = datasets.fetch_california_housing()
print(california_housing)
lat = california_housing['data'][:, 6]
lon = california_housing['data'][:, 7]
print(lat,lon)




import matplotlib.pyplot as plt
plt.scatter(lon, lat, c=california_housing['target'], s=4)
plt.axis('equal')
plt.colorbar()
plt.show()

#todo: restlichen files parsen, scatter befehl und dataset versthen
























































