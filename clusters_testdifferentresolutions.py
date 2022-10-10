# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:33:06 2020

@author: Afonso
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from matplotlib.patches import Polygon
from matplotlib.path import Path
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from netCDF4 import Dataset
import datetime
import cmocean
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\oc4so-chl\\')
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\winds\\')
fh = np.load('winds_19972019_daily.npz', allow_pickle=True)
lat_25km = fh['lat']
lon_25km = fh['lon']

os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('antarcticpeninsula_cluster.npz',allow_pickle = True)
clusters = fh['clusters']
lat = fh['lat']
lon = fh['lon']
#%%
# Original lat x lon x chl (4x4km)
#lat_10km = lat
#lon_10km = lon
#clusters_4km = clusters
# 10 by 10 km
#lat_25km = np.arange(-48, -70, -.1)
#lon_25km = np.arange(-73, -25, .1)
clusters_25km = np.empty((len(lat_25km), len(lon_25km)))*np.nan
for i in tqdm(range(0, len(lat_25km))):
    for j in range(0, len(lon_25km)):
        if (i == len(lat_25km)-1) & (j != len(lon_25km)-1):
            lat_indices = (lat < lat_25km[i])
            lon_indices = (lon > lon_25km[j]) & (lon < lon_25km[j+1])
        elif (i == len(lat_25km)-1) & (j == len(lon_25km)-1):
            lat_indices = (lat < lat_25km[i])
            lon_indices = (lon > lon_25km[j])                
        elif (i != len(lat_25km)-1) & (j == len(lon_25km)-1):
            lat_indices = (lat < lat_25km[i]) & (lat > lat_25km[i+1])
            lon_indices = (lon > lon_25km[j])
        else:                  
            lat_indices = (lat < lat_25km[i]) & (lat > lat_25km[i+1])
            lon_indices = (lon > lon_25km[j]) & (lon < lon_25km[j+1])
        clusters_25km[i, j] = np.nanmean(clusters[np.ix_(lat_indices, lon_indices)], (0,1))
#clusters_10km_clim = np.nanmean(clusters_10km, 2)
# Save datasets with lower resolution
np.savez_compressed('clusters_25km', clusters=clusters_25km,
                    lat=lat_25km, lon=lon_25km)
#%% Plot 10km
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon_25km, lat_25km, clusters_25km[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.jet)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1,
                    fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('SST 10km', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\sst\\climatology19972021_10km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
