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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
### Load data 1998-2020
fh = np.load('sst_19811996.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
#sst = fh['sst']
sst = fh['sst']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
#%%
# Original lat x lon x chl (4x4km)
lat_4km = lat
lon_4km = lon
#sst_4km = sst
# 10 by 10 km
lat_10km = np.arange(-48, -70, -.1)
lon_10km = np.arange(-73, -25, .1)
sst_10km = np.empty((len(lat_10km), len(lon_10km), len(time_date)))*np.nan
for i in tqdm(range(0, len(lat_10km))):
    for j in range(0, len(lon_10km)):
        if (i == len(lat_10km)-1) & (j != len(lon_10km)-1):
            lat_indices = (lat < lat_10km[i])
            lon_indices = (lon > lon_10km[j]) & (lon < lon_10km[j+1])
        elif (i == len(lat_10km)-1) & (j == len(lon_10km)-1):
            lat_indices = (lat < lat_10km[i])
            lon_indices = (lon > lon_10km[j])                
        elif (i != len(lat_10km)-1) & (j == len(lon_10km)-1):
            lat_indices = (lat < lat_10km[i]) & (lat > lat_10km[i+1])
            lon_indices = (lon > lon_10km[j])
        else:                  
            lat_indices = (lat < lat_10km[i]) & (lat > lat_10km[i+1])
            lon_indices = (lon > lon_10km[j]) & (lon < lon_10km[j+1])
        sst_10km[i, j, :] = np.nanmean(sst[np.ix_(lat_indices, lon_indices)], (0,1))
#sst_10km_clim = np.nanmean(sst_10km, 2)
# Save datasets with lower resolution
np.savez_compressed('sst_19811996_10km', sst=sst_10km,time_date=time_date,
                    lat=lat_10km, lon=lon_10km)
#%%
# Original lat x lon x chl (4x4km)
lat_4km = lat
lon_4km = lon
seaice_4km = seaice
# 10 by 10 km
lat_10km = np.arange(-48, -70, -.1)
lon_10km = np.arange(-73, -25, .1)
seaice_10km = np.empty((len(lat_10km), len(lon_10km), len(time_date)))*np.nan
for i in tqdm(range(0, len(lat_10km))):
    for j in range(0, len(lon_10km)):
        if (i == len(lat_10km)-1) & (j != len(lon_10km)-1):
            lat_indices = (lat < lat_10km[i])
            lon_indices = (lon > lon_10km[j]) & (lon < lon_10km[j+1])
        elif (i == len(lat_10km)-1) & (j == len(lon_10km)-1):
            lat_indices = (lat < lat_10km[i])
            lon_indices = (lon > lon_10km[j])                
        elif (i != len(lat_10km)-1) & (j == len(lon_10km)-1):
            lat_indices = (lat < lat_10km[i]) & (lat > lat_10km[i+1])
            lon_indices = (lon > lon_10km[j])
        else:                  
            lat_indices = (lat < lat_10km[i]) & (lat > lat_10km[i+1])
            lon_indices = (lon > lon_10km[j]) & (lon < lon_10km[j+1])
        seaice_10km[i, j, :] = np.nanmean(seaice[np.ix_(lat_indices, lon_indices)], (0,1))
#seaice_10km_clim = np.nanmean(seaice_10km, 2)
# Save datasets with lower resolution
np.savez_compressed('seaice_19972021_10km', seaice=seaice_10km,time_date=time_date,
                    lat=lat_10km, lon=lon_10km)

#%% Plot 10km
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon_10km, lat_10km, sst_10km_clim[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.jet)
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
