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
import scipy

def regrid(data, out_x, out_y):
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))

    return interpolating_function((xv, yv))
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\oc4so-chl\\')
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\winds\\')
### Load data 1998-2020
fh = np.load('winds_19972019.npz', allow_pickle=True)
lat = fh['lat']
#lat = np.flip(lat)
lon = fh['lon']
northward_wind = fh['northward_wind']
#northward_wind = np.flip(northward_wind)
eastward_wind = fh['eastward_wind']
#eastward_wind = np.flip(eastward_wind)
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month

# wind speed
northward_wind_clim = np.nanmean(northward_wind, 2)
eastward_wind_clim = np.nanmean(eastward_wind, 2)
windspeed_clim = np.sqrt(northward_wind_clim**2 + eastward_wind_clim**2)

#a = np.nanmean()
#a[np.isnan(a)] = -999
#%% Interpolate
lat_10km = np.arange(-48, -70, -.1)
lon_10km = np.arange(-73, -25, .1)
#scipy.interpolate.interp2d(lon_10km, lon_10km, a, kind='linear', copy=True, bounds_error=False, fill_value=None)

f = scipy.interpolate.interp2d(lon, lat, a, kind='linear')
Z2 = f(lon_10km, lat_10km)

#northward_wind[:,:,0]

#zi = interpolate.griddata(northward_wind[:,:,0], z, xnew, method='linear')

plt.figure()
plt.pcolormesh(lon_10km, lat_10km, Z2, cmap=plt.cm.jet)

#%%
Z2[Z2<-10] = np.nan
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon_10km, lat_10km, Z2[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.jet)
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\winds_test.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, northward_wind[:,:,0][:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.jet)
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\winds_test.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()








#%%
from scipy.interpolate import griddata
lat_10km = np.arange(-48, -70, -.1)
lon_10km = np.arange(-73, -25, .1)
grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')

#%%
# Original lat x lon x chl (4x4km)
lat_4km = lat
lon_4km = lon
northward_wind_4km = northward_wind
# 10 by 10 km
lat_10km = np.arange(-48, -70, -.1)
lon_10km = np.arange(-73, -25, .1)
northward_wind_10km = np.empty((len(lat_10km), len(lon_10km), len(time_date)))*np.nan
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
        northward_wind_10km[i, j, :] = np.nanmean(northward_wind[np.ix_(lat_indices, lon_indices)], (0,1))
#northward_wind_10km_clim = np.nanmean(northward_wind_10km, 2)
# Original lat x lon x chl (4x4km)
lat_4km = lat
lon_4km = lon
northward_wind_4km = northward_wind
# 10 by 10 km
lat_10km = np.arange(-48, -70, -.1)
lon_10km = np.arange(-73, -25, .1)
eastward_wind_10km = np.empty((len(lat_10km), len(lon_10km), len(time_date)))*np.nan
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
        eastward_wind_10km[i, j, :] = np.nanmean(eastward_wind[np.ix_(lat_indices, lon_indices)], (0,1))
#northward_wind_10km_clim = np.nanmean(northward_wind_10km, 2)
# Save datasets with lower resolution
np.savez_compressed('winds_19972019_10km', northward_wind=northward_wind_10km,eastward_wind=eastward_wind_10km,
                    time_date=time_date,
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
