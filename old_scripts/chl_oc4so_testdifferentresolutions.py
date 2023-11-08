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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
### Load data 1998-2020
fh = np.load('chloc4so_19972021.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
chl = fh['chl']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
#chl[chl > 100] = 100
#%%
# Original lat x lon x chl (4x4km)
lat_4km = lat
lon_4km = lon
chl_4km = chl
# 10 by 10 km
lat_10km = np.arange(-48, -70, -.1)
lon_10km = np.arange(-73, -25, .1)
chl_10km = np.empty((len(lat_10km), len(lon_10km), len(time_date)))*np.nan
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
        chl_10km[i, j, :] = np.nanmean(chl[np.ix_(lat_indices, lon_indices)], (0,1))
# 15 by 15 km
lat_15km = np.arange(-48, -70, -.15)
lon_15km = np.arange(-73, -25, .15)
chl_15km = np.empty((len(lat_15km), len(lon_15km), len(time_date)))*np.nan
for i in tqdm(range(0, len(lat_15km))):
    for j in range(0, len(lon_15km)):
        if (i == len(lat_15km)-1) & (j != len(lon_15km)-1):
            lat_indices = (lat < lat_15km[i])
            lon_indices = (lon > lon_15km[j]) & (lon < lon_15km[j+1])
        elif (i == len(lat_15km)-1) & (j == len(lon_15km)-1):
            lat_indices = (lat < lat_15km[i])
            lon_indices = (lon > lon_15km[j])                
        elif (i != len(lat_15km)-1) & (j == len(lon_15km)-1):
            lat_indices = (lat < lat_15km[i]) & (lat > lat_15km[i+1])
            lon_indices = (lon > lon_15km[j])
        else:                  
            lat_indices = (lat < lat_15km[i]) & (lat > lat_15km[i+1])
            lon_indices = (lon > lon_15km[j]) & (lon < lon_15km[j+1])
        chl_15km[i, j, :] = np.nanmean(chl[np.ix_(lat_indices, lon_indices)], (0,1))
# 20 by 20 km
lat_20km = np.arange(-48, -70, -.2)
lon_20km = np.arange(-73, -25, .2)
chl_20km = np.empty((len(lat_20km), len(lon_20km), len(time_date)))*np.nan
for i in tqdm(range(0, len(lat_20km))):
    for j in range(0, len(lon_20km)):
        if (i == len(lat_20km)-1) & (j != len(lon_20km)-1):
            lat_indices = (lat < lat_20km[i])
            lon_indices = (lon > lon_20km[j]) & (lon < lon_20km[j+1])
        elif (i == len(lat_20km)-1) & (j == len(lon_20km)-1):
            lat_indices = (lat < lat_20km[i])
            lon_indices = (lon > lon_20km[j])                
        elif (i != len(lat_20km)-1) & (j == len(lon_20km)-1):
            lat_indices = (lat < lat_20km[i]) & (lat > lat_20km[i+1])
            lon_indices = (lon > lon_20km[j])
        else:                  
            lat_indices = (lat < lat_20km[i]) & (lat > lat_20km[i+1])
            lon_indices = (lon > lon_20km[j]) & (lon < lon_20km[j+1])
        chl_20km[i, j, :] = np.nanmean(chl[np.ix_(lat_indices, lon_indices)], (0,1))
# 25 by 25 km
lat_25km = np.arange(-48, -70, -.25)
lon_25km = np.arange(-73, -25, .25)
chl_25km = np.empty((len(lat_25km), len(lon_25km), len(time_date)))*np.nan
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
        chl_25km[i, j, :] = np.nanmean(chl[np.ix_(lat_indices, lon_indices)], (0,1))
chl_10km_clim = np.nanmean(chl_10km, 2)
chl_15km_clim = np.nanmean(chl_15km, 2)
chl_20km_clim = np.nanmean(chl_20km, 2)
chl_25km_clim = np.nanmean(chl_25km, 2)
# Save datasets with lower resolution
np.savez_compressed('chloc4so_19972021_10km', chl=chl_10km,time_date=time_date,
                    lat=lat_10km, lon=lon_10km)
np.savez_compressed('chloc4so_19972021_15km', chl=chl_15km,time_date=time_date,
                    lat=lat_15km, lon=lon_15km)
np.savez_compressed('chloc4so_19972021_20km', chl=chl_20km,time_date=time_date,
                    lat=lat_20km, lon=lon_20km)
np.savez_compressed('chloc4so_19972021_25km', chl=chl_25km,time_date=time_date,
                    lat=lat_25km, lon=lon_25km)
#%% Plot 10km
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon_10km, lat_10km, np.log10(chl_10km_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
                    vmax=np.log10(10), cmap=plt.cm.viridis)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\climatology19972021_10km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# 15 km
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon_15km, lat_15km, np.log10(chl_15km_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
                    vmax=np.log10(10), cmap=plt.cm.viridis)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\climatology19972021_15km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# 20 km
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon_20km, lat_20km, np.log10(chl_20km_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
                    vmax=np.log10(10), cmap=plt.cm.viridis)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\climatology19972021_20km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# 25 km
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon_25km, lat_25km, np.log10(chl_25km_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
                    vmax=np.log10(10), cmap=plt.cm.viridis)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\climatology19972021_25km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()