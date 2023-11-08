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
from matplotlib.colors import ListedColormap
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
fh = np.load('chloc4so_19972021_15km.npz', allow_pickle=True)
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
chl[chl > 50] = np.nan
### Calculate monthly means
for i in np.arange(1998, 2022):
    yeartemp_nov = chl[:,:, (time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = chl[:,:, (time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = chl[:,:, (time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl[:,:, (time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)),2)
    if i == 1998:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
#%% Calculate linear trend for each pixel
yearchar = np.arange(1998, 2022)
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        yearchar = np.arange(1998, 2022)
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if np.isnan(chl_summerpixel).any():
            continue
        # remove nans if there exist any
        if len(np.isnan(chl_summerpixel)) > 0:
            nans_chl_summerpixel = np.isnan(chl_summerpixel)
            chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
            yearchar = yearchar[~nans_chl_summerpixel]
        # calculate linear regression
        slope, _, rvalue, pvalue, _ = stats.linregress(yearchar, chl_summerpixel)
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020[i,j] = slope
        if pvalue < 0.05:
            chl_summertrend19992020_significant[i,j] = slope
#%%
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-.1, vmax=.1,
                    cmap=cmocean.cm.balance)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1,
                    fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\trends\\summertrend_15km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020_significant[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-.1, vmax=.1,
                    cmap=cmocean.cm.balance)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1,
                    fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\trends\\summertrendsignificant_15km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
xx, yy = np.meshgrid(lon, lat)
# keep only the ones significant
xx[chl_summertrend19992020_significant != 1] = np.nan
yy[chl_summertrend19992020_significant != 1] = np.nan
lon_significant = xx[~np.isnan(xx)]
lat_significant = yy[~np.isnan(yy)]

plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-.1, vmax=.1,
                    cmap=cmocean.cm.balance)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1,
                    fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
pcol = map.pcolormesh(lon, lat, chl_summertrend19992020_significant[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-.1, vmax=.1,
                    cmap=cmocean.cm.balance, alpha=0.3, linewidth=0, edgecolor=None)
pcol.set_edgecolor('face')
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\trends\\summertrend_15km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
