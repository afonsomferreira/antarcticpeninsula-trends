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
fh = np.load('chloc4so_19972021_yearlyaveragecycles_10km.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
chl_averagecycle = fh['chl_averagecycle']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
#%%
# Calculate metrics for each pixel
month_chl_max = np.empty((len(lat), len(lon)))*np.nan
month_chl_startafterwinter = np.empty((len(lat), len(lon)))*np.nan
month_chl_stopafterspring = np.empty((len(lat), len(lon)))*np.nan
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        if np.count_nonzero(np.isnan(chl_averagecycle[i,j,:])) > 40:
            continue
        # find month where chl reaches its max
        startDate = datetime.datetime(year=1999, month=1, day=1)
        week_chl_max = np.nanargmax(chl_averagecycle[i,j,:]) + 1
        weeksToShift = week_chl_max - 1
        endDate = startDate + datetime.timedelta(weeks=int(weeksToShift))
        month_chl_max_temp = endDate.month
        month_chl_max[i,j] = month_chl_max_temp
        # find month where phytoplankton begins to grow (post winter)
        try:
            weekfirstchlafterwinter = next(i for i,v in enumerate(chl_averagecycle[i,j,:][26:]) if v > 0) + 27
        except:
            weekfirstchlafterwinter = next(i for i,v in enumerate(chl_averagecycle[i,j,:]) if v > 0)
        endDate = startDate + datetime.timedelta(weeks=int(weekfirstchlafterwinter))
        month_chl_start_temp = endDate.month
        month_chl_startafterwinter[i,j] = month_chl_start_temp  
        # find month where phytoplankton stops growing (after spring)
        try:
            weeklastchl = 26 - next(i for i,v in enumerate(np.flip(chl_averagecycle[i,j,:])[26:]) if v > 0)
        except:
            continue
        endDate = startDate + datetime.timedelta(weeks=int(weeklastchl))
        month_chl_end_temp = endDate.month
        month_chl_stopafterspring[i,j] = month_chl_end_temp     
#%% Plot maps
## Max Chl
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, month_chl_max[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.twilight_shifted,
                    vmin=1, vmax=12)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=np.arange(1,13),
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=14)
cbar.set_label('Chl-a Max Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\phenology_10km\\maxchlday_weeklydata.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## Start Chl
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, month_chl_startafterwinter[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.twilight_shifted,
                    vmin=1, vmax=12)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=np.arange(1,13),
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=14)
cbar.set_label('Chl-a Start Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\phenology_10km\\startchlday_weeklydata.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## Final Chl
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, month_chl_stopafterspring[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.twilight_shifted,
                    vmin=1, vmax=12)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=np.arange(1,13),
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=14)
cbar.set_label('Chl-a Final Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\phenology_10km\\endchlday_weeklydata.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
