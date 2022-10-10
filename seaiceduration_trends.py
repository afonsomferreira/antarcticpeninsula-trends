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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
### Load data 1998-2020
fh = np.load('sst_19972021.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
#sst = fh['sst']
seaice = fh['seaice']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
seaice = seaice * 100
# Correct values
#chl[chl > 50] = np.nan
#%%
### Calculate sea ice duration
seaice_duration = np.empty((np.size(lat), np.size(lon), np.size(np.arange(1998, 2021))))*np.nan
seaice_init = np.empty((np.size(lat), np.size(lon), np.size(np.arange(1998, 2021))))*np.nan
seaice_term = np.empty((np.size(lat), np.size(lon), np.size(np.arange(1998, 2021))))*np.nan

for i, item in enumerate(np.arange(1998, 2021)):
    print(item)
    for k in range(0, len(lat)):
        for j in range(0, len(lon)):
            yeartemp_febdec = seaice[k, j, (time_date_years == item) & ((time_date_months == 2) | (time_date_months == 3) |
                                                                     (time_date_months == 4) | (time_date_months == 5) | (time_date_months == 6) | (time_date_months == 7) |
                                                                     (time_date_months == 8) | (time_date_months == 9) | (time_date_months == 10) |
                                                                     (time_date_months == 11) | (time_date_months == 12)
                                                                     )]
            yeartemp_janafter = seaice[k, j, (time_date_years == item+1) & (time_date_months == 1)]
            yeartemp_febapr = np.hstack((yeartemp_febdec, yeartemp_janafter))
            yeartemp_timefebdec = time_date[(time_date_years == item) & ((time_date_months == 2) | (time_date_months == 3) |
                                                                     (time_date_months == 4) | (time_date_months == 5) | (time_date_months == 6) | (time_date_months == 7) |
                                                                     (time_date_months == 8) | (time_date_months == 9) | (time_date_months == 10) |
                                                                     (time_date_months == 11) | (time_date_months == 12)
                                                                     )]
            yeartemp_timejanafter = time_date[(time_date_years == item+1) & (time_date_months == 1)]
            yeartemp_time = np.hstack((yeartemp_timefebdec, yeartemp_timejanafter))
            # Check if there is sea ice throughout the year
            if (any(x > 15 for x in yeartemp_febapr) == False) | (np.size(yeartemp_febapr[yeartemp_febapr > 15]) < 5):
                seaice_duration[k, j, i] = 0
                continue
            if all(x > 15 for x in yeartemp_febapr) == True:
                seaice_duration[k, j, i] = 365
                continue
            # Check sea ice initiation/advance (>15%)
            ice_index = yeartemp_febapr > 15
            breaker_a = 0
            for l, ind in enumerate(ice_index):
                if ind == False:
                    continue
                elif (breaker_a == 0) & (all(x == True for x in ice_index[l:l+5])):
                    seaice_init_index = l
                    seaice_init_date = yeartemp_time[l]
                    breaker_a=1
            # Check final/retraction of  sea ice (>15%)
            ice_index_fin = ice_index[::-1]
            breaker_b = 0
            for l, ind in enumerate(ice_index_fin):
                if ind == False:
                    continue
                elif (breaker_b == 0) & (all(x == True for x in ice_index_fin[l:l+5])):
                    seaice_fin_index = l
                    seaice_fin_date = yeartemp_time[-l]
                    breaker_b=1
            # Calculate duration of sea ice
            if breaker_a == 1 & breaker_b == 1:
                seaiceduration_temp = (seaice_fin_date - seaice_init_date).days
                seaice_duration[k, j, i] = seaiceduration_temp
                seaice_init[k, j, i] = seaice_init_date.timetuple().tm_yday
                seaice_term[k, j, i] = seaice_fin_date .timetuple().tm_yday           


#%% Calculate linear trend for each pixel
yearchar = np.arange(1998, 2022)
seaice_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
seaice_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        yearchar = np.arange(1998, 2022)
        seaice_summerpixel = years_seaice_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if np.isnan(seaice_summerpixel).any():
            continue
        # remove nans if there exist any
        #if len(np.isnan(seaice_summerpixel)) > 0:
        #    nans_seaice_summerpixel = np.isnan(seaice_summerpixel)
        #    seaice_summerpixel = seaice_summerpixel[~nans_seaice_summerpixel]
        #    yearchar = yearchar[~nans_seaice_summerpixel]
        # calculate linear regression
        slope, _, rvalue, pvalue, _ = stats.linregress(yearchar, seaice_summerpixel)
        # add slope value (i.e. change in seaice per year) to pixel
        seaice_summertrend19992020[i,j] = slope
        if pvalue < 0.05:
            seaice_summertrend19992020_significant[i,j] = slope
#%%
#fig, ax = plt.subplots(figsize=(6, 6))
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, seaice_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance)
map.pcolor(lon, lat, seaice_summertrend19992020_significant[:-1, :-1], hatch='...', alpha=0., transform=ccrs.PlateCarree(), shading='flat')
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1,
                    fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Change in Nov-Feb Sea Ice (%)', fontsize=16)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\seaice\\trends\\summertrend_4km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
