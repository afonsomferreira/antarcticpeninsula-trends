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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\rrs\\')
### Load data 1998-2020
fh = np.load('chloc4so_8day.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
chl = fh['chl']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Load data 2012-13
fh = np.load('chloc4so_8day_20122013.npz', allow_pickle=True)
chl_20122013 = fh['chl']
# Replace 2012-2013 in original chl
chl[:,:, 644:736] = chl_20122013
chl[chl > 100] = 100
### Calculate summer means (December-February) for each year
for i in np.arange(1999, 2021):
    yeartemp_dec = chl[:,:, (time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = chl[:,:, (time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl[:,:, (time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)),2)
    if i == 1999:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
### Calculate linear trend for each pixel
yearchar = np.arange(1999, 2021)
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        yearchar = np.arange(1999, 2021)
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if len(chl_summerpixel[np.isnan(chl_summerpixel)]) > 15:
            continue
        # remove nans if there exist any
        if len(np.isnan(chl_summerpixel)) > 0:
            nans_chl_summerpixel = np.isnan(chl_summerpixel)
            chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
            yearchar = yearchar[~nans_chl_summerpixel]
        # calculate linear regression
        slope, _, rvalue, pvalue, _ = stats.linregress(yearchar, chl_summerpixel)
        # add slope value (i.e. change in chl per year) to pixel
        if pvalue < 0.05:
            chl_summertrend19992020[i,j] = slope
        
# Plot Climatology
chl_summertrend19992020[chl_summertrend19992020>1] = np.nan
chl_summertrend19992020[chl_summertrend19992020<-1] = np.nan
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020, transform=ccrs.PlateCarree(), shading='flat', cmap=cmocean.cm.balance,
                    vmin=-.1, vmax=.1)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Yearly increase in Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\summertrend_19992020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()