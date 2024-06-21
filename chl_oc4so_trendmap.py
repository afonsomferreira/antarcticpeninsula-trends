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
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\cciv6data\\')
### Load data 1998-2020
fh = np.load('chloc4so_19972022.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
chl = fh['chl_oc4so']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
chl[chl > 100] = 100
### Calculate monthly means
for i in np.arange(1998, 2023):
    yeartemp_sep = chl[:,:, (time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = chl[:,:, (time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = chl[:,:, (time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = chl[:,:, (time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = chl[:,:, (time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl[:,:, (time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = chl[:,:, (time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = chl[:,:, (time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr)),2)
    if i == 1998:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
#%% Calculate linear trend for each pixel
yearchar = np.arange(1998, 2023)
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        yearchar = np.arange(1998, 2023)
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
#        if np.isnan(chl_summerpixel).any():
#            continue
        # remove nans if there exist any
        if sum(np.isnan(chl_summerpixel)) < 2:
            nans_chl_summerpixel = np.isnan(chl_summerpixel)
            chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
            yearchar = yearchar[~nans_chl_summerpixel]
        else:
            continue
        # calculate linear regression
        slope, _, rvalue, pvalue, _ = stats.linregress(yearchar, chl_summerpixel)
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020[i,j] = slope
        if pvalue < 0.05:
            chl_summertrend19992020_significant[i,j] = slope
#%%
xx, yy = np.meshgrid(lon, lat)
xx = xx[~np.isnan(chl_summertrend19992020_significant)]
yy = yy[~np.isnan(chl_summertrend19992020_significant)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summerpixel[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-.1, vmax=.1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
#map.scatter(xx, yy, 1, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none')
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)



def add_hatch(pixels):
    for x, y in pixels:
        plt.add_patch(plt.Rectangle((x, y), 5, 5, fill=None, edgecolor='black', hatch='//'))
add_hatch(chl_summertrend19992020_significant)

#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\chl_trendmap_SEPAPR_onlysignificant.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
### Calculate monthly means
for i in np.arange(1998, 2023):
    yeartemp_sep = chl[:,:, (time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = chl[:,:, (time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = chl[:,:, (time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = chl[:,:, (time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = chl[:,:, (time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl[:,:, (time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = chl[:,:, (time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = chl[:,:, (time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_mar,
                                                yeartemp_apr)),2)
    if i == 1998:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
#%% Calculate linear trend for each pixel
yearchar = np.arange(1998, 2023)
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        yearchar = np.arange(1998, 2023)
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
#        if np.isnan(chl_summerpixel).any():
#            continue
        # remove nans if there exist any
        if sum(np.isnan(chl_summerpixel)) < 2:
            nans_chl_summerpixel = np.isnan(chl_summerpixel)
            chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
            yearchar = yearchar[~nans_chl_summerpixel]
        else:
            continue
        # calculate linear regression
        slope, _, rvalue, pvalue, _ = stats.linregress(yearchar, chl_summerpixel)
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020[i,j] = slope
        if pvalue < 0.05:
            chl_summertrend19992020_significant[i,j] = slope
#%%
xx, yy = np.meshgrid(lon, lat)
xx = xx[~np.isnan(chl_summertrend19992020_significant)]
yy = yy[~np.isnan(chl_summertrend19992020_significant)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020_significant[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-.1, vmax=.1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
#map.scatter(xx, yy, 1, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none')
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
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
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\chl_trendmap_AUTUMN_onlysignificant.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%