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
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\par\\')
### Load data 1998-2020
fh = np.load('par_19972021_new.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
par = fh['par']
par = par.astype(dtype=np.float64)
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
#par = par*100
### Calculate monthly means
for i in np.arange(1998, 2022):
    yeartemp_sep = par[:,:, (time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = par[:,:, (time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = par[:,:, (time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = par[:,:, (time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = par[:,:, (time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = par[:,:, (time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = par[:,:, (time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = par[:,:, (time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr)),2)
    if i == 1998:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
#%%
par_clim = np.nanmean(par,2)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-63))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, par_clim[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat',
                    cmap=plt.cm.jet, zorder=0)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()











#%% Calculate linear trend for each pixel
yearchar = np.arange(1998, 2022)
par_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
par_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        yearchar = np.arange(1998, 2022)
        par_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if np.isnan(par_summerpixel).any():
            continue
        # remove nans if there exist any
        #if len(np.isnan(par_summerpixel)) > 0:
        #    nans_par_summerpixel = np.isnan(par_summerpixel)
        #    par_summerpixel = par_summerpixel[~nans_par_summerpixel]
        #    yearchar = yearchar[~nans_par_summerpixel]
        # calculate linear regression
        slope, _, rvalue, pvalue, _ = stats.linregress(yearchar, par_summerpixel)
        # add slope value (i.e. change in par per year) to pixel
        par_summertrend19992020[i,j] = slope
        if pvalue < 0.05:
            par_summertrend19992020_significant[i,j] = slope
#%%
xx, yy = np.meshgrid(lon, lat)
xx = xx[~np.isnan(par_summertrend19992020_significant)]
yy = yy[~np.isnan(par_summertrend19992020_significant)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, par_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-.5, vmax=.5,
                    cmap=cmocean.cm.balance, zorder=0)
map.contourf(
    lon, lat, par_summertrend19992020_significant,
    transform=ccrs.PlateCarree(),
    colors='none',
    hatches=[3*'.',3*'.'],
)
#map.contour(lon, lat, par_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
#map.scatter(xx, yy, 1, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none')
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('PAR (units)', fontsize=14)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\par_trendmap_SEPAPR.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, par_summertrend19992020_significant[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('PAR (units)', fontsize=14)
#map.contour(lon, lat, par_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\par_trendmap_SEPAPR_onlysignificant.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()