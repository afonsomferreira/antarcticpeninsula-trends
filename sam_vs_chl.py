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
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline
from netCDF4 import Dataset
from sktime.transformations.series.outlier_detection import HampelFilter
import datetime
import cmocean
import dtw as dtw
from scipy import integrate
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.nanmedian(points, axis=0)
    diff = np.nansum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.nanmedian(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
### Load data 1998-2022
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\cciv6data\\')
fh = np.load('chloc4so_19972022.npz', allow_pickle=True)
lat = fh['lat'][144:]
lon = fh['lon']
chl = fh['chl_oc4so'][144:,:,:]
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
chl[chl > 100] = 100
# Load original 10km clusters
#os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
#fh = np.load('antarcticpeninsula_newclusters_seaicebelow15.npz',allow_pickle = True)
#clusters = fh['clusters']
#lat_clusters = fh['lat']
#lon_clusters = fh['lon'] 
# Load upscaled 4km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_4km.npz',allow_pickle = True)
clusters = fh['clusters']
#lat_clusters = fh['lat']
#lon_clusters = fh['lon'] 
#%% Load SAM
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sam\\')
sam_pd = pd.read_csv('norm.daily.aao.cdas.z700.19790101_current.csv', sep=',')
sam_daily = sam_pd['aao_index_cdas'].values
time_date_years_sam = sam_pd['year'].values
time_date_months_sam = sam_pd['month'].values
time_date_days_sam = sam_pd['day'].values
# Average monthly
time_date_sam_daily = np.empty_like(time_date_days_sam, dtype=object)
for i in range(0, len(time_date_sam_daily)):
    time_date_sam_daily[i] = datetime.datetime(year = time_date_years_sam[i],
                                               month = time_date_months_sam[i],
                                               day = time_date_days_sam[i])
sam_pd_daily = pd.Series(data=sam_daily, index=time_date_sam_daily)
sam_pd_monthly = sam_pd_daily.resample('M').mean()
#%% Calculate monthly chl means
for i in np.arange(1998, 2023):
    yeartemp_dec = chl[:,:, (time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = chl[:,:, (time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl[:,:, (time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb)),2)
    if i == 1998:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
# No Lag
for i in np.arange(1998, 2023):
    yeartemp_sepdec = sam_pd_monthly.values[(sam_pd_monthly.index.year == i-1) & (sam_pd_monthly.index.month == 12)]
    yeartemp_janapr = sam_pd_monthly.values[(sam_pd_monthly.index.year == i) & ((sam_pd_monthly.index.month == 1) | (sam_pd_monthly.index.month == 2))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        sam_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        sam_SEPAPR = np.hstack((sam_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate correlation
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if np.isnan(chl_summerpixel).any():
            continue
        # remove nans if there exist any
        #if len(np.isnan(chl_summerpixel)) > 0:
        #    nans_chl_summerpixel = np.isnan(chl_summerpixel)
        #    chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
        #    yearchar = yearchar[~nans_chl_summerpixel]
        # calculate linear regression
        corr, pvalue = stats.spearmanr(chl_summerpixel, sam_SEPAPR, nan_policy='omit')
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020[i,j] = corr
        if pvalue < 0.05:
            chl_summertrend19992020_significant[i,j] = corr
#%%
xx, yy = np.meshgrid(lon, lat)
xx = xx[~np.isnan(chl_summertrend19992020_significant)]
yy = yy[~np.isnan(chl_summertrend19992020_significant)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
map.scatter(xx, yy, 2, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none')
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Correlation', fontsize=14)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\correlation_CHLSAM_DJF.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020_significant[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
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
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\correlation_CHLSAM_onlysignificant_DJF.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Now for SEP-APRIL
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
# SEP-APR
for i in np.arange(1998, 2023):
    yeartemp_sepdec = sam_daily[(time_date_years_sam == i-1) & ((time_date_months_sam == 9) | (time_date_months_sam == 10)
                                                                | (time_date_months_sam == 11) | (time_date_months_sam == 12))]
    yeartemp_janapr = sam_daily[(time_date_years_sam == i) & ((time_date_months_sam == 1) | (time_date_months_sam == 2)
                                                                | (time_date_months_sam == 3) | (time_date_months_sam == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        sam_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        sam_SEPAPR = np.hstack((sam_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate correlation
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if np.isnan(chl_summerpixel).any():
            continue
        # remove nans if there exist any
        #if len(np.isnan(chl_summerpixel)) > 0:
        #    nans_chl_summerpixel = np.isnan(chl_summerpixel)
        #    chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
        #    yearchar = yearchar[~nans_chl_summerpixel]
        # calculate linear regression
        corr, pvalue = stats.spearmanr(chl_summerpixel, sam_SEPAPR, nan_policy='omit')
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020[i,j] = corr
        if pvalue < 0.05:
            chl_summertrend19992020_significant[i,j] = corr
#%%
xx, yy = np.meshgrid(lon, lat)
xx = xx[~np.isnan(chl_summertrend19992020_significant)]
yy = yy[~np.isnan(chl_summertrend19992020_significant)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
map.scatter(xx, yy, 2, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none')
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Correlation', fontsize=14)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\correlation_CHLSAM_SEPAPR.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020_significant[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
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
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\correlation_CHLSAM_onlysignificant_SEPAPR.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Now for SON
for i in np.arange(1998, 2023):
    yeartemp_sep = chl[:,:, (time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = chl[:,:, (time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = chl[:,:, (time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)),2)
    if i == 1998:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
# SEP-APR
for i in np.arange(1998, 2023):
    yeartemp_sepdec = sam_daily[(time_date_years_sam == i-1) & ((time_date_months_sam == 9) | (time_date_months_sam == 10)
                                                                | (time_date_months_sam == 11))]
    if i == 1998:
        sam_SEPAPR = np.nanmean(yeartemp_sepdec)
    else:
        sam_SEPAPR = np.hstack((sam_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate correlation
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if sum(np.isnan(chl_summerpixel)).any():
            continue
        # remove nans if there exist any
        #if len(np.isnan(chl_summerpixel)) > 0:
        #    nans_chl_summerpixel = np.isnan(chl_summerpixel)
        #    chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
        #    yearchar = yearchar[~nans_chl_summerpixel]
        # calculate linear regression
        corr, pvalue = stats.spearmanr(chl_summerpixel, sam_SEPAPR, nan_policy='omit')
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020[i,j] = corr
        if pvalue < 0.05:
            chl_summertrend19992020_significant[i,j] = corr
#%%
xx, yy = np.meshgrid(lon, lat)
xx = xx[~np.isnan(chl_summertrend19992020_significant)]
yy = yy[~np.isnan(chl_summertrend19992020_significant)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
map.scatter(xx, yy, 2, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none')
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Correlation', fontsize=14)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\correlation_CHLSAM_SON.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020_significant[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
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
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\correlation_CHLSAM_onlysignificant_SON.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Now for MA
for i in np.arange(1998, 2023):
    yeartemp_mar = chl[:,:, (time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = chl[:,:, (time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_mar, yeartemp_apr)),2)
    if i == 1998:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
# SEP-APR
for i in np.arange(1998, 2023):
    yeartemp_sepdec = sam_daily[(time_date_years_sam == i) & ((time_date_months_sam == 3) | (time_date_months_sam == 4)
                                                                )]
    if i == 1998:
        sam_SEPAPR = np.nanmean(yeartemp_sepdec)
    else:
        sam_SEPAPR = np.hstack((sam_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate correlation
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if sum(np.isnan(chl_summerpixel)).any():
            continue
        # remove nans if there exist any
        #if len(np.isnan(chl_summerpixel)) > 0:
        #    nans_chl_summerpixel = np.isnan(chl_summerpixel)
        #    chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
        #    yearchar = yearchar[~nans_chl_summerpixel]
        # calculate linear regression
        corr, pvalue = stats.spearmanr(chl_summerpixel, sam_SEPAPR, nan_policy='omit')
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020[i,j] = corr
        if pvalue < 0.05:
            chl_summertrend19992020_significant[i,j] = corr
#%%
xx, yy = np.meshgrid(lon, lat)
xx = xx[~np.isnan(chl_summertrend19992020_significant)]
yy = yy[~np.isnan(chl_summertrend19992020_significant)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
map.scatter(xx, yy, 2, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none')
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Correlation', fontsize=14)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\correlation_CHLSAM_MA.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020_significant[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
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
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\correlation_CHLSAM_onlysignificant_MA.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

