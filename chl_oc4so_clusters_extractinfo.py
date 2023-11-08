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
#%% Separar por clusters
#chl_4km_clusters = np.empty((len(lat), len(lon)))*np.nan
#for i in range(0, len(lat)):
#    print(i)
#    for j in range(0, len(lon)):
#        # Find which pixel
#        matchups_lat_closest = np.where(lat_clusters == min(lat_clusters, key=lambda x:abs(x-lat[i])))[0][0]
#        matchups_lon_closest = np.where(lon_clusters == min(lon_clusters, key=lambda x:abs(x-lon[j])))[0][0]
#        # Find which cluster each point belongs to
#        cluster_temp = clusters[matchups_lat_closest, matchups_lon_closest]
#        # Assign to matrix
#        chl_4km_clusters[i,j] = cluster_temp
#np.savez_compressed('clusters_upscaled_4km', lat = lat, lon = lon, clusters = chl_4km_clusters)
#%% Plot Clusters map
## Create colormap
#cmap = colors.ListedColormap(['white', '#da2b39', '#2c4ea3', '#f2a612', '#6a984e', '#534d41'])
#bounds=[0,1,2,3,4,5,6]
#norm = colors.BoundaryNorm(bounds, cmap.N)
## Plot figure
#plt.figure()
#map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-63))
#map.set_extent([-67, -53, -67, -60])
#f1 = map.pcolormesh(lon, lat, clusters[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat',
#                    cmap=cmap, norm=norm,zorder=0)
#gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
#map.coastlines(resolution='10m', color='black', linewidth=1)
#map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
#                                        edgecolor='k',
#                                        facecolor=cartopy.feature.COLORS['land']))
#legend_elements = [Patch(facecolor='#f2a612', edgecolor='k',label='DRA'),
#                   Patch(facecolor='#6a984e', edgecolor='k',label='BRS'),
#                   Patch(facecolor='#534d41', edgecolor='k',label='WED$_{N}$'),
#                   Patch(facecolor='#2c4ea3', edgecolor='k',label='GES'),
#                   Patch(facecolor='#da2b39', edgecolor='k',label='WED$_{S}$')]
#plt.legend(handles=legend_elements, loc=4)
#plt.tight_layout()
#graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\clusters_upscaled4km.png'
#plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
#plt.close()
#%% Separar para o cluster 1 (WEDsouth)
weds_cluster = chl[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
np.nanmedian(weds_cluster)
np.nanmax(weds_cluster)
np.nanmin(weds_cluster)
np.nanstd(weds_cluster)*3
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = weds_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = weds_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    # remove 29th February
    if len(yeartemp_sepapr_pd) == 243:
        yeartemp_sepapr_pd.drop(yeartemp_sepapr_pd.index[(yeartemp_sepapr_pd.index.month == 2) & (yeartemp_sepapr_pd.index.day == 29)],
                                inplace = True)
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Average for 8 days
    if i == 1998:
        weds_sepapr_8day = yeartemp_sepapr_pd_8day.values
        weds_sepapr_8day_time = yeartemp_sepapr_pd_8day.index
    else:
        weds_sepapr_8day = np.vstack((weds_sepapr_8day, yeartemp_sepapr_pd_8day.values))

#%% 
weds_cluster_mean19982022 = np.nanmean(weds_sepapr_8day,0)
#weds_cluster_mean19981999 = np.nanmean(weds_sepapr_8day[:2,:], axis=0)
weds_cluster_20012010 = np.nanmean(weds_sepapr_8day[3:13,:], axis=0)
weds_cluster_20112020 = np.nanmean(weds_sepapr_8day[13:23,:], axis=0)
#weds_cluster_20202022 = np.nanmean(weds_sepapr_8day[22:,:], axis=0)
weds_yearlycicles_p90 = np.nanpercentile(weds_sepapr_8day, 90, axis=0)
weds_yearlycicles_p10 = np.nanpercentile(weds_sepapr_8day, 10, axis=0)
weds_yearlycicles_std = np.nanstd(weds_sepapr_8day, axis=0)
#%% WEDs Cluster Figure 1
import statsmodels.api as sm

y_lowess_weds = sm.nonparametric.lowess(weds_cluster_mean19982022, np.arange(1,32), frac = 0.30)  # 30 % lowess smoothing

#plt.plot(y_lowess[:, 0], y_lowess[:, 1])
#plt.show()
#f_cubic = interp1d(np.arange(5,33),wed_cluster_mean19972021[4:-6], kind='cubic')
#xnew = np.linspace(5, 32, num=10, endpoint=True)
#f_cubic_p90 = interp1d(np.arange(3,11),weddell_yearlycicles_p90[2:10], kind='cubic')
#f_cubic_p10 = interp1d(np.arange(3,11),weddell_yearlycicles_p10[2:10], kind='cubic')
plt.figure(figsize=(6,4))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(y_lowess_weds[:, 0], y_lowess_weds[:, 1], color = '#da2b39', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),weds_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
plt.plot(np.arange(1,32),weds_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
plt.plot(np.arange(1,32),weds_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),weds_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
plt.fill_between(np.arange(1,32), weds_cluster_mean19982022, weds_cluster_mean19982022+weds_yearlycicles_std, color =[219/256, 43/256, 57/256, 1], alpha=.2, edgecolor = None)
plt.fill_between(np.arange(1,32), weds_cluster_mean19982022, weds_cluster_mean19982022-weds_yearlycicles_std, color =[219/256, 43/256, 57/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
plt.xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
plt.xlim(1,28)
plt.ylim(0, 20)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.yticks(ticks=np.arange(0,25,5))
#plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.legend(fontsize=12, loc=2)
#plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\WEDs_chlseasonalcycle.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 2 (GES)
ges_cluster = chl[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
np.nanmedian(ges_cluster)
np.nanmax(ges_cluster)
np.nanmin(ges_cluster)
np.nanstd(ges_cluster)*3
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = ges_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = ges_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    # remove 29th February
    if len(yeartemp_sepapr_pd) == 243:
        yeartemp_sepapr_pd.drop(yeartemp_sepapr_pd.index[(yeartemp_sepapr_pd.index.month == 2) & (yeartemp_sepapr_pd.index.day == 29)],
                                inplace = True)
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Average for 8 days
    if i == 1998:
        ges_sepapr_8day = yeartemp_sepapr_pd_8day.values
        ges_sepapr_8day_time = yeartemp_sepapr_pd_8day.index
    else:
        ges_sepapr_8day = np.vstack((ges_sepapr_8day, yeartemp_sepapr_pd_8day.values))

#%% 
ges_cluster_mean19982022 = np.nanmean(ges_sepapr_8day,0)
#ges_cluster_mean19981999 = np.nanmean(ges_sepapr_8day[:2,:], axis=0)
ges_cluster_20012010 = np.nanmean(ges_sepapr_8day[3:13,:], axis=0)
ges_cluster_20112020 = np.nanmean(ges_sepapr_8day[13:23,:], axis=0)
#ges_cluster_20202022 = np.nanmean(ges_sepapr_8day[22:,:], axis=0)
ges_yearlycicles_p90 = np.nanpercentile(ges_sepapr_8day, 90, axis=0)
ges_yearlycicles_p10 = np.nanpercentile(ges_sepapr_8day, 10, axis=0)
ges_yearlycicles_std = np.nanstd(ges_sepapr_8day, axis=0)
#%% GES Cluster Figure 1
import statsmodels.api as sm

y_lowess_ges = sm.nonparametric.lowess(ges_cluster_mean19982022, np.arange(1,32), frac = 0.30)  # 30 % lowess smoothing

#plt.plot(y_lowess[:, 0], y_lowess[:, 1])
#plt.show()
#f_cubic = interp1d(np.arange(5,33),wed_cluster_mean19972021[4:-6], kind='cubic')
#xnew = np.linspace(5, 32, num=10, endpoint=True)
#f_cubic_p90 = interp1d(np.arange(3,11),weddell_yearlycicles_p90[2:10], kind='cubic')
#f_cubic_p10 = interp1d(np.arange(3,11),weddell_yearlycicles_p10[2:10], kind='cubic')
plt.figure(figsize=(6,4))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(y_lowess_ges[:, 0], y_lowess_ges[:, 1], color = '#2c4ea3', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),ges_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
plt.plot(np.arange(1,32),ges_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
plt.plot(np.arange(1,32),ges_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),ges_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
plt.fill_between(np.arange(1,32), ges_cluster_mean19982022, ges_cluster_mean19982022+ges_yearlycicles_std, color ='#2c4ea3', alpha=.2, edgecolor = None)
plt.fill_between(np.arange(1,32), ges_cluster_mean19982022, ges_cluster_mean19982022-ges_yearlycicles_std, color ='#2c4ea3', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
plt.xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
plt.xlim(1,28)
plt.ylim(0, 6)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.legend(fontsize=12, loc=2)
#plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\GES_chlseasonalcycle.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 3 (DRA)
dra_cluster = chl[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
np.nanmedian(dra_cluster)
np.nanmax(dra_cluster)
np.nanmin(dra_cluster)
np.nanstd(dra_cluster)*3
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = dra_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = dra_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    # remove 29th February
    if len(yeartemp_sepapr_pd) == 243:
        yeartemp_sepapr_pd.drop(yeartemp_sepapr_pd.index[(yeartemp_sepapr_pd.index.month == 2) & (yeartemp_sepapr_pd.index.day == 29)],
                                inplace = True)
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Average for 8 days
    if i == 1998:
        dra_sepapr_8day = yeartemp_sepapr_pd_8day.values
        dra_sepapr_8day_time = yeartemp_sepapr_pd_8day.index
    else:
        dra_sepapr_8day = np.vstack((dra_sepapr_8day, yeartemp_sepapr_pd_8day.values))

#%% 
dra_cluster_mean19982022 = np.nanmean(dra_sepapr_8day,0)
#dra_cluster_mean19981999 = np.nanmean(dra_sepapr_8day[:2,:], axis=0)
dra_cluster_20012010 = np.nanmean(dra_sepapr_8day[3:13,:], axis=0)
dra_cluster_20112020 = np.nanmean(dra_sepapr_8day[13:23,:], axis=0)
#dra_cluster_20202022 = np.nanmean(dra_sepapr_8day[22:,:], axis=0)
dra_yearlycicles_p90 = np.nanpercentile(dra_sepapr_8day, 90, axis=0)
dra_yearlycicles_p10 = np.nanpercentile(dra_sepapr_8day, 10, axis=0)
dra_yearlycicles_std = np.nanstd(dra_sepapr_8day, axis=0)
#%% DRA Cluster Figure 1
import statsmodels.api as sm

y_lowess_dra = sm.nonparametric.lowess(dra_cluster_mean19982022, np.arange(1,32), frac = 0.30)  # 30 % lowess smoothing

#plt.plot(y_lowess[:, 0], y_lowess[:, 1])
#plt.show()
#f_cubic = interp1d(np.arange(5,33),wed_cluster_mean19972021[4:-6], kind='cubic')
#xnew = np.linspace(5, 32, num=10, endpoint=True)
#f_cubic_p90 = interp1d(np.arange(3,11),weddell_yearlycicles_p90[2:10], kind='cubic')
#f_cubic_p10 = interp1d(np.arange(3,11),weddell_yearlycicles_p10[2:10], kind='cubic')
plt.figure(figsize=(6,4))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(y_lowess_dra[:, 0], y_lowess_dra[:, 1], color = '#f2a612', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),dra_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
plt.plot(np.arange(1,32),dra_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
plt.plot(np.arange(1,32),dra_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),dra_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
plt.fill_between(np.arange(1,32), dra_cluster_mean19982022, dra_cluster_mean19982022+dra_yearlycicles_std, color ='#f2a612', alpha=.2, edgecolor = None)
plt.fill_between(np.arange(1,32), dra_cluster_mean19982022, dra_cluster_mean19982022-dra_yearlycicles_std, color ='#f2a612', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
plt.xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
plt.xlim(1,28)
plt.ylim(0, 1)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(fontsize=12, loc=1)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\DRA_chlseasonalcycle.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 4 (BRS)
brs_cluster = chl[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
np.nanmedian(brs_cluster)
np.nanmax(brs_cluster)
np.nanmin(brs_cluster)
np.nanstd(brs_cluster)*3
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = brs_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = brs_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    # remove 29th February
    if len(yeartemp_sepapr_pd) == 243:
        yeartemp_sepapr_pd.drop(yeartemp_sepapr_pd.index[(yeartemp_sepapr_pd.index.month == 2) & (yeartemp_sepapr_pd.index.day == 29)],
                                inplace = True)
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Average for 8 days
    if i == 1998:
        brs_sepapr_8day = yeartemp_sepapr_pd_8day.values
        brs_sepapr_8day_time = yeartemp_sepapr_pd_8day.index
    else:
        brs_sepapr_8day = np.vstack((brs_sepapr_8day, yeartemp_sepapr_pd_8day.values))

#%% 
brs_cluster_mean19982022 = np.nanmean(brs_sepapr_8day,0)
#brs_cluster_mean19981999 = np.nanmean(brs_sepapr_8day[:2,:], axis=0)
brs_cluster_20012010 = np.nanmean(brs_sepapr_8day[3:13,:], axis=0)
brs_cluster_20112020 = np.nanmean(brs_sepapr_8day[13:23,:], axis=0)
#brs_cluster_20202022 = np.nanmean(brs_sepapr_8day[22:,:], axis=0)
brs_yearlycicles_p90 = np.nanpercentile(brs_sepapr_8day, 90, axis=0)
brs_yearlycicles_p10 = np.nanpercentile(brs_sepapr_8day, 10, axis=0)
brs_yearlycicles_std = np.nanstd(brs_sepapr_8day, axis=0)
#%% BRS Cluster Figure 1
import statsmodels.api as sm

y_lowess_brs = sm.nonparametric.lowess(brs_cluster_mean19982022, np.arange(1,32), frac = 0.30)  # 30 % lowess smoothing

#plt.plot(y_lowess[:, 0], y_lowess[:, 1])
#plt.show()
#f_cubic = interp1d(np.arange(5,33),wed_cluster_mean19972021[4:-6], kind='cubic')
#xnew = np.linspace(5, 32, num=10, endpoint=True)
#f_cubic_p90 = interp1d(np.arange(3,11),weddell_yearlycicles_p90[2:10], kind='cubic')
#f_cubic_p10 = interp1d(np.arange(3,11),weddell_yearlycicles_p10[2:10], kind='cubic')
plt.figure(figsize=(6,4))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(y_lowess_brs[:, 0], y_lowess_brs[:, 1], color = '#6a984e', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),brs_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
plt.plot(np.arange(1,32),brs_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
plt.plot(np.arange(1,32),brs_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),brs_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
plt.fill_between(np.arange(1,32), brs_cluster_mean19982022, brs_cluster_mean19982022+brs_yearlycicles_std, color ='#6a984e', alpha=.2, edgecolor = None)
plt.fill_between(np.arange(1,32), brs_cluster_mean19982022, brs_cluster_mean19982022-brs_yearlycicles_std, color ='#6a984e', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
plt.xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
plt.xlim(1,28)
plt.ylim(0, 1.55)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\BRS_chlseasonalcycle.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 5 (WEDn)
wedn_cluster = chl[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
np.nanmedian(wedn_cluster)
np.nanmax(wedn_cluster)
np.nanmin(wedn_cluster)
np.nanstd(wedn_cluster)*3
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = wedn_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = wedn_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    # remove 29th February
    if len(yeartemp_sepapr_pd) == 243:
        yeartemp_sepapr_pd.drop(yeartemp_sepapr_pd.index[(yeartemp_sepapr_pd.index.month == 2) & (yeartemp_sepapr_pd.index.day == 29)],
                                inplace = True)
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Average for 8 days
    if i == 1998:
        wedn_sepapr_8day = yeartemp_sepapr_pd_8day.values
        wedn_sepapr_8day_time = yeartemp_sepapr_pd_8day.index
    else:
        wedn_sepapr_8day = np.vstack((wedn_sepapr_8day, yeartemp_sepapr_pd_8day.values))

#%% 
wedn_cluster_mean19982022 = np.nanmean(wedn_sepapr_8day,0)
#wedn_cluster_mean19981999 = np.nanmean(wedn_sepapr_8day[:2,:], axis=0)
wedn_cluster_20012010 = np.nanmean(wedn_sepapr_8day[3:13,:], axis=0)
wedn_cluster_20112020 = np.nanmean(wedn_sepapr_8day[13:23,:], axis=0)
#wedn_cluster_20202022 = np.nanmean(wedn_sepapr_8day[22:,:], axis=0)
wedn_yearlycicles_p90 = np.nanpercentile(wedn_sepapr_8day, 90, axis=0)
wedn_yearlycicles_p10 = np.nanpercentile(wedn_sepapr_8day, 10, axis=0)
wedn_yearlycicles_std = np.nanstd(wedn_sepapr_8day, axis=0)
#%% WEDn Cluster Figure 1
import statsmodels.api as sm

y_lowess_wedn = sm.nonparametric.lowess(wedn_cluster_mean19982022, np.arange(1,32), frac = 0.30)  # 30 % lowess smoothing

#plt.plot(y_lowess[:, 0], y_lowess[:, 1])
#plt.show()
#f_cubic = interp1d(np.arange(5,33),wed_cluster_mean19972021[4:-6], kind='cubic')
#xnew = np.linspace(5, 32, num=10, endpoint=True)
#f_cubic_p90 = interp1d(np.arange(3,11),weddell_yearlycicles_p90[2:10], kind='cubic')
#f_cubic_p10 = interp1d(np.arange(3,11),weddell_yearlycicles_p10[2:10], kind='cubic')
plt.figure(figsize=(6,4))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(y_lowess_wedn[:, 0], y_lowess_wedn[:, 1], color = '#534d41', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),wedn_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
plt.plot(np.arange(1,32),wedn_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
plt.plot(np.arange(1,32),wedn_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),wedn_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
plt.fill_between(np.arange(1,32), wedn_cluster_mean19982022, wedn_cluster_mean19982022+wedn_yearlycicles_std, color ='#534d41', alpha=.2, edgecolor = None)
plt.fill_between(np.arange(1,32), wedn_cluster_mean19982022, wedn_cluster_mean19982022-wedn_yearlycicles_std, color ='#534d41', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
plt.xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
plt.xlim(1,28)
plt.ylim(0, 1.8)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\WEDn_chlseasonalcycle.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Plot all in same figure!
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
# Subplot 1 - Map of regions
cmap = colors.ListedColormap(['white', '#da2b39', '#2c4ea3', '#f2a612', '#6a984e', '#534d41'])
bounds=[0,1,2,3,4,5,6]
norm = colors.BoundaryNorm(bounds, cmap.N)
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-63))
ax1.set_extent([-67, -53, -67, -60])
box = ax1.get_position()
box.x0 = box.x0 + 0.05
box.x1 = box.x1 + 0.05
ax1.set_position(box)
f1 = ax1.pcolormesh(lon, lat, clusters[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat',
                    cmap=cmap, norm=norm,zorder=0)
gl = ax1.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
gl.xlabels_top = False
gl.ylabels_right = False
ax1.coastlines(resolution='10m', color='black', linewidth=1)
ax1.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
legend_elements = [Patch(facecolor='#f2a612', edgecolor='k',label='DRA'),
                   Patch(facecolor='#6a984e', edgecolor='k',label='BRS'),
                   Patch(facecolor='#534d41', edgecolor='k',label='WED$_{N}$'),
                   Patch(facecolor='#2c4ea3', edgecolor='k',label='GES'),
                   Patch(facecolor='#da2b39', edgecolor='k',label='WED$_{S}$')]
#ax1.legend(, loc=4, fontsize=8)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(-0.45, .85),
          ncol=1, fancybox=False, shadow=False, fontsize=12, frameon=False)
# Subplot 2 - DRA region
ax2 = fig.add_subplot(gs[0, 1])
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
ax2.plot(y_lowess_dra[:, 0], y_lowess_dra[:, 1], color = '#f2a612', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),dra_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
ax2.plot(np.arange(1,32),dra_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
ax2.plot(np.arange(1,32),dra_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),dra_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
ax2.fill_between(np.arange(1,32), dra_cluster_mean19982022, dra_cluster_mean19982022+dra_yearlycicles_std, color ='#f2a612', alpha=.2, edgecolor = None)
ax2.fill_between(np.arange(1,32), dra_cluster_mean19982022, dra_cluster_mean19982022-dra_yearlycicles_std, color ='#f2a612', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
ax2.set_xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
ax2.set_xlim(1,28)
ax2.set_ylim(0, 1)
ax2.set_ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.legend(fontsize=10, loc=1)
# Subplot 3 - BRS region
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(y_lowess_brs[:, 0], y_lowess_brs[:, 1], color = '#6a984e', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),brs_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
ax3.plot(np.arange(1,32),brs_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
ax3.plot(np.arange(1,32),brs_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),brs_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
ax3.fill_between(np.arange(1,32), brs_cluster_mean19982022, brs_cluster_mean19982022+brs_yearlycicles_std, color ='#6a984e', alpha=.2, edgecolor = None)
ax3.fill_between(np.arange(1,32), brs_cluster_mean19982022, brs_cluster_mean19982022-brs_yearlycicles_std, color ='#6a984e', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
ax3.set_xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
ax3.set_xlim(1,28)
ax3.set_ylim(0, 1.55)
ax3.set_ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)
ax3.legend(fontsize=10, loc=2)
# Subplot 4 - WEDn region
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(y_lowess_wedn[:, 0], y_lowess_wedn[:, 1], color = '#534d41', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),wedn_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
ax4.plot(np.arange(1,32),wedn_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
ax4.plot(np.arange(1,32),wedn_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),wedn_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
ax4.fill_between(np.arange(1,32), wedn_cluster_mean19982022, wedn_cluster_mean19982022+wedn_yearlycicles_std, color ='#534d41', alpha=.2, edgecolor = None)
ax4.fill_between(np.arange(1,32), wedn_cluster_mean19982022, wedn_cluster_mean19982022-wedn_yearlycicles_std, color ='#534d41', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
ax4.set_xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
ax4.set_xlim(1,28)
ax4.set_ylim(0, 1.8)
ax4.set_ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
ax4.tick_params(axis='x', labelsize=12)
ax4.tick_params(axis='y', labelsize=12)
plt.legend(fontsize=10, loc=2)
# Subplot 5 - GES region
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(y_lowess_ges[:, 0], y_lowess_ges[:, 1], color = '#2c4ea3', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),ges_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
ax5.plot(np.arange(1,32),ges_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
ax5.plot(np.arange(1,32),ges_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),ges_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
ax5.fill_between(np.arange(1,32), ges_cluster_mean19982022, ges_cluster_mean19982022+ges_yearlycicles_std, color ='#2c4ea3', alpha=.2, edgecolor = None)
ax5.fill_between(np.arange(1,32), ges_cluster_mean19982022, ges_cluster_mean19982022-ges_yearlycicles_std, color ='#2c4ea3', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
ax5.set_xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
ax5.set_xlim(1,28)
ax5.set_ylim(0, 6)
ax5.set_ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
ax5.tick_params(axis='x', labelsize=12)
ax5.tick_params(axis='y', labelsize=12)
ax5.legend(fontsize=10, loc=2)
# Subplot 6 - WEDs region
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(y_lowess_weds[:, 0], y_lowess_weds[:, 1], color = '#da2b39', linewidth = 4, label='1998-2022', zorder=2)
#plt.plot(np.arange(1,32),weds_cluster_mean19981999, color = 'k', linewidth = 1, linestyle='--', label='1998-1999', alpha=0.4, marker='o', zorder=1)
ax6.plot(np.arange(1,32),weds_cluster_20012010, color = 'k', linewidth = 1, linestyle='--', label='2001-2010', alpha=0.5, marker='s', zorder=1)
ax6.plot(np.arange(1,32),weds_cluster_20112020, color = 'k', linewidth = 1, linestyle='-', label='2011-2020', alpha=0.5, marker='o', zorder=1)
#plt.plot(np.arange(1,32),weds_cluster_20202022, color = 'k', linewidth = 1, linestyle='-', label='2020-2022', alpha=0.4, marker='*', zorder=1)
#plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
#             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
ax6.fill_between(np.arange(1,32), weds_cluster_mean19982022, weds_cluster_mean19982022+weds_yearlycicles_std, color =[219/256, 43/256, 57/256, 1], alpha=.2, edgecolor = None)
ax6.fill_between(np.arange(1,32), weds_cluster_mean19982022, weds_cluster_mean19982022-weds_yearlycicles_std, color =[219/256, 43/256, 57/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
ax6.set_xticks(ticks= [1, 5, 9, 13, 17, 21, 24, 28, 32], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'], fontsize=12)
ax6.set_xlim(1,28)
ax6.set_ylim(0, 20)
ax6.set_ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
ax6.tick_params(axis='x', labelsize=12)
ax6.tick_params(axis='y', labelsize=12)
ax6.legend(fontsize=10, loc=2)
fig.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\Fig1_v2.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

#%%