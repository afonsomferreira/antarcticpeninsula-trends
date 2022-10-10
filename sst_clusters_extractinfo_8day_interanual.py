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
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\oc4so-chl\\')
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
### Load data 1998-2020
fh = np.load('sst_19972021_10km.npz', allow_pickle=True)
lat = fh['lat'][100:]
lon = fh['lon'][30:250]
sst = fh['sst'][100:, 30:250, :]
time_date = fh['time_date'][:]
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
#chl[chl > 50] = 50
# Load clusters
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('antarcticpeninsula_cluster.npz',allow_pickle = True)
clusters = fh['clusters']
#%% Separar para o cluster 1 (Weddell)
weddell_cluster = sst[clusters == 1,:]
weddell_cluster = np.nanmean(weddell_cluster,0)
np.nanmedian(weddell_cluster)
np.nanmax(weddell_cluster)
np.nanmin(weddell_cluster)
np.nanstd(weddell_cluster)*3
#weddell_cluster = np.where(weddell_cluster > np.nanmedian(weddell_cluster)-np.nanstd(weddell_cluster)*3, weddell_cluster, np.nan)
#weddell_cluster = np.where(weddell_cluster < np.nanmedian(weddell_cluster)+np.nanstd(weddell_cluster)*3, weddell_cluster, np.nan)
# Create pandas
weddell_cluster_df = pd.Series(data=weddell_cluster, index=time_date)
weddell_cluster_df_monthly = weddell_cluster_df.resample('M').mean()
# Extract variables to plot
weddell_monthly_19972021 = weddell_cluster_df_monthly.values
weddell_monthly_19972021_index = weddell_cluster_df_monthly.index
# Calculate rolling mean
weddell_monthly_19972021_movmean = weddell_cluster_df_monthly.rolling(window=3, min_periods=1).mean().values
#%% Linear Plot
plt.figure(figsize=(6,6))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(np.arange(0, 300), weddell_monthly_19972021_movmean, color = [43/256, 131/256, 186/256, 1], linewidth = 2, zorder=2)
plt.scatter(np.arange(0, 300),weddell_monthly_19972021, color = 'k', alpha=0.2, marker='o', zorder=1)
plt.axhline(np.nanmean(weddell_monthly_19972021), color = 'k', linewidth = 2, linestyle=':', alpha=0.5, zorder=1)
plt.xticks(ticks=[12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132,
                  144, 156, 168, 180, 192, 204, 216, 228, 240,
                  252, 264, 276, 288, 300], labels=['98', '99', '00', '01', '02', '03', '04', '05', '06', '07',
                                                 '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                                                 '18', '19', '20', '21', '22'], fontsize=12, rotation=90)
plt.xlim(0,300)
plt.ylabel('SST (ºC)', fontsize=14)
#plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\interannual_plots\\weddell_sst.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Boxplot
# Separate data per year
for i in np.arange(1998, 2022):
    yeartemp_sep = weddell_monthly_19972021[(weddell_monthly_19972021_index.year == i-1) & (weddell_monthly_19972021_index.month == 9)]
    yeartemp_oct = weddell_monthly_19972021[(weddell_monthly_19972021_index.year == i-1) & (weddell_monthly_19972021_index.month == 10)]
    yeartemp_nov = weddell_monthly_19972021[(weddell_monthly_19972021_index.year == i-1) & (weddell_monthly_19972021_index.month == 11)]
    yeartemp_dec = weddell_monthly_19972021[(weddell_monthly_19972021_index.year == i-1) & (weddell_monthly_19972021_index.month == 12)]
    yeartemp_jan = weddell_monthly_19972021[(weddell_monthly_19972021_index.year == i) & (weddell_monthly_19972021_index.month == 1)]
    yeartemp_feb = weddell_monthly_19972021[(weddell_monthly_19972021_index.year == i) & (weddell_monthly_19972021_index.month == 2)]
    yeartemp_mar = weddell_monthly_19972021[(weddell_monthly_19972021_index.year == i) & (weddell_monthly_19972021_index.month == 3)]
    yeartemp_apr = weddell_monthly_19972021[(weddell_monthly_19972021_index.year == i) & (weddell_monthly_19972021_index.month == 4)]
    yeartemp_summermean = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                     yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    if i == 1998:
        weddell_boxplots = yeartemp_summermean
    else:
        weddell_boxplots = np.vstack((weddell_boxplots, yeartemp_summermean))
# Boxplot Seasonal
c_weddell = [43/256, 131/256, 186/256, 1]
weddell_boxplots_df = pd.DataFrame(data=weddell_boxplots, index=np.arange(1998, 2022))
weddell_boxplots_df.plot(kind='box', widths = 0.7, 
    patch_artist=True, boxprops=dict(facecolor=c_weddell, color='k'),
    capprops=dict(color='k'), whiskerprops=dict(color='k'),
    flierprops=dict(color=c_weddell, markeredgecolor=c_weddell),
    medianprops=dict(color='k', linewidth=2))
plt.xticks(ticks=np.arange(1,9), labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR'], fontsize=14)
plt.ylabel('SST (ºC)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\seasonal_plots\\weddell_sst_boxplot.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Boxplot Interannual
weddell_boxplots_df = np.transpose(pd.DataFrame(data=weddell_boxplots, index=np.arange(1998, 2022)))
weddell_boxplots_df.plot(kind='box', widths = 0.7, 
    patch_artist=True, boxprops=dict(facecolor=c_weddell, color='k'),
    capprops=dict(color='k'), whiskerprops=dict(color='k'),
    flierprops=dict(color=c_weddell, markeredgecolor=c_weddell),
    medianprops=dict(color='k', linewidth=2))
plt.xticks(ticks=np.arange(1,25), labels=['98', '99', '00', '01', '02', '03', '04', '05',
                                               '06', '07', '08', '09', '10', '11', '12', '13',
                                               '14', '15', '16', '17', '18', '19', '20', '21',
                                               ], fontsize=12)
plt.ylabel('SST (ºC)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\interannual_plots\\weddell_sst_boxplot.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

#%% Separar para o cluster 2 (Gerlache)
gerlache_cluster = sst[clusters == 2,:]
gerlache_cluster = np.nanmean(gerlache_cluster,0)
np.nanmedian(gerlache_cluster)
np.nanmax(gerlache_cluster)
np.nanmin(gerlache_cluster)
np.nanstd(gerlache_cluster)*3
#gerlache_cluster = np.where(gerlache_cluster > np.nanmedian(gerlache_cluster)-np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
#gerlache_cluster = np.where(gerlache_cluster < np.nanmedian(gerlache_cluster)+np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
# Create pandas
gerlache_cluster_df = pd.Series(data=gerlache_cluster, index=time_date)
gerlache_cluster_df_monthly = gerlache_cluster_df.resample('M').mean()
# Extract variables to plot
gerlache_monthly_19972021 = gerlache_cluster_df_monthly.values
gerlache_monthly_19972021_index = gerlache_cluster_df_monthly.index
# Calculate rolling mean
gerlache_monthly_19972021_movmean = gerlache_cluster_df_monthly.rolling(window=3, min_periods=1).mean().values
#%%
plt.figure(figsize=(6,6))
#plt.plot(np.arange(1,13),gerlache_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(np.arange(0, 300), gerlache_monthly_19972021_movmean, color = [215/256, 25/256, 28/256, 1], linewidth = 2, zorder=2)
plt.scatter(np.arange(0, 300),gerlache_monthly_19972021, color = 'k', alpha=0.2, marker='o', zorder=1)
plt.axhline(np.nanmean(gerlache_monthly_19972021), color = 'k', linewidth = 2, linestyle=':', alpha=0.5, zorder=1)


plt.xticks(ticks=[12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132,
                  144, 156, 168, 180, 192, 204, 216, 228, 240,
                  252, 264, 276, 288, 300], labels=['98', '99', '00', '01', '02', '03', '04', '05', '06', '07',
                                                 '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                                                 '18', '19', '20', '21', '22'], fontsize=12, rotation=90)
plt.xlim(0,300)
plt.ylabel('SST (ºC)', fontsize=14)
#plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\interannual_plots\\gerlache_sst.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Boxplot
# Separate data per year
for i in np.arange(1998, 2022):
    yeartemp_sep = gerlache_monthly_19972021[(gerlache_monthly_19972021_index.year == i-1) & (gerlache_monthly_19972021_index.month == 9)]
    yeartemp_oct = gerlache_monthly_19972021[(gerlache_monthly_19972021_index.year == i-1) & (gerlache_monthly_19972021_index.month == 10)]
    yeartemp_nov = gerlache_monthly_19972021[(gerlache_monthly_19972021_index.year == i-1) & (gerlache_monthly_19972021_index.month == 11)]
    yeartemp_dec = gerlache_monthly_19972021[(gerlache_monthly_19972021_index.year == i-1) & (gerlache_monthly_19972021_index.month == 12)]
    yeartemp_jan = gerlache_monthly_19972021[(gerlache_monthly_19972021_index.year == i) & (gerlache_monthly_19972021_index.month == 1)]
    yeartemp_feb = gerlache_monthly_19972021[(gerlache_monthly_19972021_index.year == i) & (gerlache_monthly_19972021_index.month == 2)]
    yeartemp_mar = gerlache_monthly_19972021[(gerlache_monthly_19972021_index.year == i) & (gerlache_monthly_19972021_index.month == 3)]
    yeartemp_apr = gerlache_monthly_19972021[(gerlache_monthly_19972021_index.year == i) & (gerlache_monthly_19972021_index.month == 4)]
    yeartemp_summermean = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                     yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    if i == 1998:
        gerlache_boxplots = yeartemp_summermean
    else:
        gerlache_boxplots = np.vstack((gerlache_boxplots, yeartemp_summermean))
# Boxplot Seasonal
c_gerlache = [215/256, 25/256, 28/256, 1]
c_bransfield = '#9800cb'
c_oceanic = '#d09c26'
gerlache_boxplots_df = pd.DataFrame(data=gerlache_boxplots, index=np.arange(1998, 2022))
gerlache_boxplots_df.plot(kind='box', widths = 0.7, 
    patch_artist=True, boxprops=dict(facecolor=c_gerlache, color='k'),
    capprops=dict(color='k'), whiskerprops=dict(color='k'),
    flierprops=dict(color=c_gerlache, markeredgecolor=c_gerlache),
    medianprops=dict(color='k', linewidth=2))
plt.xticks(ticks=np.arange(1,9), labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR'], fontsize=14)
plt.ylabel('SST (ºC)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\seasonal_plots\\gerlache_sst_boxplot.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Boxplot Interannual
gerlache_boxplots_df = np.transpose(pd.DataFrame(data=gerlache_boxplots, index=np.arange(1998, 2022)))
gerlache_boxplots_df.plot(kind='box', widths = 0.7, 
    patch_artist=True, boxprops=dict(facecolor=c_gerlache, color='k'),
    capprops=dict(color='k'), whiskerprops=dict(color='k'),
    flierprops=dict(color=c_gerlache, markeredgecolor=c_gerlache),
    medianprops=dict(color='k', linewidth=2))
plt.xticks(ticks=np.arange(1,25), labels=['98', '99', '00', '01', '02', '03', '04', '05',
                                               '06', '07', '08', '09', '10', '11', '12', '13',
                                               '14', '15', '16', '17', '18', '19', '20', '21',
                                               ], fontsize=12)
plt.ylabel('SST (ºC)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\interannual_plots\\gerlache_sst_boxplot.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 3 (Oceanic)
oceanic_cluster = sst[clusters == 3,:]
oceanic_cluster = np.nanmean(oceanic_cluster,0)
np.nanmedian(oceanic_cluster)
np.nanmax(oceanic_cluster)
np.nanmin(oceanic_cluster)
np.nanstd(oceanic_cluster)*3
#oceanic_cluster = np.where(oceanic_cluster > np.nanmedian(oceanic_cluster)-np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
#oceanic_cluster = np.where(oceanic_cluster < np.nanmedian(oceanic_cluster)+np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
# Create pandas
oceanic_cluster_df = pd.Series(data=oceanic_cluster, index=time_date)
oceanic_cluster_df_monthly = oceanic_cluster_df.resample('M').mean()
# Extract variables to plot
oceanic_monthly_19972021 = oceanic_cluster_df_monthly.values
oceanic_monthly_19972021_index = oceanic_cluster_df_monthly.index
# Calculate rolling mean
oceanic_monthly_19972021_movmean = oceanic_cluster_df_monthly.rolling(window=3, min_periods=1).mean().values
#%%
plt.figure(figsize=(6,6))
#plt.plot(np.arange(1,13),oceanic_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(np.arange(0, 300), oceanic_monthly_19972021_movmean, color = '#d09c26', linewidth = 2, zorder=2)
plt.scatter(np.arange(0, 300),oceanic_monthly_19972021, color = 'k', alpha=0.2, marker='o', zorder=1)
plt.axhline(np.nanmean(oceanic_monthly_19972021), color = 'k', linewidth = 2, linestyle=':', alpha=0.5, zorder=1)

plt.xticks(ticks=[12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132,
                  144, 156, 168, 180, 192, 204, 216, 228, 240,
                  252, 264, 276, 288, 300], labels=['98', '99', '00', '01', '02', '03', '04', '05', '06', '07',
                                                 '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                                                 '18', '19', '20', '21', '22'], fontsize=12, rotation=90)
plt.xlim(0,300)
plt.ylabel('SST (ºC)', fontsize=14)
#plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\interannual_plots\\oceanic_sst.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Boxplot
# Separate data per year
for i in np.arange(1998, 2022):
    yeartemp_sep = oceanic_monthly_19972021[(oceanic_monthly_19972021_index.year == i-1) & (oceanic_monthly_19972021_index.month == 9)]
    yeartemp_oct = oceanic_monthly_19972021[(oceanic_monthly_19972021_index.year == i-1) & (oceanic_monthly_19972021_index.month == 10)]
    yeartemp_nov = oceanic_monthly_19972021[(oceanic_monthly_19972021_index.year == i-1) & (oceanic_monthly_19972021_index.month == 11)]
    yeartemp_dec = oceanic_monthly_19972021[(oceanic_monthly_19972021_index.year == i-1) & (oceanic_monthly_19972021_index.month == 12)]
    yeartemp_jan = oceanic_monthly_19972021[(oceanic_monthly_19972021_index.year == i) & (oceanic_monthly_19972021_index.month == 1)]
    yeartemp_feb = oceanic_monthly_19972021[(oceanic_monthly_19972021_index.year == i) & (oceanic_monthly_19972021_index.month == 2)]
    yeartemp_mar = oceanic_monthly_19972021[(oceanic_monthly_19972021_index.year == i) & (oceanic_monthly_19972021_index.month == 3)]
    yeartemp_apr = oceanic_monthly_19972021[(oceanic_monthly_19972021_index.year == i) & (oceanic_monthly_19972021_index.month == 4)]
    yeartemp_summermean = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                     yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    if i == 1998:
        oceanic_boxplots = yeartemp_summermean
    else:
        oceanic_boxplots = np.vstack((oceanic_boxplots, yeartemp_summermean))
# Boxplot Seasonal
c_bransfield = '#9800cb'
c_oceanic = '#d09c26'
oceanic_boxplots_df = pd.DataFrame(data=oceanic_boxplots, index=np.arange(1998, 2022))
oceanic_boxplots_df.plot(kind='box', widths = 0.7, 
    patch_artist=True, boxprops=dict(facecolor=c_oceanic, color='k'),
    capprops=dict(color='k'), whiskerprops=dict(color='k'),
    flierprops=dict(color=c_oceanic, markeredgecolor=c_oceanic),
    medianprops=dict(color='k', linewidth=2))
plt.xticks(ticks=np.arange(1,9), labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR'], fontsize=14)
plt.ylabel('SST (ºC)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\seasonal_plots\\oceanic_sst_boxplot.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Boxplot Interannual
oceanic_boxplots_df = np.transpose(pd.DataFrame(data=oceanic_boxplots, index=np.arange(1998, 2022)))
oceanic_boxplots_df.plot(kind='box', widths = 0.7, 
    patch_artist=True, boxprops=dict(facecolor=c_oceanic, color='k'),
    capprops=dict(color='k'), whiskerprops=dict(color='k'),
    flierprops=dict(color=c_oceanic, markeredgecolor=c_oceanic),
    medianprops=dict(color='k', linewidth=2))
plt.xticks(ticks=np.arange(1,25), labels=['98', '99', '00', '01', '02', '03', '04', '05',
                                               '06', '07', '08', '09', '10', '11', '12', '13',
                                               '14', '15', '16', '17', '18', '19', '20', '21',
                                               ], fontsize=12)
plt.ylabel('SST (ºC)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\interannual_plots\\oceanic_sst_boxplot.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 4 (Bransfield)
bransfield_cluster = sst[clusters == 4,:]
bransfield_cluster = np.nanmean(bransfield_cluster,0)
np.nanmedian(bransfield_cluster)
np.nanmax(bransfield_cluster)
np.nanmin(bransfield_cluster)
np.nanstd(bransfield_cluster)*3
#bransfield_cluster = np.where(bransfield_cluster > np.nanmedian(bransfield_cluster)-np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
#bransfield_cluster = np.where(bransfield_cluster < np.nanmedian(bransfield_cluster)+np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
# Create pandas
bransfield_cluster_df = pd.Series(data=bransfield_cluster, index=time_date)
bransfield_cluster_df_monthly = bransfield_cluster_df.resample('M').mean()
# Extract variables to plot
bransfield_monthly_19972021 = bransfield_cluster_df_monthly.values
bransfield_monthly_19972021_index = bransfield_cluster_df_monthly.index
# Calculate rolling mean
bransfield_monthly_19972021_movmean = bransfield_cluster_df_monthly.rolling(window=3, min_periods=1).mean().values
#%%
plt.figure(figsize=(6,6))
#plt.plot(np.arange(1,13),bransfield_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(np.arange(0, 300), bransfield_monthly_19972021_movmean, color = '#9800cb', linewidth = 2, zorder=2)
plt.scatter(np.arange(0, 300),bransfield_monthly_19972021, color = 'k', alpha=0.2, marker='o', zorder=1)
plt.axhline(np.nanmean(bransfield_monthly_19972021), color = 'k', linewidth = 2, linestyle=':', alpha=0.5, zorder=1)


plt.xticks(ticks=[12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132,
                  144, 156, 168, 180, 192, 204, 216, 228, 240,
                  252, 264, 276, 288, 300], labels=['98', '99', '00', '01', '02', '03', '04', '05', '06', '07',
                                                 '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                                                 '18', '19', '20', '21', '22'], fontsize=12, rotation=90)
plt.xlim(0,300)
plt.ylabel('SST (ºC)', fontsize=14)
#plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\interannual_plots\\bransfield_sst.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Boxplot
# Separate data per year
for i in np.arange(1998, 2022):
    yeartemp_sep = bransfield_monthly_19972021[(bransfield_monthly_19972021_index.year == i-1) & (bransfield_monthly_19972021_index.month == 9)]
    yeartemp_oct = bransfield_monthly_19972021[(bransfield_monthly_19972021_index.year == i-1) & (bransfield_monthly_19972021_index.month == 10)]
    yeartemp_nov = bransfield_monthly_19972021[(bransfield_monthly_19972021_index.year == i-1) & (bransfield_monthly_19972021_index.month == 11)]
    yeartemp_dec = bransfield_monthly_19972021[(bransfield_monthly_19972021_index.year == i-1) & (bransfield_monthly_19972021_index.month == 12)]
    yeartemp_jan = bransfield_monthly_19972021[(bransfield_monthly_19972021_index.year == i) & (bransfield_monthly_19972021_index.month == 1)]
    yeartemp_feb = bransfield_monthly_19972021[(bransfield_monthly_19972021_index.year == i) & (bransfield_monthly_19972021_index.month == 2)]
    yeartemp_mar = bransfield_monthly_19972021[(bransfield_monthly_19972021_index.year == i) & (bransfield_monthly_19972021_index.month == 3)]
    yeartemp_apr = bransfield_monthly_19972021[(bransfield_monthly_19972021_index.year == i) & (bransfield_monthly_19972021_index.month == 4)]
    yeartemp_summermean = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                     yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    if i == 1998:
        bransfield_boxplots = yeartemp_summermean
    else:
        bransfield_boxplots = np.vstack((bransfield_boxplots, yeartemp_summermean))
# Boxplot Seasonal
c_bransfield = '#9800cb'
bransfield_boxplots_df = pd.DataFrame(data=bransfield_boxplots, index=np.arange(1998, 2022))
bransfield_boxplots_df.plot(kind='box', widths = 0.7, 
    patch_artist=True, boxprops=dict(facecolor=c_bransfield, color='k'),
    capprops=dict(color='k'), whiskerprops=dict(color='k'),
    flierprops=dict(color=c_bransfield, markeredgecolor=c_bransfield),
    medianprops=dict(color='k', linewidth=2))
plt.xticks(ticks=np.arange(1,9), labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR'], fontsize=14)
plt.ylabel('SST (ºC)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\seasonal_plots\\bransfield_sst_boxplot.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Boxplot Interannual
bransfield_boxplots_df = np.transpose(pd.DataFrame(data=bransfield_boxplots, index=np.arange(1998, 2022)))
bransfield_boxplots_df.plot(kind='box', widths = 0.7, 
    patch_artist=True, boxprops=dict(facecolor=c_bransfield, color='k'),
    capprops=dict(color='k'), whiskerprops=dict(color='k'),
    flierprops=dict(color=c_bransfield, markeredgecolor=c_bransfield),
    medianprops=dict(color='k', linewidth=2))
plt.xticks(ticks=np.arange(1,25), labels=['98', '99', '00', '01', '02', '03', '04', '05',
                                               '06', '07', '08', '09', '10', '11', '12', '13',
                                               '14', '15', '16', '17', '18', '19', '20', '21',
                                               ], fontsize=12)
plt.ylabel('SST (ºC)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\interannual_plots\\bransfield_sst_boxplot.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()