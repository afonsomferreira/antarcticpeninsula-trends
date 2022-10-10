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
from matplotlib.colors import LinearSegmentedColormap
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
### Load data 1998-2020
fh = np.load('chloc4so_19972021_10km.npz', allow_pickle=True)
lat = fh['lat'][100:]
lon = fh['lon'][30:250]
chl = fh['chl'][100:, 30:250, :]
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
chl[chl > 50] = 50
# Load clusters
fh = np.load('antarcticpeninsula_cluster.npz',allow_pickle = True)
clusters = fh['clusters']
#%% Separar para o cluster 1 (Weddell)
weddell_cluster = chl[clusters == 1,:]
weddell_cluster = np.nanmean(weddell_cluster,0)
np.nanmedian(weddell_cluster)
np.nanmax(weddell_cluster)
np.nanmin(weddell_cluster)
np.nanstd(weddell_cluster)*3
weddell_cluster = np.where(weddell_cluster > np.nanmedian(weddell_cluster)-np.nanstd(weddell_cluster)*3, weddell_cluster, np.nan)
weddell_cluster = np.where(weddell_cluster < np.nanmedian(weddell_cluster)+np.nanstd(weddell_cluster)*3, weddell_cluster, np.nan)
# Convert to pandas series
pixel_alldata_series = pd.Series(weddell_cluster, index=time_date)
# Convert to monthly data
#pixel_alldata_series_monthly = pixel_alldata_series.resample('W').mean()
ix = pd.date_range(start=datetime.date(1997, 8, 1), end=datetime.date(1998, 5, 31), freq='D')

for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 5, 31), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_augmay = weddell_cluster[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_augmay = weddell_cluster[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)        
    ## Take mean value for each half of each month
    aug_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 8][:15])
    aug_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 8][15:])
    sep_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 9][:15])
    sep_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 9][15:])
    oct_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 10][:15])
    oct_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 10][15:])
    nov_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 11][:15])
    nov_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 11][15:])
    dec_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 12][:15])
    dec_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 12][15:])    
    jan_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 1][:15])
    jan_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 1][15:])       
    feb_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 2][:15])
    feb_16_28_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 2][15:])       
    mar_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 3][:15])
    mar_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 3][15:])   
    apr_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 4][:15])
    apr_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 4][15:])     
    may_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 5][:15])
    may_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 5][15:])  
    ## Average Weekly and calculate metrics
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    chl_init_8day_temp = np.argmax(~np.isnan(yeartemp_augmay_pd_8day))
    chl_peak_8day_temp = np.nanargmax(yeartemp_augmay_pd_8day)
    chl_term_8day_temp = np.size(yeartemp_augmay_pd_8day) - np.argmax(~np.isnan(yeartemp_augmay_pd_8day[::-1])) -1  
    chl_max_8day_temp = np.nanmax(yeartemp_augmay_pd_8day)
    chl_cumulativesum_8day_temp = np.nansum(yeartemp_augmay_pd_8day)  
    # Average Biweekly
    yeartemp_augmay_pd_15day = yeartemp_augmay_pd.resample('15D').mean()        
    chl_init_15day_temp = np.argmax(~np.isnan(yeartemp_augmay_pd_15day))
    chl_peak_15day_temp = np.nanargmax(yeartemp_augmay_pd_15day)
    chl_term_15day_temp = np.size(yeartemp_augmay_pd_15day) - np.argmax(~np.isnan(yeartemp_augmay_pd_15day[::-1])) -1  
    chl_max_15day_temp = np.nanmax(yeartemp_augmay_pd_15day)
    chl_cumulativesum_15day_temp = np.nansum(yeartemp_augmay_pd_15day)        
    # Join data
    if i == 1998:
        wed_chl_init_8day = chl_init_8day_temp
        wed_chl_peak_8day = chl_peak_8day_temp
        wed_chl_term_8day = chl_term_8day_temp
        wed_chl_max_8dayp = chl_max_8day_temp
        wed_chl_cumulativesum_8day = chl_cumulativesum_8day_temp
        wed_chl_init_15day = chl_init_15day_temp
        wed_chl_peak_15day = chl_peak_15day_temp
        wed_chl_term_15day = chl_term_15day_temp
        wed_chl_max_15day = chl_max_15day_temp
        wed_chl_cumulativesum_15day = chl_cumulativesum_15day_temp
        wed_aug_1_15 = aug_1_15_temp
        wed_aug_16_31 = aug_16_31_temp
        wed_sep_1_15 = sep_1_15_temp
        wed_sep_16_30 = sep_16_30_temp
        wed_oct_1_15 = oct_1_15_temp
        wed_oct_16_31 = oct_16_31_temp
        wed_nov_1_15 = nov_1_15_temp
        wed_nov_16_30 = nov_16_30_temp
        wed_dec_1_15 = dec_1_15_temp
        wed_dec_16_31 = dec_16_31_temp 
        wed_jan_1_15 = jan_1_15_temp
        wed_jan_16_31 = jan_16_31_temp       
        wed_feb_1_15 = feb_1_15_temp
        wed_feb_16_28 = feb_16_28_temp      
        wed_mar_1_15 = mar_1_15_temp
        wed_mar_16_31 = mar_16_31_temp  
        wed_apr_1_15 = apr_1_15_temp
        wed_apr_16_30 = apr_16_30_temp    
        wed_may_1_15 = may_1_15_temp
        wed_may_16_31 = may_16_31_temp
    else:
        wed_chl_init_8day = np.hstack((wed_chl_init_8day, chl_init_8day_temp))
        wed_chl_peak_8day = np.hstack((wed_chl_peak_8day, chl_peak_8day_temp))
        wed_chl_term_8day = np.hstack((wed_chl_term_8day, chl_term_8day_temp))
        wed_chl_max_8dayp = np.hstack((wed_chl_max_8dayp, chl_max_8day_temp))
        wed_chl_cumulativesum_8day = np.hstack((wed_chl_cumulativesum_8day, chl_cumulativesum_8day_temp))
        wed_chl_init_15day = np.hstack((wed_chl_init_15day, chl_init_15day_temp))
        wed_chl_peak_15day = np.hstack((wed_chl_peak_15day, chl_peak_15day_temp))
        wed_chl_term_15day = np.hstack((wed_chl_term_15day, chl_term_15day_temp))
        wed_chl_max_15day = np.hstack((wed_chl_max_15day, chl_max_15day_temp))
        wed_chl_cumulativesum_15day = np.hstack((wed_chl_cumulativesum_15day, chl_cumulativesum_15day_temp))
        wed_aug_1_15 = np.hstack((wed_aug_1_15, aug_1_15_temp))
        wed_aug_16_31 = np.hstack((wed_aug_16_31, aug_16_31_temp))
        wed_sep_1_15 = np.hstack((wed_sep_1_15, sep_1_15_temp))
        wed_sep_16_30 = np.hstack((wed_sep_16_30, sep_16_30_temp))
        wed_oct_1_15 = np.hstack((wed_oct_1_15, oct_1_15_temp))
        wed_oct_16_31 = np.hstack((wed_oct_16_31, oct_16_31_temp))
        wed_nov_1_15 = np.hstack((wed_nov_1_15, nov_1_15_temp))
        wed_nov_16_30 = np.hstack((wed_nov_16_30, nov_16_30_temp))
        wed_dec_1_15 = np.hstack((wed_dec_1_15, dec_1_15_temp))
        wed_dec_16_31 = np.hstack((wed_dec_16_31, dec_16_31_temp)) 
        wed_jan_1_15 = np.hstack((wed_jan_1_15, jan_1_15_temp))
        wed_jan_16_31 = np.hstack((wed_jan_16_31, jan_16_31_temp))       
        wed_feb_1_15 = np.hstack((wed_feb_1_15, feb_1_15_temp))
        wed_feb_16_28 = np.hstack((wed_feb_16_28, feb_16_28_temp))      
        wed_mar_1_15 = np.hstack((wed_mar_1_15, mar_1_15_temp))
        wed_mar_16_31 = np.hstack((wed_mar_16_31, mar_16_31_temp))  
        wed_apr_1_15 = np.hstack((wed_apr_1_15, apr_1_15_temp))
        wed_apr_16_30 = np.hstack((wed_apr_16_30, apr_16_30_temp))    
        wed_may_1_15 = np.hstack((wed_may_1_15, may_1_15_temp))
        wed_may_16_31 = np.hstack((wed_may_16_31, may_16_31_temp)) 
#%% Join all and plot hovmoller like
wed_fulyearsmeans = np.vstack((wed_aug_1_15, wed_aug_16_31, wed_sep_1_15, wed_sep_16_30,
                               wed_oct_1_15, wed_oct_16_31, wed_nov_1_15, wed_nov_16_30,
                               wed_dec_1_15, wed_dec_16_31, wed_jan_1_15, wed_jan_16_31,
                               wed_feb_1_15, wed_feb_16_28, wed_mar_1_15, wed_mar_16_31,
                               wed_apr_1_15, wed_apr_16_30, wed_may_1_15, wed_may_16_31))
wed_fulyearsmeans_t = np.transpose(wed_fulyearsmeans)
# Test for linear trend
#stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_aug_1_15)][1:], wed_aug_1_15[~np.isnan(wed_aug_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_aug_16_31)][1:], wed_aug_16_31[~np.isnan(wed_aug_16_31)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_sep_1_15)][1:], wed_sep_1_15[~np.isnan(wed_sep_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_sep_16_30)][1:], wed_sep_16_30[~np.isnan(wed_sep_16_30)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_oct_1_15)][1:], wed_oct_1_15[~np.isnan(wed_oct_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_oct_16_31)][1:], wed_oct_16_31[~np.isnan(wed_oct_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_nov_1_15)][1:], wed_nov_1_15[~np.isnan(wed_nov_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_nov_16_30)][1:], wed_nov_16_30[~np.isnan(wed_nov_16_30)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_dec_1_15)][1:], wed_dec_1_15[~np.isnan(wed_dec_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_dec_16_31)][1:], wed_dec_16_31[~np.isnan(wed_dec_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_jan_1_15)][1:], wed_jan_1_15[~np.isnan(wed_jan_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_jan_16_31)][1:], wed_jan_16_31[~np.isnan(wed_jan_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_feb_1_15)][1:], wed_feb_1_15[~np.isnan(wed_feb_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_feb_16_28)][1:], wed_feb_16_28[~np.isnan(wed_feb_16_28)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_mar_1_15)][1:], wed_mar_1_15[~np.isnan(wed_mar_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_mar_16_31)][1:], wed_mar_16_31[~np.isnan(wed_mar_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_apr_1_15)][1:], wed_apr_1_15[~np.isnan(wed_apr_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_apr_16_30)][1:], wed_apr_16_30[~np.isnan(wed_apr_16_30)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_may_1_15)][1:], wed_may_1_15[~np.isnan(wed_may_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(wed_may_16_31)][1:], wed_may_16_31[~np.isnan(wed_may_16_31)][1:])






fig, ax = plt.subplots(figsize=(12, 6))
plt.pcolor(wed_fulyearsmeans_t, cmap=plt.cm.turbo, vmin = 0, vmax=20)
plt.yticks(ticks=np.arange(.5, 24.5), labels=['97-98', '98-99', '99-00', '00-01', '01-02',
                                              '02-03', '03-04', '04-05', '05-06', '06-07',
                                              '07-08', '08-09', '09-10', '10-11', '11-12',
                                              '12-13', '13-14', '14-15', '15-16', '16-17',
                                              '17-18', '18-19', '19-20', '20-21'], fontsize=12)
plt.gca().invert_yaxis()
plt.xticks(ticks=np.arange(1,21,2), labels=['AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'])
plt.axvline(2, c='k', linewidth=1, alpha=0.5)
plt.axvline(4, c='k', linewidth=1, alpha=0.5)
plt.axvline(6, c='k', linewidth=1, alpha=0.5)
plt.axvline(8, c='k', linewidth=1, alpha=0.5)
plt.axvline(10, c='k', linewidth=1, alpha=0.5)
plt.axvline(12, c='k', linewidth=1, alpha=0.5)
plt.axvline(14, c='k', linewidth=1, alpha=0.5)
plt.axvline(16, c='k', linewidth=1, alpha=0.5)
plt.axvline(18, c='k', linewidth=1, alpha=0.5)
cbar = plt.colorbar()
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clusters_plots\\weddell_15dayshovmoller.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 2 (Gerlache)
gerlache_cluster = chl[clusters == 2,:]
gerlache_cluster = np.nanmean(gerlache_cluster,0)
np.nanmedian(gerlache_cluster)
np.nanmax(gerlache_cluster)
np.nanmin(gerlache_cluster)
np.nanstd(gerlache_cluster)*3
gerlache_cluster = np.where(gerlache_cluster > np.nanmedian(gerlache_cluster)-np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
gerlache_cluster = np.where(gerlache_cluster < np.nanmedian(gerlache_cluster)+np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
# Convert to pandas series
pixel_alldata_series = pd.Series(gerlache_cluster, index=time_date)
# Convert to monthly data
#pixel_alldata_series_monthly = pixel_alldata_series.resample('W').mean()
ix = pd.date_range(start=datetime.date(1997, 8, 1), end=datetime.date(1998, 5, 31), freq='D')

for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 5, 31), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_augmay = gerlache_cluster[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_augmay = gerlache_cluster[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)        
    ## Take mean value for each half of each month
    aug_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 8][:15])
    aug_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 8][15:])
    sep_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 9][:15])
    sep_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 9][15:])
    oct_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 10][:15])
    oct_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 10][15:])
    nov_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 11][:15])
    nov_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 11][15:])
    dec_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 12][:15])
    dec_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 12][15:])    
    jan_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 1][:15])
    jan_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 1][15:])       
    feb_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 2][:15])
    feb_16_28_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 2][15:])       
    mar_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 3][:15])
    mar_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 3][15:])   
    apr_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 4][:15])
    apr_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 4][15:])     
    may_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 5][:15])
    may_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 5][15:])  
    ## Average Weekly and calculate metrics
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    chl_init_8day_temp = np.argmax(~np.isnan(yeartemp_augmay_pd_8day))
    chl_peak_8day_temp = np.nanargmax(yeartemp_augmay_pd_8day)
    chl_term_8day_temp = np.size(yeartemp_augmay_pd_8day) - np.argmax(~np.isnan(yeartemp_augmay_pd_8day[::-1])) -1  
    chl_max_8day_temp = np.nanmax(yeartemp_augmay_pd_8day)
    chl_cumulativesum_8day_temp = np.nansum(yeartemp_augmay_pd_8day)  
    # Average Biweekly
    yeartemp_augmay_pd_15day = yeartemp_augmay_pd.resample('15D').mean()        
    chl_init_15day_temp = np.argmax(~np.isnan(yeartemp_augmay_pd_15day))
    chl_peak_15day_temp = np.nanargmax(yeartemp_augmay_pd_15day)
    chl_term_15day_temp = np.size(yeartemp_augmay_pd_15day) - np.argmax(~np.isnan(yeartemp_augmay_pd_15day[::-1])) -1  
    chl_max_15day_temp = np.nanmax(yeartemp_augmay_pd_15day)
    chl_cumulativesum_15day_temp = np.nansum(yeartemp_augmay_pd_15day)        
    # Join data
    if i == 1998:
        ger_chl_init_8day = chl_init_8day_temp
        ger_chl_peak_8day = chl_peak_8day_temp
        ger_chl_term_8day = chl_term_8day_temp
        ger_chl_max_8dayp = chl_max_8day_temp
        ger_chl_cumulativesum_8day = chl_cumulativesum_8day_temp
        ger_chl_init_15day = chl_init_15day_temp
        ger_chl_peak_15day = chl_peak_15day_temp
        ger_chl_term_15day = chl_term_15day_temp
        ger_chl_max_15day = chl_max_15day_temp
        ger_chl_cumulativesum_15day = chl_cumulativesum_15day_temp
        ger_aug_1_15 = aug_1_15_temp
        ger_aug_16_31 = aug_16_31_temp
        ger_sep_1_15 = sep_1_15_temp
        ger_sep_16_30 = sep_16_30_temp
        ger_oct_1_15 = oct_1_15_temp
        ger_oct_16_31 = oct_16_31_temp
        ger_nov_1_15 = nov_1_15_temp
        ger_nov_16_30 = nov_16_30_temp
        ger_dec_1_15 = dec_1_15_temp
        ger_dec_16_31 = dec_16_31_temp 
        ger_jan_1_15 = jan_1_15_temp
        ger_jan_16_31 = jan_16_31_temp       
        ger_feb_1_15 = feb_1_15_temp
        ger_feb_16_28 = feb_16_28_temp      
        ger_mar_1_15 = mar_1_15_temp
        ger_mar_16_31 = mar_16_31_temp  
        ger_apr_1_15 = apr_1_15_temp
        ger_apr_16_30 = apr_16_30_temp    
        ger_may_1_15 = may_1_15_temp
        ger_may_16_31 = may_16_31_temp
    else:
        ger_chl_init_8day = np.hstack((ger_chl_init_8day, chl_init_8day_temp))
        ger_chl_peak_8day = np.hstack((ger_chl_peak_8day, chl_peak_8day_temp))
        ger_chl_term_8day = np.hstack((ger_chl_term_8day, chl_term_8day_temp))
        ger_chl_max_8dayp = np.hstack((ger_chl_max_8dayp, chl_max_8day_temp))
        ger_chl_cumulativesum_8day = np.hstack((ger_chl_cumulativesum_8day, chl_cumulativesum_8day_temp))
        ger_chl_init_15day = np.hstack((ger_chl_init_15day, chl_init_15day_temp))
        ger_chl_peak_15day = np.hstack((ger_chl_peak_15day, chl_peak_15day_temp))
        ger_chl_term_15day = np.hstack((ger_chl_term_15day, chl_term_15day_temp))
        ger_chl_max_15day = np.hstack((ger_chl_max_15day, chl_max_15day_temp))
        ger_chl_cumulativesum_15day = np.hstack((ger_chl_cumulativesum_15day, chl_cumulativesum_15day_temp))
        ger_aug_1_15 = np.hstack((ger_aug_1_15, aug_1_15_temp))
        ger_aug_16_31 = np.hstack((ger_aug_16_31, aug_16_31_temp))
        ger_sep_1_15 = np.hstack((ger_sep_1_15, sep_1_15_temp))
        ger_sep_16_30 = np.hstack((ger_sep_16_30, sep_16_30_temp))
        ger_oct_1_15 = np.hstack((ger_oct_1_15, oct_1_15_temp))
        ger_oct_16_31 = np.hstack((ger_oct_16_31, oct_16_31_temp))
        ger_nov_1_15 = np.hstack((ger_nov_1_15, nov_1_15_temp))
        ger_nov_16_30 = np.hstack((ger_nov_16_30, nov_16_30_temp))
        ger_dec_1_15 = np.hstack((ger_dec_1_15, dec_1_15_temp))
        ger_dec_16_31 = np.hstack((ger_dec_16_31, dec_16_31_temp)) 
        ger_jan_1_15 = np.hstack((ger_jan_1_15, jan_1_15_temp))
        ger_jan_16_31 = np.hstack((ger_jan_16_31, jan_16_31_temp))       
        ger_feb_1_15 = np.hstack((ger_feb_1_15, feb_1_15_temp))
        ger_feb_16_28 = np.hstack((ger_feb_16_28, feb_16_28_temp))      
        ger_mar_1_15 = np.hstack((ger_mar_1_15, mar_1_15_temp))
        ger_mar_16_31 = np.hstack((ger_mar_16_31, mar_16_31_temp))  
        ger_apr_1_15 = np.hstack((ger_apr_1_15, apr_1_15_temp))
        ger_apr_16_30 = np.hstack((ger_apr_16_30, apr_16_30_temp))    
        ger_may_1_15 = np.hstack((ger_may_1_15, may_1_15_temp))
        ger_may_16_31 = np.hstack((ger_may_16_31, may_16_31_temp)) 
#%% Join all and plot hovmoller like
ger_fulyearsmeans = np.vstack((ger_aug_1_15, ger_aug_16_31, ger_sep_1_15, ger_sep_16_30,
                               ger_oct_1_15, ger_oct_16_31, ger_nov_1_15, ger_nov_16_30,
                               ger_dec_1_15, ger_dec_16_31, ger_jan_1_15, ger_jan_16_31,
                               ger_feb_1_15, ger_feb_16_28, ger_mar_1_15, ger_mar_16_31,
                               ger_apr_1_15, ger_apr_16_30, ger_may_1_15, ger_may_16_31))
ger_fulyearsmeans_t = np.transpose(ger_fulyearsmeans)
# Test for linear trend
#stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_aug_1_15)][1:], ger_aug_1_15[~np.isnan(ger_aug_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_aug_16_31)][1:], ger_aug_16_31[~np.isnan(ger_aug_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_sep_1_15)][1:], ger_sep_1_15[~np.isnan(ger_sep_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_sep_16_30)][1:], ger_sep_16_30[~np.isnan(ger_sep_16_30)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_oct_1_15)][1:], ger_oct_1_15[~np.isnan(ger_oct_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_oct_16_31)][1:], ger_oct_16_31[~np.isnan(ger_oct_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_nov_1_15)][1:], ger_nov_1_15[~np.isnan(ger_nov_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_nov_16_30)][1:], ger_nov_16_30[~np.isnan(ger_nov_16_30)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_dec_1_15)][1:], ger_dec_1_15[~np.isnan(ger_dec_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_dec_16_31)][1:], ger_dec_16_31[~np.isnan(ger_dec_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_jan_1_15)][1:], ger_jan_1_15[~np.isnan(ger_jan_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_jan_16_31)][1:], ger_jan_16_31[~np.isnan(ger_jan_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_feb_1_15)][1:], ger_feb_1_15[~np.isnan(ger_feb_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_feb_16_28)][1:], ger_feb_16_28[~np.isnan(ger_feb_16_28)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_mar_1_15)][1:], ger_mar_1_15[~np.isnan(ger_mar_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_mar_16_31)][1:], ger_mar_16_31[~np.isnan(ger_mar_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_apr_1_15)][1:], ger_apr_1_15[~np.isnan(ger_apr_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_apr_16_30)][1:], ger_apr_16_30[~np.isnan(ger_apr_16_30)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_may_1_15)][1:], ger_may_1_15[~np.isnan(ger_may_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(ger_may_16_31)][1:], ger_may_16_31[~np.isnan(ger_may_16_31)][1:])







fig, ax = plt.subplots(figsize=(12, 6))
plt.pcolor(ger_fulyearsmeans_t, cmap=plt.cm.turbo, vmin = 0, vmax=8)
plt.yticks(ticks=np.arange(.5, 24.5), labels=['97-98', '98-99', '99-00', '00-01', '01-02',
                                              '02-03', '03-04', '04-05', '05-06', '06-07',
                                              '07-08', '08-09', '09-10', '10-11', '11-12',
                                              '12-13', '13-14', '14-15', '15-16', '16-17',
                                              '17-18', '18-19', '19-20', '20-21'], fontsize=12)
plt.gca().invert_yaxis()
plt.xticks(ticks=np.arange(1,21,2), labels=['AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'])
plt.axvline(2, c='k', linewidth=1, alpha=0.5)
plt.axvline(4, c='k', linewidth=1, alpha=0.5)
plt.axvline(6, c='k', linewidth=1, alpha=0.5)
plt.axvline(8, c='k', linewidth=1, alpha=0.5)
plt.axvline(10, c='k', linewidth=1, alpha=0.5)
plt.axvline(12, c='k', linewidth=1, alpha=0.5)
plt.axvline(14, c='k', linewidth=1, alpha=0.5)
plt.axvline(16, c='k', linewidth=1, alpha=0.5)
plt.axvline(18, c='k', linewidth=1, alpha=0.5)
cbar = plt.colorbar()
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clusters_plots\\gerlache_15dayshovmoller.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

#%% Separar para o cluster 3 (Oceanic)
oceanic_cluster = chl[clusters == 3,:]
oceanic_cluster = np.nanmean(oceanic_cluster,0)
np.nanmedian(oceanic_cluster)
np.nanmax(oceanic_cluster)
np.nanmin(oceanic_cluster)
np.nanstd(oceanic_cluster)*3
oceanic_cluster = np.where(oceanic_cluster > np.nanmedian(oceanic_cluster)-np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
oceanic_cluster = np.where(oceanic_cluster < np.nanmedian(oceanic_cluster)+np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
# Convert to pandas series
pixel_alldata_series = pd.Series(oceanic_cluster, index=time_date)
# Convert to monthly data
#pixel_alldata_series_monthly = pixel_alldata_series.resample('W').mean()
ix = pd.date_range(start=datetime.date(1997, 8, 1), end=datetime.date(1998, 5, 31), freq='D')

for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 5, 31), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_augmay = oceanic_cluster[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_augmay = oceanic_cluster[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)        
    ## Take mean value for each half of each month
    aug_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 8][:15])
    aug_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 8][15:])
    sep_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 9][:15])
    sep_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 9][15:])
    oct_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 10][:15])
    oct_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 10][15:])
    nov_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 11][:15])
    nov_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 11][15:])
    dec_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 12][:15])
    dec_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 12][15:])    
    jan_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 1][:15])
    jan_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 1][15:])       
    feb_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 2][:15])
    feb_16_28_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 2][15:])       
    mar_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 3][:15])
    mar_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 3][15:])   
    apr_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 4][:15])
    apr_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 4][15:])     
    may_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 5][:15])
    may_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 5][15:])  
    ## Average Weekly and calculate metrics
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    chl_init_8day_temp = np.argmax(~np.isnan(yeartemp_augmay_pd_8day))
    chl_peak_8day_temp = np.nanargmax(yeartemp_augmay_pd_8day)
    chl_term_8day_temp = np.size(yeartemp_augmay_pd_8day) - np.argmax(~np.isnan(yeartemp_augmay_pd_8day[::-1])) -1  
    chl_max_8day_temp = np.nanmax(yeartemp_augmay_pd_8day)
    chl_cumulativesum_8day_temp = np.nansum(yeartemp_augmay_pd_8day)  
    # Average Biweekly
    yeartemp_augmay_pd_15day = yeartemp_augmay_pd.resample('15D').mean()        
    chl_init_15day_temp = np.argmax(~np.isnan(yeartemp_augmay_pd_15day))
    chl_peak_15day_temp = np.nanargmax(yeartemp_augmay_pd_15day)
    chl_term_15day_temp = np.size(yeartemp_augmay_pd_15day) - np.argmax(~np.isnan(yeartemp_augmay_pd_15day[::-1])) -1  
    chl_max_15day_temp = np.nanmax(yeartemp_augmay_pd_15day)
    chl_cumulativesum_15day_temp = np.nansum(yeartemp_augmay_pd_15day)        
    # Join data
    if i == 1998:
        oce_chl_init_8day = chl_init_8day_temp
        oce_chl_peak_8day = chl_peak_8day_temp
        oce_chl_term_8day = chl_term_8day_temp
        oce_chl_max_8dayp = chl_max_8day_temp
        oce_chl_cumulativesum_8day = chl_cumulativesum_8day_temp
        oce_chl_init_15day = chl_init_15day_temp
        oce_chl_peak_15day = chl_peak_15day_temp
        oce_chl_term_15day = chl_term_15day_temp
        oce_chl_max_15day = chl_max_15day_temp
        oce_chl_cumulativesum_15day = chl_cumulativesum_15day_temp
        oce_aug_1_15 = aug_1_15_temp
        oce_aug_16_31 = aug_16_31_temp
        oce_sep_1_15 = sep_1_15_temp
        oce_sep_16_30 = sep_16_30_temp
        oce_oct_1_15 = oct_1_15_temp
        oce_oct_16_31 = oct_16_31_temp
        oce_nov_1_15 = nov_1_15_temp
        oce_nov_16_30 = nov_16_30_temp
        oce_dec_1_15 = dec_1_15_temp
        oce_dec_16_31 = dec_16_31_temp 
        oce_jan_1_15 = jan_1_15_temp
        oce_jan_16_31 = jan_16_31_temp       
        oce_feb_1_15 = feb_1_15_temp
        oce_feb_16_28 = feb_16_28_temp      
        oce_mar_1_15 = mar_1_15_temp
        oce_mar_16_31 = mar_16_31_temp  
        oce_apr_1_15 = apr_1_15_temp
        oce_apr_16_30 = apr_16_30_temp    
        oce_may_1_15 = may_1_15_temp
        oce_may_16_31 = may_16_31_temp
    else:
        oce_chl_init_8day = np.hstack((oce_chl_init_8day, chl_init_8day_temp))
        oce_chl_peak_8day = np.hstack((oce_chl_peak_8day, chl_peak_8day_temp))
        oce_chl_term_8day = np.hstack((oce_chl_term_8day, chl_term_8day_temp))
        oce_chl_max_8dayp = np.hstack((oce_chl_max_8dayp, chl_max_8day_temp))
        oce_chl_cumulativesum_8day = np.hstack((oce_chl_cumulativesum_8day, chl_cumulativesum_8day_temp))
        oce_chl_init_15day = np.hstack((oce_chl_init_15day, chl_init_15day_temp))
        oce_chl_peak_15day = np.hstack((oce_chl_peak_15day, chl_peak_15day_temp))
        oce_chl_term_15day = np.hstack((oce_chl_term_15day, chl_term_15day_temp))
        oce_chl_max_15day = np.hstack((oce_chl_max_15day, chl_max_15day_temp))
        oce_chl_cumulativesum_15day = np.hstack((oce_chl_cumulativesum_15day, chl_cumulativesum_15day_temp))
        oce_aug_1_15 = np.hstack((oce_aug_1_15, aug_1_15_temp))
        oce_aug_16_31 = np.hstack((oce_aug_16_31, aug_16_31_temp))
        oce_sep_1_15 = np.hstack((oce_sep_1_15, sep_1_15_temp))
        oce_sep_16_30 = np.hstack((oce_sep_16_30, sep_16_30_temp))
        oce_oct_1_15 = np.hstack((oce_oct_1_15, oct_1_15_temp))
        oce_oct_16_31 = np.hstack((oce_oct_16_31, oct_16_31_temp))
        oce_nov_1_15 = np.hstack((oce_nov_1_15, nov_1_15_temp))
        oce_nov_16_30 = np.hstack((oce_nov_16_30, nov_16_30_temp))
        oce_dec_1_15 = np.hstack((oce_dec_1_15, dec_1_15_temp))
        oce_dec_16_31 = np.hstack((oce_dec_16_31, dec_16_31_temp)) 
        oce_jan_1_15 = np.hstack((oce_jan_1_15, jan_1_15_temp))
        oce_jan_16_31 = np.hstack((oce_jan_16_31, jan_16_31_temp))       
        oce_feb_1_15 = np.hstack((oce_feb_1_15, feb_1_15_temp))
        oce_feb_16_28 = np.hstack((oce_feb_16_28, feb_16_28_temp))      
        oce_mar_1_15 = np.hstack((oce_mar_1_15, mar_1_15_temp))
        oce_mar_16_31 = np.hstack((oce_mar_16_31, mar_16_31_temp))  
        oce_apr_1_15 = np.hstack((oce_apr_1_15, apr_1_15_temp))
        oce_apr_16_30 = np.hstack((oce_apr_16_30, apr_16_30_temp))    
        oce_may_1_15 = np.hstack((oce_may_1_15, may_1_15_temp))
        oce_may_16_31 = np.hstack((oce_may_16_31, may_16_31_temp))  
    
#%% Join all and plot hovmoller like
oce_fulyearsmeans = np.vstack((oce_aug_1_15, oce_aug_16_31, oce_sep_1_15, oce_sep_16_30,
                               oce_oct_1_15, oce_oct_16_31, oce_nov_1_15, oce_nov_16_30,
                               oce_dec_1_15, oce_dec_16_31, oce_jan_1_15, oce_jan_16_31,
                               oce_feb_1_15, oce_feb_16_28, oce_mar_1_15, oce_mar_16_31,
                               oce_apr_1_15, oce_apr_16_30, oce_may_1_15, oce_may_16_31))
oce_fulyearsmeans_t = np.transpose(oce_fulyearsmeans)
# Test for linear trend
#stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_aug_1_15)][1:], oce_aug_1_15[~np.isnan(oce_aug_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_aug_16_31)][1:], oce_aug_16_31[~np.isnan(oce_aug_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_sep_1_15)][1:], oce_sep_1_15[~np.isnan(oce_sep_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_sep_16_30)][1:], oce_sep_16_30[~np.isnan(oce_sep_16_30)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_oct_1_15)][1:], oce_oct_1_15[~np.isnan(oce_oct_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_oct_16_31)][1:], oce_oct_16_31[~np.isnan(oce_oct_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_nov_1_15)][1:], oce_nov_1_15[~np.isnan(oce_nov_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_nov_16_30)][1:], oce_nov_16_30[~np.isnan(oce_nov_16_30)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_dec_1_15)][1:], oce_dec_1_15[~np.isnan(oce_dec_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_dec_16_31)][1:], oce_dec_16_31[~np.isnan(oce_dec_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_jan_1_15)][1:], oce_jan_1_15[~np.isnan(oce_jan_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_jan_16_31)][1:], oce_jan_16_31[~np.isnan(oce_jan_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_feb_1_15)][1:], oce_feb_1_15[~np.isnan(oce_feb_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_feb_16_28)][1:], oce_feb_16_28[~np.isnan(oce_feb_16_28)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_mar_1_15)][1:], oce_mar_1_15[~np.isnan(oce_mar_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_mar_16_31)][1:], oce_mar_16_31[~np.isnan(oce_mar_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_apr_1_15)][1:], oce_apr_1_15[~np.isnan(oce_apr_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_apr_16_30)][1:], oce_apr_16_30[~np.isnan(oce_apr_16_30)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_may_1_15)][1:], oce_may_1_15[~np.isnan(oce_may_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(oce_may_16_31)][1:], oce_may_16_31[~np.isnan(oce_may_16_31)][1:])







fig, ax = plt.subplots(figsize=(12, 6))
plt.pcolor(oce_fulyearsmeans_t, cmap=plt.cm.turbo, vmin = 0, vmax=1.5)
plt.yticks(ticks=np.arange(.5, 24.5), labels=['97-98', '98-99', '99-00', '00-01', '01-02',
                                              '02-03', '03-04', '04-05', '05-06', '06-07',
                                              '07-08', '08-09', '09-10', '10-11', '11-12',
                                              '12-13', '13-14', '14-15', '15-16', '16-17',
                                              '17-18', '18-19', '19-20', '20-21'], fontsize=12)
plt.gca().invert_yaxis()
plt.xticks(ticks=np.arange(1,21,2), labels=['AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'])
plt.axvline(2, c='k', linewidth=1, alpha=0.5)
plt.axvline(4, c='k', linewidth=1, alpha=0.5)
plt.axvline(6, c='k', linewidth=1, alpha=0.5)
plt.axvline(8, c='k', linewidth=1, alpha=0.5)
plt.axvline(10, c='k', linewidth=1, alpha=0.5)
plt.axvline(12, c='k', linewidth=1, alpha=0.5)
plt.axvline(14, c='k', linewidth=1, alpha=0.5)
plt.axvline(16, c='k', linewidth=1, alpha=0.5)
plt.axvline(18, c='k', linewidth=1, alpha=0.5)
cbar = plt.colorbar()
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clusters_plots\\oceanic_15dayshovmoller.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()



#%% Separar para o cluster 4 (Bransfield)
bransfield_cluster = chl[clusters == 4,:]
bransfield_cluster = np.nanmean(bransfield_cluster,0)
np.nanmedian(bransfield_cluster)
np.nanmax(bransfield_cluster)
np.nanmin(bransfield_cluster)
np.nanstd(bransfield_cluster)*3
bransfield_cluster = np.where(bransfield_cluster > np.nanmedian(bransfield_cluster)-np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
bransfield_cluster = np.where(bransfield_cluster < np.nanmedian(bransfield_cluster)+np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
# Convert to pandas series
pixel_alldata_series = pd.Series(bransfield_cluster, index=time_date)
# Convert to monthly data
#pixel_alldata_series_monthly = pixel_alldata_series.resample('W').mean()
ix = pd.date_range(start=datetime.date(1997, 8, 1), end=datetime.date(1998, 5, 31), freq='D')

for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 5, 31), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_augmay = bransfield_cluster[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_augmay = bransfield_cluster[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)        
    ## Take mean value for each half of each month
    aug_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 8][:15])
    aug_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 8][15:])
    sep_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 9][:15])
    sep_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 9][15:])
    oct_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 10][:15])
    oct_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 10][15:])
    nov_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 11][:15])
    nov_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 11][15:])
    dec_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 12][:15])
    dec_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 12][15:])    
    jan_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 1][:15])
    jan_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 1][15:])       
    feb_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 2][:15])
    feb_16_28_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 2][15:])       
    mar_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 3][:15])
    mar_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 3][15:])   
    apr_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 4][:15])
    apr_16_30_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 4][15:])     
    may_1_15_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 5][:15])
    may_16_31_temp = np.nanmean(yeartemp_augmay_pd[yeartemp_augmay_pd.index.month == 5][15:])  
    ## Average Weekly and calculate metrics
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    chl_init_8day_temp = np.argmax(~np.isnan(yeartemp_augmay_pd_8day))
    chl_peak_8day_temp = np.nanargmax(yeartemp_augmay_pd_8day)
    chl_term_8day_temp = np.size(yeartemp_augmay_pd_8day) - np.argmax(~np.isnan(yeartemp_augmay_pd_8day[::-1])) -1  
    chl_max_8day_temp = np.nanmax(yeartemp_augmay_pd_8day)
    chl_cumulativesum_8day_temp = np.nansum(yeartemp_augmay_pd_8day)  
    # Average Biweekly
    yeartemp_augmay_pd_15day = yeartemp_augmay_pd.resample('15D').mean()        
    chl_init_15day_temp = np.argmax(~np.isnan(yeartemp_augmay_pd_15day))
    chl_peak_15day_temp = np.nanargmax(yeartemp_augmay_pd_15day)
    chl_term_15day_temp = np.size(yeartemp_augmay_pd_15day) - np.argmax(~np.isnan(yeartemp_augmay_pd_15day[::-1])) -1  
    chl_max_15day_temp = np.nanmax(yeartemp_augmay_pd_15day)
    chl_cumulativesum_15day_temp = np.nansum(yeartemp_augmay_pd_15day)        
    # Join data
    if i == 1998:
        bra_chl_init_8day = chl_init_8day_temp
        bra_chl_peak_8day = chl_peak_8day_temp
        bra_chl_term_8day = chl_term_8day_temp
        bra_chl_max_8dayp = chl_max_8day_temp
        bra_chl_cumulativesum_8day = chl_cumulativesum_8day_temp
        bra_chl_init_15day = chl_init_15day_temp
        bra_chl_peak_15day = chl_peak_15day_temp
        bra_chl_term_15day = chl_term_15day_temp
        bra_chl_max_15day = chl_max_15day_temp
        bra_chl_cumulativesum_15day = chl_cumulativesum_15day_temp
        bra_aug_1_15 = aug_1_15_temp
        bra_aug_16_31 = aug_16_31_temp
        bra_sep_1_15 = sep_1_15_temp
        bra_sep_16_30 = sep_16_30_temp
        bra_oct_1_15 = oct_1_15_temp
        bra_oct_16_31 = oct_16_31_temp
        bra_nov_1_15 = nov_1_15_temp
        bra_nov_16_30 = nov_16_30_temp
        bra_dec_1_15 = dec_1_15_temp
        bra_dec_16_31 = dec_16_31_temp 
        bra_jan_1_15 = jan_1_15_temp
        bra_jan_16_31 = jan_16_31_temp       
        bra_feb_1_15 = feb_1_15_temp
        bra_feb_16_28 = feb_16_28_temp      
        bra_mar_1_15 = mar_1_15_temp
        bra_mar_16_31 = mar_16_31_temp  
        bra_apr_1_15 = apr_1_15_temp
        bra_apr_16_30 = apr_16_30_temp    
        bra_may_1_15 = may_1_15_temp
        bra_may_16_31 = may_16_31_temp
    else:
        bra_chl_init_8day = np.hstack((bra_chl_init_8day, chl_init_8day_temp))
        bra_chl_peak_8day = np.hstack((bra_chl_peak_8day, chl_peak_8day_temp))
        bra_chl_term_8day = np.hstack((bra_chl_term_8day, chl_term_8day_temp))
        bra_chl_max_8dayp = np.hstack((bra_chl_max_8dayp, chl_max_8day_temp))
        bra_chl_cumulativesum_8day = np.hstack((bra_chl_cumulativesum_8day, chl_cumulativesum_8day_temp))
        bra_chl_init_15day = np.hstack((bra_chl_init_15day, chl_init_15day_temp))
        bra_chl_peak_15day = np.hstack((bra_chl_peak_15day, chl_peak_15day_temp))
        bra_chl_term_15day = np.hstack((bra_chl_term_15day, chl_term_15day_temp))
        bra_chl_max_15day = np.hstack((bra_chl_max_15day, chl_max_15day_temp))
        bra_chl_cumulativesum_15day = np.hstack((bra_chl_cumulativesum_15day, chl_cumulativesum_15day_temp))
        bra_aug_1_15 = np.hstack((bra_aug_1_15, aug_1_15_temp))
        bra_aug_16_31 = np.hstack((bra_aug_16_31, aug_16_31_temp))
        bra_sep_1_15 = np.hstack((bra_sep_1_15, sep_1_15_temp))
        bra_sep_16_30 = np.hstack((bra_sep_16_30, sep_16_30_temp))
        bra_oct_1_15 = np.hstack((bra_oct_1_15, oct_1_15_temp))
        bra_oct_16_31 = np.hstack((bra_oct_16_31, oct_16_31_temp))
        bra_nov_1_15 = np.hstack((bra_nov_1_15, nov_1_15_temp))
        bra_nov_16_30 = np.hstack((bra_nov_16_30, nov_16_30_temp))
        bra_dec_1_15 = np.hstack((bra_dec_1_15, dec_1_15_temp))
        bra_dec_16_31 = np.hstack((bra_dec_16_31, dec_16_31_temp)) 
        bra_jan_1_15 = np.hstack((bra_jan_1_15, jan_1_15_temp))
        bra_jan_16_31 = np.hstack((bra_jan_16_31, jan_16_31_temp))       
        bra_feb_1_15 = np.hstack((bra_feb_1_15, feb_1_15_temp))
        bra_feb_16_28 = np.hstack((bra_feb_16_28, feb_16_28_temp))      
        bra_mar_1_15 = np.hstack((bra_mar_1_15, mar_1_15_temp))
        bra_mar_16_31 = np.hstack((bra_mar_16_31, mar_16_31_temp))  
        bra_apr_1_15 = np.hstack((bra_apr_1_15, apr_1_15_temp))
        bra_apr_16_30 = np.hstack((bra_apr_16_30, apr_16_30_temp))    
        bra_may_1_15 = np.hstack((bra_may_1_15, may_1_15_temp))
        bra_may_16_31 = np.hstack((bra_may_16_31, may_16_31_temp))  
    
#%% Join all and plot hovmoller like
bra_fulyearsmeans = np.vstack((bra_aug_1_15, bra_aug_16_31, bra_sep_1_15, bra_sep_16_30,
                               bra_oct_1_15, bra_oct_16_31, bra_nov_1_15, bra_nov_16_30,
                               bra_dec_1_15, bra_dec_16_31, bra_jan_1_15, bra_jan_16_31,
                               bra_feb_1_15, bra_feb_16_28, bra_mar_1_15, bra_mar_16_31,
                               bra_apr_1_15, bra_apr_16_30, bra_may_1_15, bra_may_16_31))
bra_fulyearsmeans_t = np.transpose(bra_fulyearsmeans)
# Test for linear trend
#stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_aug_1_15)][1:], bra_aug_1_15[~np.isnan(bra_aug_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_aug_16_31)][1:], bra_aug_16_31[~np.isnan(bra_aug_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_sep_1_15)][1:], bra_sep_1_15[~np.isnan(bra_sep_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_sep_16_30)][1:], bra_sep_16_30[~np.isnan(bra_sep_16_30)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_oct_1_15)][1:], bra_oct_1_15[~np.isnan(bra_oct_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_oct_16_31)][1:], bra_oct_16_31[~np.isnan(bra_oct_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_nov_1_15)][1:], bra_nov_1_15[~np.isnan(bra_nov_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_nov_16_30)][1:], bra_nov_16_30[~np.isnan(bra_nov_16_30)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_dec_1_15)][1:], bra_dec_1_15[~np.isnan(bra_dec_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_dec_16_31)][1:], bra_dec_16_31[~np.isnan(bra_dec_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_jan_1_15)][1:], bra_jan_1_15[~np.isnan(bra_jan_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_jan_16_31)][1:], bra_jan_16_31[~np.isnan(bra_jan_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_feb_1_15)][1:], bra_feb_1_15[~np.isnan(bra_feb_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_feb_16_28)][1:], bra_feb_16_28[~np.isnan(bra_feb_16_28)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_mar_1_15)][1:], bra_mar_1_15[~np.isnan(bra_mar_1_15)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_mar_16_31)][1:], bra_mar_16_31[~np.isnan(bra_mar_16_31)][1:])
stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_apr_1_15)][1:], bra_apr_1_15[~np.isnan(bra_apr_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_apr_16_30)][1:], bra_apr_16_30[~np.isnan(bra_apr_16_30)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_may_1_15)][1:], bra_may_1_15[~np.isnan(bra_may_1_15)][1:])
#stats.linregress(np.arange(1998, 2022)[~np.isnan(bra_may_16_31)][1:], bra_may_16_31[~np.isnan(bra_may_16_31)][1:])







fig, ax = plt.subplots(figsize=(12, 6))
plt.pcolor(bra_fulyearsmeans_t, cmap=plt.cm.turbo, vmin = 0, vmax=2.5)
plt.yticks(ticks=np.arange(.5, 24.5), labels=['97-98', '98-99', '99-00', '00-01', '01-02',
                                              '02-03', '03-04', '04-05', '05-06', '06-07',
                                              '07-08', '08-09', '09-10', '10-11', '11-12',
                                              '12-13', '13-14', '14-15', '15-16', '16-17',
                                              '17-18', '18-19', '19-20', '20-21'], fontsize=12)
plt.gca().invert_yaxis()
plt.xticks(ticks=np.arange(1,21,2), labels=['AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY'])
plt.axvline(2, c='k', linewidth=1, alpha=0.5)
plt.axvline(4, c='k', linewidth=1, alpha=0.5)
plt.axvline(6, c='k', linewidth=1, alpha=0.5)
plt.axvline(8, c='k', linewidth=1, alpha=0.5)
plt.axvline(10, c='k', linewidth=1, alpha=0.5)
plt.axvline(12, c='k', linewidth=1, alpha=0.5)
plt.axvline(14, c='k', linewidth=1, alpha=0.5)
plt.axvline(16, c='k', linewidth=1, alpha=0.5)
plt.axvline(18, c='k', linewidth=1, alpha=0.5)
cbar = plt.colorbar()
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clusters_plots\\bransfield_15dayshovmoller.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()    
#%% Calculate Trends
# WEDDELL
stats.linregress(np.arange(1999, 2022), wed_chl_init_8day[1:])
stats.linregress(np.arange(1999, 2022), wed_chl_init_15day[1:])
stats.linregress(np.arange(1999, 2022), wed_chl_peak_8day[1:]) # **
stats.linregress(np.arange(1999, 2022), wed_chl_peak_15day[1:])
stats.linregress(np.arange(1999, 2022), wed_chl_term_8day[1:])
stats.linregress(np.arange(1999, 2022), wed_chl_term_15day[1:])
stats.linregress(np.arange(1999, 2022), wed_chl_max_8dayp[1:])
stats.linregress(np.arange(1999, 2022), wed_chl_max_15day[1:])
stats.linregress(np.arange(1999, 2022), wed_chl_cumulativesum_8day[1:])
stats.linregress(np.arange(1999, 2022), wed_chl_cumulativesum_15day[1:])
# GERLACHE
stats.linregress(np.arange(1999, 2022), ger_chl_init_8day[1:])
stats.linregress(np.arange(1999, 2022), ger_chl_init_15day[1:])
stats.linregress(np.arange(1999, 2022), ger_chl_peak_8day[1:])
stats.linregress(np.arange(1999, 2022), ger_chl_peak_15day[1:])
stats.linregress(np.arange(1999, 2022), ger_chl_term_8day[1:])
stats.linregress(np.arange(1999, 2022), ger_chl_term_15day[1:])
stats.linregress(np.arange(1999, 2022), ger_chl_max_8dayp[1:]) #*
stats.linregress(np.arange(1999, 2022), ger_chl_max_15day[1:])
stats.linregress(np.arange(1999, 2022), ger_chl_cumulativesum_8day[1:])
stats.linregress(np.arange(1999, 2022), ger_chl_cumulativesum_15day[1:])
# OCEANIC
stats.linregress(np.arange(1999, 2022), oce_chl_init_8day[1:]) # *
stats.linregress(np.arange(1999, 2022), oce_chl_init_15day[1:])
stats.linregress(np.arange(1999, 2022), oce_chl_peak_8day[1:])
stats.linregress(np.arange(1999, 2022), oce_chl_peak_15day[1:])
stats.linregress(np.arange(1999, 2022), oce_chl_term_8day[1:]) # **
stats.linregress(np.arange(1999, 2022), oce_chl_term_15day[1:])
stats.linregress(np.arange(1999, 2022), oce_chl_max_8dayp[1:])
stats.linregress(np.arange(1999, 2022), oce_chl_max_15day[1:])
stats.linregress(np.arange(1999, 2022), oce_chl_cumulativesum_8day[1:])
stats.linregress(np.arange(1999, 2022), oce_chl_cumulativesum_15day[1:])
# BRANSFIELD
stats.linregress(np.arange(1999, 2022), bra_chl_init_8day[1:])
stats.linregress(np.arange(1999, 2022), bra_chl_init_15day[1:])
stats.linregress(np.arange(1999, 2022), bra_chl_peak_8day[1:])
stats.linregress(np.arange(1999, 2022), bra_chl_peak_15day[1:])
stats.linregress(np.arange(1999, 2022), bra_chl_term_8day[1:])
stats.linregress(np.arange(1999, 2022), bra_chl_term_15day[1:])
stats.linregress(np.arange(1999, 2022), bra_chl_max_8dayp[1:]) # **
stats.linregress(np.arange(1999, 2022), bra_chl_max_15day[1:])
stats.linregress(np.arange(1999, 2022), bra_chl_cumulativesum_8day[1:]) # ***
stats.linregress(np.arange(1999, 2022), bra_chl_cumulativesum_15day[1:])



#%%
#%% Plot
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
sns.heatmap([wed_chl_init_8day[1:], wed_chl_peak_8day[1:], wed_chl_term_8day[1:],
             ger_chl_init_8day[1:], ger_chl_peak_8day[1:], ger_chl_term_8day[1:],
             bra_chl_init_8day[1:], bra_chl_peak_8day[1:], bra_chl_term_8day[1:],
             oce_chl_init_8day[1:], oce_chl_peak_8day[1:], oce_chl_term_8day[1:]], square=True, annot=True,
            vmin=1, vmax=38, cmap=cmap_new, cbar_kws={"fraction":0.025, "pad":0.05})
plt.yticks(ticks=np.arange(0.5, 12), labels = ['INIT', 'PEAK', 'TERM',
                                               'INIT', 'PEAK', 'TERM',
                                               'INIT', 'PEAK', 'TERM',
                                               'INIT', 'PEAK', 'TERM'], rotation = 360)
plt.xticks(ticks=np.arange(0.5, 23), labels = ['99', '00', '01', '02', '03', '04',
                                                  '05', '06', '07', '08', '09', '10',
                                                  '11', '12', '13', '14', '15', '16',
                                                  '17', '18', '19', '20', '21'])
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\phenologycomparison_percluster.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()





#stats.linregress(np.arange(1999, 2022), wed_may_1_15[1:])































#%% Separar para o cluster 3 (Oceanic)
oceanic_cluster = chl[clusters == 3,:]
oceanic_cluster = np.nanmean(oceanic_cluster,0)
np.nanmedian(oceanic_cluster)
np.nanmax(oceanic_cluster)
np.nanmin(oceanic_cluster)
np.nanstd(oceanic_cluster)*3
oceanic_cluster = np.where(oceanic_cluster > np.nanmedian(oceanic_cluster)-np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
oceanic_cluster = np.where(oceanic_cluster < np.nanmedian(oceanic_cluster)+np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
# Convert to pandas series
pixel_alldata_series = pd.Series(oceanic_cluster, index=time_date)
# Convert to monthly data
pixel_alldata_series_monthly = pixel_alldata_series.resample('M').mean()
for i in np.arange(1998, 2022):
    yeartemp_aug = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 8)]    
    yeartemp_sep = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = oceanic_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = oceanic_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = oceanic_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = oceanic_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_may = oceanic_cluster[(time_date_years == i) & (time_date_months == 5)]
    if i == 1998:
        years_aug_19982021 = np.nanmean(yeartemp_aug)
        years_sep_19982021 = np.nanmean(yeartemp_sep)
        years_oct_19982021 = np.nanmean(yeartemp_oct)
        years_nov_19982021 = np.nanmean(yeartemp_nov)
        years_dec_19982021 = np.nanmean(yeartemp_dec)
        years_jan_19982021 = np.nanmean(yeartemp_jan)
        years_feb_19982021 = np.nanmean(yeartemp_feb)
        years_mar_19982021 = np.nanmean(yeartemp_mar)
        years_apr_19982021 = np.nanmean(yeartemp_apr)
        years_may_19982021 = np.nanmean(yeartemp_may)
    else:
        years_aug_19982021 = np.hstack((years_aug_19982021, np.nanmean(yeartemp_aug)))
        years_sep_19982021 = np.hstack((years_sep_19982021, np.nanmean(yeartemp_sep)))
        years_oct_19982021 = np.hstack((years_oct_19982021, np.nanmean(yeartemp_oct)))
        years_nov_19982021 = np.hstack((years_nov_19982021, np.nanmean(yeartemp_nov)))
        years_dec_19982021 = np.hstack((years_dec_19982021, np.nanmean(yeartemp_dec)))
        years_jan_19982021 = np.hstack((years_jan_19982021, np.nanmean(yeartemp_jan)))
        years_feb_19982021 = np.hstack((years_feb_19982021, np.nanmean(yeartemp_feb)))
        years_mar_19982021 = np.hstack((years_mar_19982021, np.nanmean(yeartemp_mar)))
        years_apr_19982021 = np.hstack((years_apr_19982021, np.nanmean(yeartemp_apr)))
        years_may_19982021 = np.hstack((years_may_19982021, np.nanmean(yeartemp_may)))

yearlycycle_oceanic_19982021 = np.vstack((years_aug_19982021, years_sep_19982021, years_oct_19982021,
                                          years_nov_19982021, years_dec_19982021, years_jan_19982021,
                                          years_feb_19982021, years_mar_19982021, years_apr_19982021,
                                          years_may_19982021))
months_inds = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
# Calculate fenology
for i in range(0,24):
    yearlycycletemp = yearlycycle_oceanic_19982021[:,i]
    # First month with Chl-a
    chl_init_temp = months_inds[np.argmax(~np.isnan(yearlycycletemp))]
    # Month with peak Chl-a
    chl_peak_temp = months_inds[np.nanargmax(yearlycycletemp)]
    # Last month with Chl-a
    chl_term_temp = months_inds[9-np.argmax(~np.isnan(yearlycycletemp[::-1]))]    
    # Max Chl-a
    chl_max_temp = np.nanmax(yearlycycle_oceanic_19982021[:,i])
    # Chl-a Area
    chl_area_temp = integrate.simps(yearlycycle_gerlache_19982021[3:7,i])
    if i == 0:
        chl_init_all_gerlache = chl_init_temp
        chl_peak_all_gerlache = chl_peak_temp
        chl_term_all_gerlache = chl_term_temp
        chl_max_all_gerlache = chl_max_temp
        chl_area_all_gerlache = chl_area_temp
    else:
        chl_init_all_gerlache = np.hstack((chl_init_all_gerlache, chl_init_temp))
        chl_peak_all_gerlache = np.hstack((chl_peak_all_gerlache, chl_peak_temp))
        chl_term_all_gerlache = np.hstack((chl_term_all_gerlache, chl_term_temp))
        chl_max_all_gerlache = np.hstack((chl_max_all_gerlache, chl_max_temp))
        chl_area_all_gerlache = np.hstack((chl_area_all_gerlache, chl_area_temp))
#%% Separar para o cluster 4 (Bransfield)
bransfield_cluster = chl[clusters == 4,:]
bransfield_cluster = np.nanmean(bransfield_cluster,0)
np.nanmedian(bransfield_cluster)
np.nanmax(bransfield_cluster)
np.nanmin(bransfield_cluster)
np.nanstd(bransfield_cluster)*3
bransfield_cluster = np.where(bransfield_cluster > np.nanmedian(bransfield_cluster)-np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
bransfield_cluster = np.where(bransfield_cluster < np.nanmedian(bransfield_cluster)+np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
# Convert to pandas series
pixel_alldata_series = pd.Series(bransfield_cluster, index=time_date)
# Convert to monthly data
pixel_alldata_series_monthly = pixel_alldata_series.resample('M').mean()
for i in np.arange(1998, 2022):
    yeartemp_aug = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 8)]    
    yeartemp_sep = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = bransfield_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = bransfield_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = bransfield_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = bransfield_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_may = bransfield_cluster[(time_date_years == i) & (time_date_months == 5)]
    if i == 1998:
        years_aug_19982021 = np.nanmean(yeartemp_aug)
        years_sep_19982021 = np.nanmean(yeartemp_sep)
        years_oct_19982021 = np.nanmean(yeartemp_oct)
        years_nov_19982021 = np.nanmean(yeartemp_nov)
        years_dec_19982021 = np.nanmean(yeartemp_dec)
        years_jan_19982021 = np.nanmean(yeartemp_jan)
        years_feb_19982021 = np.nanmean(yeartemp_feb)
        years_mar_19982021 = np.nanmean(yeartemp_mar)
        years_apr_19982021 = np.nanmean(yeartemp_apr)
        years_may_19982021 = np.nanmean(yeartemp_may)
    else:
        years_aug_19982021 = np.hstack((years_aug_19982021, np.nanmean(yeartemp_aug)))
        years_sep_19982021 = np.hstack((years_sep_19982021, np.nanmean(yeartemp_sep)))
        years_oct_19982021 = np.hstack((years_oct_19982021, np.nanmean(yeartemp_oct)))
        years_nov_19982021 = np.hstack((years_nov_19982021, np.nanmean(yeartemp_nov)))
        years_dec_19982021 = np.hstack((years_dec_19982021, np.nanmean(yeartemp_dec)))
        years_jan_19982021 = np.hstack((years_jan_19982021, np.nanmean(yeartemp_jan)))
        years_feb_19982021 = np.hstack((years_feb_19982021, np.nanmean(yeartemp_feb)))
        years_mar_19982021 = np.hstack((years_mar_19982021, np.nanmean(yeartemp_mar)))
        years_apr_19982021 = np.hstack((years_apr_19982021, np.nanmean(yeartemp_apr)))
        years_may_19982021 = np.hstack((years_may_19982021, np.nanmean(yeartemp_may)))

yearlycycle_bransfield_19982021 = np.vstack((years_aug_19982021, years_sep_19982021, years_oct_19982021,
                                          years_nov_19982021, years_dec_19982021, years_jan_19982021,
                                          years_feb_19982021, years_mar_19982021, years_apr_19982021,
                                          years_may_19982021))
months_inds = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
# Calculate fenology
for i in range(0,24):
    yearlycycletemp = yearlycycle_bransfield_19982021[:,i]
    # First month with Chl-a
    chl_init_temp = months_inds[np.argmax(~np.isnan(yearlycycletemp))]
    # Month with peak Chl-a
    chl_peak_temp = months_inds[np.nanargmax(yearlycycletemp)]
    # Last month with Chl-a
    chl_term_temp = months_inds[9-np.argmax(~np.isnan(yearlycycletemp[::-1]))]    
    # Max Chl-a
    chl_max_temp = np.nanmax(yearlycycle_bransfield_19982021[:,i])
    # Chl-a Area
    chl_area_temp = integrate.simps(yearlycycle_bransfield_19982021[3:7,i])
    if i == 0:
        chl_init_all_bransfield = chl_init_temp
        chl_peak_all_bransfield = chl_peak_temp
        chl_term_all_bransfield = chl_term_temp
        chl_max_all_bransfield = chl_max_temp
        chl_area_all_bransfield = chl_area_temp
    else:
        chl_init_all_bransfield = np.hstack((chl_init_all_bransfield, chl_init_temp))
        chl_peak_all_bransfield = np.hstack((chl_peak_all_bransfield, chl_peak_temp))
        chl_term_all_bransfield = np.hstack((chl_term_all_bransfield, chl_term_temp))
        chl_max_all_bransfield = np.hstack((chl_max_all_bransfield, chl_max_temp))
        chl_area_all_bransfield = np.hstack((chl_area_all_bransfield, chl_area_temp))
#%% Separar para o cluster 3 (Oceanic)
oceanic_cluster = chl[clusters == 3,:]
oceanic_cluster = np.nanmean(oceanic_cluster,0)
np.nanmedian(oceanic_cluster)
np.nanmax(oceanic_cluster)
np.nanmin(oceanic_cluster)
np.nanstd(oceanic_cluster)*3
oceanic_cluster = np.where(oceanic_cluster > np.nanmedian(oceanic_cluster)-np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
oceanic_cluster = np.where(oceanic_cluster < np.nanmedian(oceanic_cluster)+np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
# Convert to pandas series
pixel_alldata_series = pd.Series(oceanic_cluster, index=time_date)
# Convert to monthly data
pixel_alldata_series_monthly = pixel_alldata_series.resample('M').mean()
for i in np.arange(1998, 2022):
    yeartemp_aug = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 8)]    
    yeartemp_sep = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = oceanic_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = oceanic_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = oceanic_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = oceanic_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_may = oceanic_cluster[(time_date_years == i) & (time_date_months == 5)]
    if i == 1998:
        years_aug_19982021 = np.nanmean(yeartemp_aug)
        years_sep_19982021 = np.nanmean(yeartemp_sep)
        years_oct_19982021 = np.nanmean(yeartemp_oct)
        years_nov_19982021 = np.nanmean(yeartemp_nov)
        years_dec_19982021 = np.nanmean(yeartemp_dec)
        years_jan_19982021 = np.nanmean(yeartemp_jan)
        years_feb_19982021 = np.nanmean(yeartemp_feb)
        years_mar_19982021 = np.nanmean(yeartemp_mar)
        years_apr_19982021 = np.nanmean(yeartemp_apr)
        years_may_19982021 = np.nanmean(yeartemp_may)
    else:
        years_aug_19982021 = np.hstack((years_aug_19982021, np.nanmean(yeartemp_aug)))
        years_sep_19982021 = np.hstack((years_sep_19982021, np.nanmean(yeartemp_sep)))
        years_oct_19982021 = np.hstack((years_oct_19982021, np.nanmean(yeartemp_oct)))
        years_nov_19982021 = np.hstack((years_nov_19982021, np.nanmean(yeartemp_nov)))
        years_dec_19982021 = np.hstack((years_dec_19982021, np.nanmean(yeartemp_dec)))
        years_jan_19982021 = np.hstack((years_jan_19982021, np.nanmean(yeartemp_jan)))
        years_feb_19982021 = np.hstack((years_feb_19982021, np.nanmean(yeartemp_feb)))
        years_mar_19982021 = np.hstack((years_mar_19982021, np.nanmean(yeartemp_mar)))
        years_apr_19982021 = np.hstack((years_apr_19982021, np.nanmean(yeartemp_apr)))
        years_may_19982021 = np.hstack((years_may_19982021, np.nanmean(yeartemp_may)))

yearlycycle_oceanic_19982021 = np.vstack((years_aug_19982021, years_sep_19982021, years_oct_19982021,
                                          years_nov_19982021, years_dec_19982021, years_jan_19982021,
                                          years_feb_19982021, years_mar_19982021, years_apr_19982021,
                                          years_may_19982021))
months_inds = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
# Calculate fenology
for i in range(0,24):
    yearlycycletemp = yearlycycle_oceanic_19982021[:,i]
    # First month with Chl-a
    chl_init_temp = months_inds[np.argmax(~np.isnan(yearlycycletemp))]
    # Month with peak Chl-a
    chl_peak_temp = months_inds[np.nanargmax(yearlycycletemp)]
    # Last month with Chl-a
    chl_term_temp = months_inds[9-np.argmax(~np.isnan(yearlycycletemp[::-1]))]    
    # Max Chl-a
    chl_max_temp = np.nanmax(yearlycycle_oceanic_19982021[:,i])
    # Chl-a Area
    chl_area_temp = integrate.simps(yearlycycle_oceanic_19982021[3:7,i])
    if i == 0:
        chl_init_all_oceanic = chl_init_temp
        chl_peak_all_oceanic = chl_peak_temp
        chl_term_all_oceanic = chl_term_temp
        chl_max_all_oceanic = chl_max_temp
        chl_area_all_oceanic = chl_area_temp
    else:
        chl_init_all_oceanic = np.hstack((chl_init_all_oceanic, chl_init_temp))
        chl_peak_all_oceanic = np.hstack((chl_peak_all_oceanic, chl_peak_temp))
        chl_term_all_oceanic = np.hstack((chl_term_all_oceanic, chl_term_temp))
        chl_max_all_oceanic = np.hstack((chl_max_all_oceanic, chl_max_temp))
        chl_area_all_oceanic = np.hstack((chl_area_all_oceanic, chl_area_temp))
#%% Plot
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
sns.heatmap([chl_init_all_gerlache[1:], chl_peak_all_gerlache[1:], chl_term_all_gerlache[1:],
             chl_init_all_gerlache[1:], chl_peak_all_gerlache[1:], chl_term_all_gerlache[1:],
             chl_init_all_bransfield[1:], chl_peak_all_bransfield[1:], chl_term_all_bransfield[1:],
             chl_init_all_oceanic[1:], chl_peak_all_oceanic[1:], chl_term_all_oceanic[1:]], square=True, annot=True,
            vmin=1, vmax=12, cmap=cmap_new, cbar_kws={"fraction":0.025, "pad":0.05})
plt.yticks(ticks=np.arange(0.5, 12), labels = ['INIT', 'PEAK', 'TERM',
                                               'INIT', 'PEAK', 'TERM',
                                               'INIT', 'PEAK', 'TERM',
                                               'INIT', 'PEAK', 'TERM'], rotation = 360)
plt.xticks(ticks=np.arange(0.5, 23), labels = ['99', '00', '01', '02', '03', '04',
                                                  '05', '06', '07', '08', '09', '10',
                                                  '11', '12', '13', '14', '15', '16',
                                                  '17', '18', '19', '20', '21'])
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\phenologycomparison_percluster.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Test if there are temporal trends for each metric
a = chl_peak_all_oceanic+3
a[a==13] = 1
a[a==14] = 2
a[a==15] = 3
a
stats.linregress(np.arange(1999, 2022)[~np.isnan(a[1:])], a[1:][~np.isnan(a[1:])])

#%% Plot 
plt.scatter(chl_max_all_gerlache[1:], np.full((23), 4), c=np.arange(1999, 2022), cmap=plt.cm.Blues, edgecolors='k', s=100)
plt.scatter(chl_max_all_gerlache[1:], np.full((23), 3), c=np.arange(1999, 2022), cmap=plt.cm.Reds, edgecolors='k', s=100)
plt.scatter(chl_max_all_bransfield[1:], np.full((23), 2), c=np.arange(1999, 2022), cmap=plt.cm.Purples, edgecolors='k', s=100)
plt.scatter(chl_max_all_oceanic[1:], np.full((23), 1), c=np.arange(1999, 2022), cmap=plt.cm.Yellows, edgecolors='k', s=100)


stats.linregress(np.arange(1999, 2022)[~np.isnan(chl_area_all_gerlache[1:])], chl_area_all_gerlache[1:][~np.isnan(chl_area_all_gerlache[1:])])


#%%
# Test outliers for gerlache_cluster
#weddell_cluster = is_outlier(weddell_cluster, thresh=2.5)
#weddell_cluster = median_filter(weddell_cluster, size=30)
#t = weddell_cluster*np.nan
#m=30
#for i in np.arange(len(t)):
#    t[i] = np.nanmean(weddell_cluster[np.nanmax([0,i-m]):(i+1)])
#weddell_cluster = t
#plt.plot(t, T)
#N = y_test[:,0] - T
#plt.figure()
#plt.plot(t,N)
### Divide data per year (July to June)
## 1997-1998
#idx_year_chl_start = np.argwhere((time_date_years == 1997) & (time_date_months == 7)).ravel()
idx_year_chl_end = np.argwhere((time_date_years == 1998) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_19971998_daily = weddell_cluster[:idx_year_chl_end]
time_date19971998 = time_date[:idx_year_chl_end]
weddell_cluster_19971998 = pd.Series(weddell_cluster[:idx_year_chl_end], index=time_date[:idx_year_chl_end])
weddell_cluster_19971998_monthly = weddell_cluster_19971998.resample('M').mean()
weddell_cluster_19971998_monthly = np.hstack((np.nan, np.nan, weddell_cluster_19971998_monthly.values))
## 1998-1999
idx_year_chl_start = np.argwhere((time_date_years == 1998) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 1999) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_19981999_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date19981999 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_19981999 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_19981999_monthly = weddell_cluster_19981999.resample('M').mean()
weddell_cluster_19981999_monthly = weddell_cluster_19981999_monthly.values
## 1999-2000
idx_year_chl_start = np.argwhere((time_date_years == 1999) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2000) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_19992000_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date19992000 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_19992000 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_19992000_monthly = weddell_cluster_19992000.resample('M').mean()
weddell_cluster_19992000_monthly = weddell_cluster_19992000_monthly.values
## 2000-2001
idx_year_chl_start = np.argwhere((time_date_years == 2000) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2001) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20002001_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20002001 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20002001 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20002001_monthly = weddell_cluster_20002001.resample('M').mean()
weddell_cluster_20002001_monthly = weddell_cluster_20002001_monthly.values
## 2001-2002
idx_year_chl_start = np.argwhere((time_date_years == 2001) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2002) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20012002_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20012002 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20012002 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20012002_monthly = weddell_cluster_20012002.resample('M').mean()
weddell_cluster_20012002_monthly = weddell_cluster_20012002_monthly.values
## 2002-2003
idx_year_chl_start = np.argwhere((time_date_years == 2002) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2003) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20022003_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20022003 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20022003 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20022003_monthly = weddell_cluster_20022003.resample('M').mean()
weddell_cluster_20022003_monthly = weddell_cluster_20022003_monthly.values
## 2003-2004
idx_year_chl_start = np.argwhere((time_date_years == 2003) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2004) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20032004_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20032004 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20032004 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20032004_monthly = weddell_cluster_20032004.resample('M').mean()
weddell_cluster_20032004_monthly = weddell_cluster_20032004_monthly.values
## 2004-2005
idx_year_chl_start = np.argwhere((time_date_years == 2004) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2005) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20042005_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20042005 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20042005 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20042005_monthly = weddell_cluster_20042005.resample('M').mean()
weddell_cluster_20042005_monthly = weddell_cluster_20042005_monthly.values
## 2005-2006
idx_year_chl_start = np.argwhere((time_date_years == 2005) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2006) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20052006_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20052006 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20052006 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20052006_monthly = weddell_cluster_20052006.resample('M').mean()
weddell_cluster_20052006_monthly = weddell_cluster_20052006_monthly.values
## 2006-2007
idx_year_chl_start = np.argwhere((time_date_years == 2006) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2007) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20062007_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20062007 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20062007 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20062007_monthly = weddell_cluster_20062007.resample('M').mean()
weddell_cluster_20062007_monthly = weddell_cluster_20062007_monthly.values
## 2007-2008
idx_year_chl_start = np.argwhere((time_date_years == 2007) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2008) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20072008_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20072008 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20072008 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20072008_monthly = weddell_cluster_20072008.resample('M').mean()
weddell_cluster_20072008_monthly = weddell_cluster_20072008_monthly.values
## 2008-2009
idx_year_chl_start = np.argwhere((time_date_years == 2008) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2009) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20082009_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20082009 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20082009 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20082009_monthly = weddell_cluster_20082009.resample('M').mean()
weddell_cluster_20082009_monthly = weddell_cluster_20082009_monthly.values
## 2009-2010
idx_year_chl_start = np.argwhere((time_date_years == 2009) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2010) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20092010_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20092010 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20092010 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20092010_monthly = weddell_cluster_20092010.resample('M').mean()
weddell_cluster_20092010_monthly = weddell_cluster_20092010_monthly.values
## 2010-2011
idx_year_chl_start = np.argwhere((time_date_years == 2010) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2011) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20102011_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20102011 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20102011 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20102011_monthly = weddell_cluster_20102011.resample('M').mean()
weddell_cluster_20102011_monthly = weddell_cluster_20102011_monthly.values
## 2011-2012
idx_year_chl_start = np.argwhere((time_date_years == 2011) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2012) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20112012_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20112012 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20112012 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20112012_monthly = weddell_cluster_20112012.resample('M').mean()
weddell_cluster_20112012_monthly = weddell_cluster_20112012_monthly.values
## 2012-2013
idx_year_chl_start = np.argwhere((time_date_years == 2012) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2013) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20122013_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20122013 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20122013 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20122013_monthly = weddell_cluster_20122013.resample('M').mean()
weddell_cluster_20122013_monthly = weddell_cluster_20122013_monthly.values
## 2013-2014
idx_year_chl_start = np.argwhere((time_date_years == 2013) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2014) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20132014_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20132014 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20132014 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20132014_monthly = weddell_cluster_20132014.resample('M').mean()
weddell_cluster_20132014_monthly = weddell_cluster_20132014_monthly.values
## 2014-2015
idx_year_chl_start = np.argwhere((time_date_years == 2014) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2015) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20142015_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20142015 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20142015 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20142015_monthly = weddell_cluster_20142015.resample('M').mean()
weddell_cluster_20142015_monthly = weddell_cluster_20142015_monthly.values
## 2015-2016
idx_year_chl_start = np.argwhere((time_date_years == 2015) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2016) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20152016_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20152016 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20152016 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20152016_monthly = weddell_cluster_20152016.resample('M').mean()
weddell_cluster_20152016_monthly = weddell_cluster_20152016_monthly.values
## 2016-2017
idx_year_chl_start = np.argwhere((time_date_years == 2016) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2017) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20162017_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20162017 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20162017 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20162017_monthly = weddell_cluster_20162017.resample('M').mean()
weddell_cluster_20162017_monthly = weddell_cluster_20162017_monthly.values
## 2017-2018
idx_year_chl_start = np.argwhere((time_date_years == 2017) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2018) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20172018_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20172018 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20172018 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20172018_monthly = weddell_cluster_20172018.resample('M').mean()
weddell_cluster_20172018_monthly = weddell_cluster_20172018_monthly.values
## 2018-2019
idx_year_chl_start = np.argwhere((time_date_years == 2018) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2019) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20182019_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20182019 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20182019 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20182019_monthly = weddell_cluster_20182019.resample('M').mean()
weddell_cluster_20182019_monthly = weddell_cluster_20182019_monthly.values
## 2019-2020
idx_year_chl_start = np.argwhere((time_date_years == 2019) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2020) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20192020_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20192020 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20192020 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20192020_monthly = weddell_cluster_20192020.resample('M').mean()
weddell_cluster_20192020_monthly = weddell_cluster_20192020_monthly.values
## 2020-2021
idx_year_chl_start = np.argwhere((time_date_years == 2020) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2021) & (time_date_months == 6)).ravel()[-1]+1
weddell_cluster_20202021_daily = weddell_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20202021 = time_date[idx_year_chl_start:idx_year_chl_end]
weddell_cluster_20202021 = pd.Series(weddell_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
weddell_cluster_20202021_monthly = weddell_cluster_20202021.resample('M').mean()
weddell_cluster_20202021_monthly = weddell_cluster_20202021_monthly.values
## 1997-2021 Mean cycle
weddell_cluster_19972021_july =  np.nanmean(weddell_cluster[time_date_months == 7])
weddell_cluster_19972021_august =  np.nanmean(weddell_cluster[time_date_months == 8])
weddell_cluster_19972021_september =  np.nanmean(weddell_cluster[time_date_months == 9])
weddell_cluster_19972021_october =  np.nanmean(weddell_cluster[time_date_months == 10])
weddell_cluster_19972021_november =  np.nanmean(weddell_cluster[time_date_months == 11])
weddell_cluster_19972021_december =  np.nanmean(weddell_cluster[time_date_months == 12])
weddell_cluster_19972021_january =  np.nanmean(weddell_cluster[time_date_months == 1])
weddell_cluster_19972021_february =  np.nanmean(weddell_cluster[time_date_months == 2])
weddell_cluster_19972021_march =  np.nanmean(weddell_cluster[time_date_months == 3])
weddell_cluster_19972021_april =  np.nanmean(weddell_cluster[time_date_months == 4])
weddell_cluster_19972021_may =  np.nanmean(weddell_cluster[time_date_months == 5])
weddell_cluster_19972021_june =  np.nanmean(weddell_cluster[time_date_months == 6])
weddell_cluster_19972021_monthly = np.hstack((weddell_cluster_19972021_july,
                                              weddell_cluster_19972021_august,
                                              weddell_cluster_19972021_september,
                                              weddell_cluster_19972021_october,
                                              weddell_cluster_19972021_november,
                                              weddell_cluster_19972021_december,
                                              weddell_cluster_19972021_january,
                                              weddell_cluster_19972021_february,
                                              weddell_cluster_19972021_march,
                                              weddell_cluster_19972021_april,
                                              weddell_cluster_19972021_may,
                                              weddell_cluster_19972021_june
                                              ))
#weddel_cluster_allcicles_1D = np.hstack([weddell_cluster_19971998_monthly,
#                                      weddell_cluster_19981999_monthly,
#                                      weddell_cluster_19992000_monthly,
#                                      weddell_cluster_20002001_monthly,
#                                      weddell_cluster_20012002_monthly,
#                                      weddell_cluster_20022003_monthly,
#                                      weddell_cluster_20032004_monthly,
#                                      weddell_cluster_20042005_monthly,
#                                      weddell_cluster_20052006_monthly,
#                                      weddell_cluster_20062007_monthly,
#                                      weddell_cluster_20072008_monthly,
#                                      weddell_cluster_20082009_monthly,
#                                      weddell_cluster_20092010_monthly,
#                                      weddell_cluster_20102011_monthly,
#                                      weddell_cluster_20112012_monthly,
#                                      weddell_cluster_20122013_monthly,
#                                      weddell_cluster_20132014_monthly,
#                                      weddell_cluster_20142015_monthly,
#                                      weddell_cluster_20152016_monthly,
#                                      weddell_cluster_20162017_monthly,
#                                      weddell_cluster_20172018_monthly,
#                                      weddell_cluster_20182019_monthly,
#                                      weddell_cluster_20192020_monthly,
#                                      weddell_cluster_20202021_monthly])


#transformer = HampelFilter(window_length=12, n_sigma=3, k=1.4826, return_bool=False)
#weddel_cluster_allcicles_1D_filtered = np.squeeze(transformer.fit_transform(weddel_cluster_allcicles_1D))
#weddell_cluster_19971998_monthly = weddel_cluster_allcicles_1D_filtered[:12]
#weddell_cluster_19981999_monthly = weddel_cluster_allcicles_1D_filtered[12:24]
#weddell_cluster_19992000_monthly = weddel_cluster_allcicles_1D_filtered[24:36]
#weddell_cluster_20002001_monthly = weddel_cluster_allcicles_1D_filtered[36:48]
#weddell_cluster_20012002_monthly = weddel_cluster_allcicles_1D_filtered[48:60]
#weddell_cluster_20022003_monthly = weddel_cluster_allcicles_1D_filtered[60:72]
#weddell_cluster_20032004_monthly = weddel_cluster_allcicles_1D_filtered[72:84]
#weddell_cluster_20042005_monthly = weddel_cluster_allcicles_1D_filtered[84:96]
#weddell_cluster_20052006_monthly = weddel_cluster_allcicles_1D_filtered[96:108]
#weddell_cluster_20062007_monthly = weddel_cluster_allcicles_1D_filtered[108:120]
#weddell_cluster_20072008_monthly = weddel_cluster_allcicles_1D_filtered[120:132]
#weddell_cluster_20082009_monthly = weddel_cluster_allcicles_1D_filtered[132:144]
#weddell_cluster_20092010_monthly = weddel_cluster_allcicles_1D_filtered[144:156]
#weddell_cluster_20102011_monthly = weddel_cluster_allcicles_1D_filtered[156:168]
#weddell_cluster_20112012_monthly = weddel_cluster_allcicles_1D_filtered[168:180]
#weddell_cluster_20122013_monthly = weddel_cluster_allcicles_1D_filtered[180:192]
#weddell_cluster_20132014_monthly = weddel_cluster_allcicles_1D_filtered[192:204]
#weddell_cluster_20142015_monthly = weddel_cluster_allcicles_1D_filtered[204:216]
#weddell_cluster_20152016_monthly = weddel_cluster_allcicles_1D_filtered[216:228]
#weddell_cluster_20162017_monthly = weddel_cluster_allcicles_1D_filtered[228:240]
#weddell_cluster_20172018_monthly = weddel_cluster_allcicles_1D_filtered[240:252]
#weddell_cluster_20182019_monthly = weddel_cluster_allcicles_1D_filtered[252:264]
#weddell_cluster_20192020_monthly = weddel_cluster_allcicles_1D_filtered[264:276]
#weddell_cluster_20202021_monthly = weddel_cluster_allcicles_1D_filtered[276:288]

# Join yearly cicles
weddel_cluster_allcicles = np.vstack([weddell_cluster_19971998_monthly,
                                      weddell_cluster_19981999_monthly,
                                      weddell_cluster_19992000_monthly,
                                      weddell_cluster_20002001_monthly,
                                      weddell_cluster_20012002_monthly,
                                      weddell_cluster_20022003_monthly,
                                      weddell_cluster_20032004_monthly,
                                      weddell_cluster_20042005_monthly,
                                      weddell_cluster_20052006_monthly,
                                      weddell_cluster_20062007_monthly,
                                      weddell_cluster_20072008_monthly,
                                      weddell_cluster_20082009_monthly,
                                      weddell_cluster_20092010_monthly,
                                      weddell_cluster_20102011_monthly,
                                      weddell_cluster_20112012_monthly,
                                      weddell_cluster_20122013_monthly,
                                      weddell_cluster_20132014_monthly,
                                      weddell_cluster_20142015_monthly,
                                      weddell_cluster_20152016_monthly,
                                      weddell_cluster_20162017_monthly,
                                      weddell_cluster_20172018_monthly,
                                      weddell_cluster_20182019_monthly,
                                      weddell_cluster_20192020_monthly,
                                      weddell_cluster_20202021_monthly])

weddell_cluster_19982005 = np.nanmean(weddel_cluster_allcicles[:8,:], axis=0)
weddell_cluster_20062014 = np.nanmean(weddel_cluster_allcicles[8:16,:], axis=0)
weddell_cluster_20152021 = np.nanmean(weddel_cluster_allcicles[16:,:], axis=0)
weddell_yearlycicles_p90 = np.nanpercentile(weddel_cluster_allcicles, 90, axis=0)
weddell_yearlycicles_p10 = np.nanpercentile(weddel_cluster_allcicles, 10, axis=0)
weddell_yearlycicles_std = np.nanstd(weddel_cluster_allcicles, axis=0)
weddell_cluster_mean19972021 = np.nanmean(weddel_cluster_allcicles, axis=0)
#%% Weddell Cluster Figure 1
f_cubic = interp1d(np.arange(3,11),weddell_cluster_mean19972021[2:10], kind='cubic')
xnew = np.linspace(3, 10, num=50, endpoint=True)
#f_cubic_p90 = interp1d(np.arange(3,11),weddell_yearlycicles_p90[2:10], kind='cubic')
#f_cubic_p10 = interp1d(np.arange(3,11),weddell_yearlycicles_p10[2:10], kind='cubic')
plt.figure()
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.plot(xnew, f_cubic(xnew), color = [43/256, 131/256, 186/256, 1], linewidth = 5, label='1997-2021')

plt.plot(np.arange(1,13),weddell_cluster_19982005, color = 'k', linewidth = 2, linestyle='--', label='1997-2005', alpha=0.6)
plt.plot(np.arange(1,13),weddell_cluster_20062014, color = 'k', linewidth = 2, linestyle=':', label='2005-2014', alpha=0.6)
plt.plot(np.arange(1,13),weddell_cluster_20152021, color = 'k', linewidth = 2, linestyle='-.', label='2015-2021', alpha=0.6)
plt.errorbar(np.arange(1,13),weddell_cluster_mean19972021, weddell_yearlycicles_std, linestyle='None', marker='None',
             color = [43/256, 131/256, 186/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_yearlycicles_p90, color =[43/256, 131/256, 186/256, 1], alpha=.5, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_yearlycicles_p10, color =[43/256, 131/256, 186/256, 1], alpha=.5, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021+weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), weddell_cluster_mean19972021, weddell_cluster_mean19972021-weddell_yearlycicles_std, color =[43/256, 131/256, 186/256, 1], alpha=.2, edgecolor = None)


plt.xticks([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
           ['Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr','May', 'Jun'], fontsize=12)
plt.xlim(3,10)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\Weddell_meancycle_errorbar.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 2 (Gerlache)
gerlache_cluster = chl[clusters == 2,:]
gerlache_cluster = np.nanmean(gerlache_cluster,0)

np.nanmedian(gerlache_cluster)
np.nanmax(gerlache_cluster)
np.nanmin(gerlache_cluster)
np.nanstd(gerlache_cluster)*3
gerlache_cluster = np.where(gerlache_cluster > np.nanmedian(gerlache_cluster)-np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
gerlache_cluster = np.where(gerlache_cluster < np.nanmedian(gerlache_cluster)+np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)

### Divide data per year (July to June)
## 1997-1998
#idx_year_chl_start = np.argwhere((time_date_years == 1997) & (time_date_months == 7)).ravel()
idx_year_chl_end = np.argwhere((time_date_years == 1998) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_19971998_daily = gerlache_cluster[:idx_year_chl_end]
time_date19971998 = time_date[:idx_year_chl_end]
gerlache_cluster_19971998 = pd.Series(gerlache_cluster[:idx_year_chl_end], index=time_date[:idx_year_chl_end])
gerlache_cluster_19971998_monthly = gerlache_cluster_19971998.resample('M').mean()
gerlache_cluster_19971998_monthly = np.hstack((np.nan, np.nan, gerlache_cluster_19971998_monthly.values))
## 1998-1999
idx_year_chl_start = np.argwhere((time_date_years == 1998) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 1999) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_19981999_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date19981999 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_19981999 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_19981999_monthly = gerlache_cluster_19981999.resample('M').mean()
gerlache_cluster_19981999_monthly = gerlache_cluster_19981999_monthly.values
## 1999-2000
idx_year_chl_start = np.argwhere((time_date_years == 1999) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2000) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_19992000_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date19992000 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_19992000 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_19992000_monthly = gerlache_cluster_19992000.resample('M').mean()
gerlache_cluster_19992000_monthly = gerlache_cluster_19992000_monthly.values
## 2000-2001
idx_year_chl_start = np.argwhere((time_date_years == 2000) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2001) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20002001_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20002001 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20002001 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20002001_monthly = gerlache_cluster_20002001.resample('M').mean()
gerlache_cluster_20002001_monthly = gerlache_cluster_20002001_monthly.values
## 2001-2002
idx_year_chl_start = np.argwhere((time_date_years == 2001) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2002) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20012002_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20012002 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20012002 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20012002_monthly = gerlache_cluster_20012002.resample('M').mean()
gerlache_cluster_20012002_monthly = gerlache_cluster_20012002_monthly.values
## 2002-2003
idx_year_chl_start = np.argwhere((time_date_years == 2002) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2003) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20022003_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20022003 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20022003 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20022003_monthly = gerlache_cluster_20022003.resample('M').mean()
gerlache_cluster_20022003_monthly = gerlache_cluster_20022003_monthly.values
## 2003-2004
idx_year_chl_start = np.argwhere((time_date_years == 2003) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2004) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20032004_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20032004 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20032004 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20032004_monthly = gerlache_cluster_20032004.resample('M').mean()
gerlache_cluster_20032004_monthly = gerlache_cluster_20032004_monthly.values
## 2004-2005
idx_year_chl_start = np.argwhere((time_date_years == 2004) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2005) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20042005_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20042005 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20042005 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20042005_monthly = gerlache_cluster_20042005.resample('M').mean()
gerlache_cluster_20042005_monthly = gerlache_cluster_20042005_monthly.values
## 2005-2006
idx_year_chl_start = np.argwhere((time_date_years == 2005) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2006) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20052006_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20052006 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20052006 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20052006_monthly = gerlache_cluster_20052006.resample('M').mean()
gerlache_cluster_20052006_monthly = gerlache_cluster_20052006_monthly.values
## 2006-2007
idx_year_chl_start = np.argwhere((time_date_years == 2006) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2007) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20062007_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20062007 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20062007 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20062007_monthly = gerlache_cluster_20062007.resample('M').mean()
gerlache_cluster_20062007_monthly = gerlache_cluster_20062007_monthly.values
## 2007-2008
idx_year_chl_start = np.argwhere((time_date_years == 2007) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2008) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20072008_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20072008 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20072008 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20072008_monthly = gerlache_cluster_20072008.resample('M').mean()
gerlache_cluster_20072008_monthly = gerlache_cluster_20072008_monthly.values
## 2008-2009
idx_year_chl_start = np.argwhere((time_date_years == 2008) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2009) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20082009_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20082009 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20082009 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20082009_monthly = gerlache_cluster_20082009.resample('M').mean()
gerlache_cluster_20082009_monthly = gerlache_cluster_20082009_monthly.values
## 2009-2010
idx_year_chl_start = np.argwhere((time_date_years == 2009) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2010) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20092010_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20092010 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20092010 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20092010_monthly = gerlache_cluster_20092010.resample('M').mean()
gerlache_cluster_20092010_monthly = gerlache_cluster_20092010_monthly.values
## 2010-2011
idx_year_chl_start = np.argwhere((time_date_years == 2010) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2011) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20102011_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20102011 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20102011 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20102011_monthly = gerlache_cluster_20102011.resample('M').mean()
gerlache_cluster_20102011_monthly = gerlache_cluster_20102011_monthly.values
## 2011-2012
idx_year_chl_start = np.argwhere((time_date_years == 2011) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2012) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20112012_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20112012 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20112012 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20112012_monthly = gerlache_cluster_20112012.resample('M').mean()
gerlache_cluster_20112012_monthly = gerlache_cluster_20112012_monthly.values
## 2012-2013
idx_year_chl_start = np.argwhere((time_date_years == 2012) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2013) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20122013_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20122013 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20122013 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20122013_monthly = gerlache_cluster_20122013.resample('M').mean()
gerlache_cluster_20122013_monthly = gerlache_cluster_20122013_monthly.values
## 2013-2014
idx_year_chl_start = np.argwhere((time_date_years == 2013) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2014) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20132014_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20132014 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20132014 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20132014_monthly = gerlache_cluster_20132014.resample('M').mean()
gerlache_cluster_20132014_monthly = gerlache_cluster_20132014_monthly.values
## 2014-2015
idx_year_chl_start = np.argwhere((time_date_years == 2014) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2015) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20142015_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20142015 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20142015 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20142015_monthly = gerlache_cluster_20142015.resample('M').mean()
gerlache_cluster_20142015_monthly = gerlache_cluster_20142015_monthly.values
## 2015-2016
idx_year_chl_start = np.argwhere((time_date_years == 2015) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2016) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20152016_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20152016 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20152016 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20152016_monthly = gerlache_cluster_20152016.resample('M').mean()
gerlache_cluster_20152016_monthly = gerlache_cluster_20152016_monthly.values
## 2016-2017
idx_year_chl_start = np.argwhere((time_date_years == 2016) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2017) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20162017_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20162017 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20162017 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20162017_monthly = gerlache_cluster_20162017.resample('M').mean()
gerlache_cluster_20162017_monthly = gerlache_cluster_20162017_monthly.values
## 2017-2018
idx_year_chl_start = np.argwhere((time_date_years == 2017) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2018) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20172018_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20172018 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20172018 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20172018_monthly = gerlache_cluster_20172018.resample('M').mean()
gerlache_cluster_20172018_monthly = gerlache_cluster_20172018_monthly.values
## 2018-2019
idx_year_chl_start = np.argwhere((time_date_years == 2018) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2019) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20182019_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20182019 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20182019 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20182019_monthly = gerlache_cluster_20182019.resample('M').mean()
gerlache_cluster_20182019_monthly = gerlache_cluster_20182019_monthly.values
## 2019-2020
idx_year_chl_start = np.argwhere((time_date_years == 2019) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2020) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20192020_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20192020 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20192020 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20192020_monthly = gerlache_cluster_20192020.resample('M').mean()
gerlache_cluster_20192020_monthly = gerlache_cluster_20192020_monthly.values
## 2020-2021
idx_year_chl_start = np.argwhere((time_date_years == 2020) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2021) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20202021_daily = gerlache_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20202021 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20202021 = pd.Series(gerlache_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
gerlache_cluster_20202021_monthly = gerlache_cluster_20202021.resample('M').mean()
gerlache_cluster_20202021_monthly = gerlache_cluster_20202021_monthly.values
## 1997-2021 Mean cycle
gerlache_cluster_19972021_july =  np.nanmean(gerlache_cluster[time_date_months == 7])
gerlache_cluster_19972021_august =  np.nanmean(gerlache_cluster[time_date_months == 8])
gerlache_cluster_19972021_september =  np.nanmean(gerlache_cluster[time_date_months == 9])
gerlache_cluster_19972021_october =  np.nanmean(gerlache_cluster[time_date_months == 10])
gerlache_cluster_19972021_november =  np.nanmean(gerlache_cluster[time_date_months == 11])
gerlache_cluster_19972021_december =  np.nanmean(gerlache_cluster[time_date_months == 12])
gerlache_cluster_19972021_january =  np.nanmean(gerlache_cluster[time_date_months == 1])
gerlache_cluster_19972021_february =  np.nanmean(gerlache_cluster[time_date_months == 2])
gerlache_cluster_19972021_march =  np.nanmean(gerlache_cluster[time_date_months == 3])
gerlache_cluster_19972021_april =  np.nanmean(gerlache_cluster[time_date_months == 4])
gerlache_cluster_19972021_may =  np.nanmean(gerlache_cluster[time_date_months == 5])
gerlache_cluster_19972021_june =  np.nanmean(gerlache_cluster[time_date_months == 6])
gerlache_cluster_19972021_monthly = np.hstack((gerlache_cluster_19972021_july,
                                              gerlache_cluster_19972021_august,
                                              gerlache_cluster_19972021_september,
                                              gerlache_cluster_19972021_october,
                                              gerlache_cluster_19972021_november,
                                              gerlache_cluster_19972021_december,
                                              gerlache_cluster_19972021_january,
                                              gerlache_cluster_19972021_february,
                                              gerlache_cluster_19972021_march,
                                              gerlache_cluster_19972021_april,
                                              gerlache_cluster_19972021_may,
                                              gerlache_cluster_19972021_june
                                              ))
# Join yearly cicles
gerlache_cluster_allcicles = np.vstack([gerlache_cluster_19971998_monthly,
                                      gerlache_cluster_19981999_monthly,
                                      gerlache_cluster_19992000_monthly,
                                      gerlache_cluster_20002001_monthly,
                                      gerlache_cluster_20012002_monthly,
                                      gerlache_cluster_20022003_monthly,
                                      gerlache_cluster_20032004_monthly,
                                      gerlache_cluster_20042005_monthly,
                                      gerlache_cluster_20052006_monthly,
                                      gerlache_cluster_20062007_monthly,
                                      gerlache_cluster_20072008_monthly,
                                      gerlache_cluster_20082009_monthly,
                                      gerlache_cluster_20092010_monthly,
                                      gerlache_cluster_20102011_monthly,
                                      gerlache_cluster_20112012_monthly,
                                      gerlache_cluster_20122013_monthly,
                                      gerlache_cluster_20132014_monthly,
                                      gerlache_cluster_20142015_monthly,
                                      gerlache_cluster_20152016_monthly,
                                      gerlache_cluster_20162017_monthly,
                                      gerlache_cluster_20172018_monthly,
                                      gerlache_cluster_20182019_monthly,
                                      gerlache_cluster_20192020_monthly,
                                      gerlache_cluster_20202021_monthly])

gerlache_cluster_19982005 = np.nanmean(gerlache_cluster_allcicles[:8,:], axis=0)
gerlache_cluster_20062014 = np.nanmean(gerlache_cluster_allcicles[8:16,:], axis=0)
gerlache_cluster_20152021 = np.nanmean(gerlache_cluster_allcicles[16:,:], axis=0)
gerlache_yearlycicles_p90 = np.nanpercentile(gerlache_cluster_allcicles, 90, axis=0)
gerlache_yearlycicles_p10 = np.nanpercentile(gerlache_cluster_allcicles, 10, axis=0)
gerlache_yearlycicles_std = np.nanstd(gerlache_cluster_allcicles, axis=0)
gerlache_cluster_mean19972021 = np.nanmean(gerlache_cluster_allcicles, axis=0)
#%% gerlache Cluster Figure 1
f_cubic = interp1d(np.arange(3,11),gerlache_cluster_mean19972021[2:10], kind='cubic')
xnew = np.linspace(3, 10, num=50, endpoint=True)
plt.figure()
plt.plot(xnew, f_cubic(xnew), color = [215/256, 25/256, 28/256, 1], linewidth = 5, label='1997-2021')
#plt.plot(np.arange(1,13),gerlache_cluster_mean19972021, color = [215/256, 25/256, 28/256, 1], linewidth = 4, label='1997-2021')
plt.plot(np.arange(1,13),gerlache_cluster_19982005, color = 'k', linewidth = 2, linestyle='--', label='1997-2005', alpha=0.6)
plt.plot(np.arange(1,13),gerlache_cluster_20062014, color = 'k', linewidth = 2, linestyle=':', label='2005-2014', alpha=0.6)
plt.plot(np.arange(1,13),gerlache_cluster_20152021, color = 'k', linewidth = 2, linestyle='-.', label='2015-2021', alpha=0.6)
plt.errorbar(np.arange(1,13),gerlache_cluster_mean19972021, gerlache_yearlycicles_std, linestyle='None', marker='None',
             color = [215/256, 25/256, 28/256, 1], alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
#plt.fill_between(np.arange(1,13), gerlache_cluster_mean19972021, gerlache_cluster_mean19972021+gerlache_yearlycicles_std, color =[215/256, 25/256, 28/256, 1], alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), gerlache_cluster_mean19972021, gerlache_cluster_mean19972021-gerlache_yearlycicles_std, color =[215/256, 25/256, 28/256, 1], alpha=.2, edgecolor = None)
plt.xticks([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
           ['Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr','May', 'Jun'], fontsize=12)
plt.xlim(3,10)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\gerlache_meancycle_errorbar.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 3 (Oceanic)
oceanic_cluster = chl[clusters == 3,:]
oceanic_cluster = np.nanmean(oceanic_cluster,0)

np.nanmedian(oceanic_cluster)
np.nanmax(oceanic_cluster)
np.nanmin(oceanic_cluster)
np.nanstd(oceanic_cluster)*3
oceanic_cluster = np.where(oceanic_cluster > np.nanmedian(oceanic_cluster)-np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
oceanic_cluster = np.where(oceanic_cluster < np.nanmedian(oceanic_cluster)+np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)

### Divide data per year (July to June)
## 1997-1998
#idx_year_chl_start = np.argwhere((time_date_years == 1997) & (time_date_months == 7)).ravel()
idx_year_chl_end = np.argwhere((time_date_years == 1998) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_19971998_daily = oceanic_cluster[:idx_year_chl_end]
time_date19971998 = time_date[:idx_year_chl_end]
oceanic_cluster_19971998 = pd.Series(oceanic_cluster[:idx_year_chl_end], index=time_date[:idx_year_chl_end])
oceanic_cluster_19971998_monthly = oceanic_cluster_19971998.resample('M').mean()
oceanic_cluster_19971998_monthly = np.hstack((np.nan, np.nan, oceanic_cluster_19971998_monthly.values))
## 1998-1999
idx_year_chl_start = np.argwhere((time_date_years == 1998) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 1999) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_19981999_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date19981999 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_19981999 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_19981999_monthly = oceanic_cluster_19981999.resample('M').mean()
oceanic_cluster_19981999_monthly = oceanic_cluster_19981999_monthly.values
## 1999-2000
idx_year_chl_start = np.argwhere((time_date_years == 1999) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2000) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_19992000_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date19992000 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_19992000 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_19992000_monthly = oceanic_cluster_19992000.resample('M').mean()
oceanic_cluster_19992000_monthly = oceanic_cluster_19992000_monthly.values
## 2000-2001
idx_year_chl_start = np.argwhere((time_date_years == 2000) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2001) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20002001_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20002001 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20002001 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20002001_monthly = oceanic_cluster_20002001.resample('M').mean()
oceanic_cluster_20002001_monthly = oceanic_cluster_20002001_monthly.values
## 2001-2002
idx_year_chl_start = np.argwhere((time_date_years == 2001) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2002) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20012002_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20012002 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20012002 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20012002_monthly = oceanic_cluster_20012002.resample('M').mean()
oceanic_cluster_20012002_monthly = oceanic_cluster_20012002_monthly.values
## 2002-2003
idx_year_chl_start = np.argwhere((time_date_years == 2002) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2003) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20022003_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20022003 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20022003 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20022003_monthly = oceanic_cluster_20022003.resample('M').mean()
oceanic_cluster_20022003_monthly = oceanic_cluster_20022003_monthly.values
## 2003-2004
idx_year_chl_start = np.argwhere((time_date_years == 2003) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2004) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20032004_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20032004 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20032004 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20032004_monthly = oceanic_cluster_20032004.resample('M').mean()
oceanic_cluster_20032004_monthly = oceanic_cluster_20032004_monthly.values
## 2004-2005
idx_year_chl_start = np.argwhere((time_date_years == 2004) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2005) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20042005_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20042005 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20042005 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20042005_monthly = oceanic_cluster_20042005.resample('M').mean()
oceanic_cluster_20042005_monthly = oceanic_cluster_20042005_monthly.values
## 2005-2006
idx_year_chl_start = np.argwhere((time_date_years == 2005) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2006) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20052006_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20052006 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20052006 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20052006_monthly = oceanic_cluster_20052006.resample('M').mean()
oceanic_cluster_20052006_monthly = oceanic_cluster_20052006_monthly.values
## 2006-2007
idx_year_chl_start = np.argwhere((time_date_years == 2006) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2007) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20062007_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20062007 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20062007 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20062007_monthly = oceanic_cluster_20062007.resample('M').mean()
oceanic_cluster_20062007_monthly = oceanic_cluster_20062007_monthly.values
## 2007-2008
idx_year_chl_start = np.argwhere((time_date_years == 2007) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2008) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20072008_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20072008 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20072008 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20072008_monthly = oceanic_cluster_20072008.resample('M').mean()
oceanic_cluster_20072008_monthly = oceanic_cluster_20072008_monthly.values
## 2008-2009
idx_year_chl_start = np.argwhere((time_date_years == 2008) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2009) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20082009_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20082009 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20082009 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20082009_monthly = oceanic_cluster_20082009.resample('M').mean()
oceanic_cluster_20082009_monthly = oceanic_cluster_20082009_monthly.values
## 2009-2010
idx_year_chl_start = np.argwhere((time_date_years == 2009) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2010) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20092010_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20092010 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20092010 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20092010_monthly = oceanic_cluster_20092010.resample('M').mean()
oceanic_cluster_20092010_monthly = oceanic_cluster_20092010_monthly.values
## 2010-2011
idx_year_chl_start = np.argwhere((time_date_years == 2010) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2011) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20102011_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20102011 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20102011 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20102011_monthly = oceanic_cluster_20102011.resample('M').mean()
oceanic_cluster_20102011_monthly = oceanic_cluster_20102011_monthly.values
## 2011-2012
idx_year_chl_start = np.argwhere((time_date_years == 2011) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2012) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20112012_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20112012 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20112012 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20112012_monthly = oceanic_cluster_20112012.resample('M').mean()
oceanic_cluster_20112012_monthly = oceanic_cluster_20112012_monthly.values
## 2012-2013
idx_year_chl_start = np.argwhere((time_date_years == 2012) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2013) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20122013_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20122013 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20122013 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20122013_monthly = oceanic_cluster_20122013.resample('M').mean()
oceanic_cluster_20122013_monthly = oceanic_cluster_20122013_monthly.values
## 2013-2014
idx_year_chl_start = np.argwhere((time_date_years == 2013) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2014) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20132014_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20132014 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20132014 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20132014_monthly = oceanic_cluster_20132014.resample('M').mean()
oceanic_cluster_20132014_monthly = oceanic_cluster_20132014_monthly.values
## 2014-2015
idx_year_chl_start = np.argwhere((time_date_years == 2014) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2015) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20142015_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20142015 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20142015 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20142015_monthly = oceanic_cluster_20142015.resample('M').mean()
oceanic_cluster_20142015_monthly = oceanic_cluster_20142015_monthly.values
## 2015-2016
idx_year_chl_start = np.argwhere((time_date_years == 2015) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2016) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20152016_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20152016 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20152016 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20152016_monthly = oceanic_cluster_20152016.resample('M').mean()
oceanic_cluster_20152016_monthly = oceanic_cluster_20152016_monthly.values
## 2016-2017
idx_year_chl_start = np.argwhere((time_date_years == 2016) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2017) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20162017_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20162017 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20162017 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20162017_monthly = oceanic_cluster_20162017.resample('M').mean()
oceanic_cluster_20162017_monthly = oceanic_cluster_20162017_monthly.values
## 2017-2018
idx_year_chl_start = np.argwhere((time_date_years == 2017) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2018) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20172018_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20172018 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20172018 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20172018_monthly = oceanic_cluster_20172018.resample('M').mean()
oceanic_cluster_20172018_monthly = oceanic_cluster_20172018_monthly.values
## 2018-2019
idx_year_chl_start = np.argwhere((time_date_years == 2018) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2019) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20182019_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20182019 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20182019 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20182019_monthly = oceanic_cluster_20182019.resample('M').mean()
oceanic_cluster_20182019_monthly = oceanic_cluster_20182019_monthly.values
## 2019-2020
idx_year_chl_start = np.argwhere((time_date_years == 2019) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2020) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20192020_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20192020 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20192020 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20192020_monthly = oceanic_cluster_20192020.resample('M').mean()
oceanic_cluster_20192020_monthly = oceanic_cluster_20192020_monthly.values
## 2020-2021
idx_year_chl_start = np.argwhere((time_date_years == 2020) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2021) & (time_date_months == 6)).ravel()[-1]+1
oceanic_cluster_20202021_daily = oceanic_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20202021 = time_date[idx_year_chl_start:idx_year_chl_end]
oceanic_cluster_20202021 = pd.Series(oceanic_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
oceanic_cluster_20202021_monthly = oceanic_cluster_20202021.resample('M').mean()
oceanic_cluster_20202021_monthly = oceanic_cluster_20202021_monthly.values
## 1997-2021 Mean cycle
oceanic_cluster_19972021_july =  np.nanmean(oceanic_cluster[time_date_months == 7])
oceanic_cluster_19972021_august =  np.nanmean(oceanic_cluster[time_date_months == 8])
oceanic_cluster_19972021_september =  np.nanmean(oceanic_cluster[time_date_months == 9])
oceanic_cluster_19972021_october =  np.nanmean(oceanic_cluster[time_date_months == 10])
oceanic_cluster_19972021_november =  np.nanmean(oceanic_cluster[time_date_months == 11])
oceanic_cluster_19972021_december =  np.nanmean(oceanic_cluster[time_date_months == 12])
oceanic_cluster_19972021_january =  np.nanmean(oceanic_cluster[time_date_months == 1])
oceanic_cluster_19972021_february =  np.nanmean(oceanic_cluster[time_date_months == 2])
oceanic_cluster_19972021_march =  np.nanmean(oceanic_cluster[time_date_months == 3])
oceanic_cluster_19972021_april =  np.nanmean(oceanic_cluster[time_date_months == 4])
oceanic_cluster_19972021_may =  np.nanmean(oceanic_cluster[time_date_months == 5])
oceanic_cluster_19972021_june =  np.nanmean(oceanic_cluster[time_date_months == 6])
oceanic_cluster_19972021_monthly = np.hstack((oceanic_cluster_19972021_july,
                                              oceanic_cluster_19972021_august,
                                              oceanic_cluster_19972021_september,
                                              oceanic_cluster_19972021_october,
                                              oceanic_cluster_19972021_november,
                                              oceanic_cluster_19972021_december,
                                              oceanic_cluster_19972021_january,
                                              oceanic_cluster_19972021_february,
                                              oceanic_cluster_19972021_march,
                                              oceanic_cluster_19972021_april,
                                              oceanic_cluster_19972021_may,
                                              oceanic_cluster_19972021_june
                                              ))
# Join yearly cicles
oceanic_cluster_allcicles = np.vstack([oceanic_cluster_19971998_monthly,
                                      oceanic_cluster_19981999_monthly,
                                      oceanic_cluster_19992000_monthly,
                                      oceanic_cluster_20002001_monthly,
                                      oceanic_cluster_20012002_monthly,
                                      oceanic_cluster_20022003_monthly,
                                      oceanic_cluster_20032004_monthly,
                                      oceanic_cluster_20042005_monthly,
                                      oceanic_cluster_20052006_monthly,
                                      oceanic_cluster_20062007_monthly,
                                      oceanic_cluster_20072008_monthly,
                                      oceanic_cluster_20082009_monthly,
                                      oceanic_cluster_20092010_monthly,
                                      oceanic_cluster_20102011_monthly,
                                      oceanic_cluster_20112012_monthly,
                                      oceanic_cluster_20122013_monthly,
                                      oceanic_cluster_20132014_monthly,
                                      oceanic_cluster_20142015_monthly,
                                      oceanic_cluster_20152016_monthly,
                                      oceanic_cluster_20162017_monthly,
                                      oceanic_cluster_20172018_monthly,
                                      oceanic_cluster_20182019_monthly,
                                      oceanic_cluster_20192020_monthly,
                                      oceanic_cluster_20202021_monthly])

oceanic_cluster_19982005 = np.nanmean(oceanic_cluster_allcicles[:8,:], axis=0)
oceanic_cluster_20062014 = np.nanmean(oceanic_cluster_allcicles[8:16,:], axis=0)
oceanic_cluster_20152021 = np.nanmean(oceanic_cluster_allcicles[16:,:], axis=0)
oceanic_yearlycicles_p90 = np.nanpercentile(oceanic_cluster_allcicles, 90, axis=0)
oceanic_yearlycicles_p10 = np.nanpercentile(oceanic_cluster_allcicles, 10, axis=0)
oceanic_yearlycicles_std = np.nanstd(oceanic_cluster_allcicles, axis=0)
oceanic_cluster_mean19972021 = np.nanmean(oceanic_cluster_allcicles, axis=0)
#%% oceanic Cluster Figure 1
f_cubic = interp1d(np.arange(3,11),oceanic_cluster_mean19972021[2:10], kind='cubic')
xnew = np.linspace(3, 10, num=50, endpoint=True)
plt.figure()
plt.plot(xnew, f_cubic(xnew), color = '#d09c26', linewidth = 5, label='1997-2021')

#plt.plot(np.arange(1,13),oceanic_cluster_mean19972021, color = [171/256, 221/256, 164/256, 1], linewidth = 4, label='1997-2021')
plt.plot(np.arange(1,13),oceanic_cluster_19982005, color = 'k', linewidth = 2, linestyle='--', label='1997-2005', alpha=0.6)
plt.plot(np.arange(1,13),oceanic_cluster_20062014, color = 'k', linewidth = 2, linestyle=':', label='2005-2014', alpha=0.6)
plt.plot(np.arange(1,13),oceanic_cluster_20152021, color = 'k', linewidth = 2, linestyle='-.', label='2015-2021', alpha=0.6)
#plt.errorbar(np.arange(1,13),oceanic_cluster_mean19972021, oceanic_yearlycicles_std, linestyle='None', marker='None',
#             color = '#d09c26', alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
plt.fill_between(np.arange(1,13), oceanic_cluster_mean19972021, oceanic_cluster_mean19972021+oceanic_yearlycicles_std, color ='#d09c26', alpha=.2, edgecolor = None)
plt.fill_between(np.arange(1,13), oceanic_cluster_mean19972021, oceanic_cluster_mean19972021-oceanic_yearlycicles_std, color ='#d09c26', alpha=.2, edgecolor = None)
plt.xticks([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
           ['Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr','May', 'Jun'], fontsize=12)
plt.xlim(3,10)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.legend(fontsize=12, loc=1)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\oceanic_meancycle.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Separar para o cluster 4 (Bransfield)
bransfield_cluster = chl[clusters == 4,:]
bransfield_cluster = np.nanmean(bransfield_cluster,0)

np.nanmedian(bransfield_cluster)
np.nanmax(bransfield_cluster)
np.nanmin(bransfield_cluster)
np.nanstd(bransfield_cluster)*3
bransfield_cluster = np.where(bransfield_cluster > np.nanmedian(bransfield_cluster)-np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
bransfield_cluster = np.where(bransfield_cluster < np.nanmedian(bransfield_cluster)+np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)

### Divide data per year (July to June)
## 1997-1998
#idx_year_chl_start = np.argwhere((time_date_years == 1997) & (time_date_months == 7)).ravel()
idx_year_chl_end = np.argwhere((time_date_years == 1998) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_19971998_daily = bransfield_cluster[:idx_year_chl_end]
time_date19971998 = time_date[:idx_year_chl_end]
bransfield_cluster_19971998 = pd.Series(bransfield_cluster[:idx_year_chl_end], index=time_date[:idx_year_chl_end])
bransfield_cluster_19971998_monthly = bransfield_cluster_19971998.resample('M').mean()
bransfield_cluster_19971998_monthly = np.hstack((np.nan, np.nan, bransfield_cluster_19971998_monthly.values))
## 1998-1999
idx_year_chl_start = np.argwhere((time_date_years == 1998) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 1999) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_19981999_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date19981999 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_19981999 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_19981999_monthly = bransfield_cluster_19981999.resample('M').mean()
bransfield_cluster_19981999_monthly = bransfield_cluster_19981999_monthly.values
## 1999-2000
idx_year_chl_start = np.argwhere((time_date_years == 1999) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2000) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_19992000_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date19992000 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_19992000 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_19992000_monthly = bransfield_cluster_19992000.resample('M').mean()
bransfield_cluster_19992000_monthly = bransfield_cluster_19992000_monthly.values
## 2000-2001
idx_year_chl_start = np.argwhere((time_date_years == 2000) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2001) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20002001_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20002001 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20002001 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20002001_monthly = bransfield_cluster_20002001.resample('M').mean()
bransfield_cluster_20002001_monthly = bransfield_cluster_20002001_monthly.values
## 2001-2002
idx_year_chl_start = np.argwhere((time_date_years == 2001) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2002) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20012002_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20012002 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20012002 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20012002_monthly = bransfield_cluster_20012002.resample('M').mean()
bransfield_cluster_20012002_monthly = bransfield_cluster_20012002_monthly.values
## 2002-2003
idx_year_chl_start = np.argwhere((time_date_years == 2002) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2003) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20022003_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20022003 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20022003 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20022003_monthly = bransfield_cluster_20022003.resample('M').mean()
bransfield_cluster_20022003_monthly = bransfield_cluster_20022003_monthly.values
## 2003-2004
idx_year_chl_start = np.argwhere((time_date_years == 2003) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2004) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20032004_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20032004 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20032004 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20032004_monthly = bransfield_cluster_20032004.resample('M').mean()
bransfield_cluster_20032004_monthly = bransfield_cluster_20032004_monthly.values
## 2004-2005
idx_year_chl_start = np.argwhere((time_date_years == 2004) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2005) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20042005_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20042005 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20042005 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20042005_monthly = bransfield_cluster_20042005.resample('M').mean()
bransfield_cluster_20042005_monthly = bransfield_cluster_20042005_monthly.values
## 2005-2006
idx_year_chl_start = np.argwhere((time_date_years == 2005) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2006) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20052006_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20052006 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20052006 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20052006_monthly = bransfield_cluster_20052006.resample('M').mean()
bransfield_cluster_20052006_monthly = bransfield_cluster_20052006_monthly.values
## 2006-2007
idx_year_chl_start = np.argwhere((time_date_years == 2006) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2007) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20062007_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20062007 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20062007 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20062007_monthly = bransfield_cluster_20062007.resample('M').mean()
bransfield_cluster_20062007_monthly = bransfield_cluster_20062007_monthly.values
## 2007-2008
idx_year_chl_start = np.argwhere((time_date_years == 2007) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2008) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20072008_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20072008 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20072008 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20072008_monthly = bransfield_cluster_20072008.resample('M').mean()
bransfield_cluster_20072008_monthly = bransfield_cluster_20072008_monthly.values
## 2008-2009
idx_year_chl_start = np.argwhere((time_date_years == 2008) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2009) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20082009_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20082009 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20082009 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20082009_monthly = bransfield_cluster_20082009.resample('M').mean()
bransfield_cluster_20082009_monthly = bransfield_cluster_20082009_monthly.values
## 2009-2010
idx_year_chl_start = np.argwhere((time_date_years == 2009) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2010) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20092010_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20092010 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20092010 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20092010_monthly = bransfield_cluster_20092010.resample('M').mean()
bransfield_cluster_20092010_monthly = bransfield_cluster_20092010_monthly.values
## 2010-2011
idx_year_chl_start = np.argwhere((time_date_years == 2010) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2011) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20102011_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20102011 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20102011 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20102011_monthly = bransfield_cluster_20102011.resample('M').mean()
bransfield_cluster_20102011_monthly = bransfield_cluster_20102011_monthly.values
## 2011-2012
idx_year_chl_start = np.argwhere((time_date_years == 2011) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2012) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20112012_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20112012 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20112012 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20112012_monthly = bransfield_cluster_20112012.resample('M').mean()
bransfield_cluster_20112012_monthly = bransfield_cluster_20112012_monthly.values
## 2012-2013
idx_year_chl_start = np.argwhere((time_date_years == 2012) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2013) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20122013_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20122013 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20122013 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20122013_monthly = bransfield_cluster_20122013.resample('M').mean()
bransfield_cluster_20122013_monthly = bransfield_cluster_20122013_monthly.values
## 2013-2014
idx_year_chl_start = np.argwhere((time_date_years == 2013) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2014) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20132014_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20132014 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20132014 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20132014_monthly = bransfield_cluster_20132014.resample('M').mean()
bransfield_cluster_20132014_monthly = bransfield_cluster_20132014_monthly.values
## 2014-2015
idx_year_chl_start = np.argwhere((time_date_years == 2014) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2015) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20142015_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20142015 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20142015 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20142015_monthly = bransfield_cluster_20142015.resample('M').mean()
bransfield_cluster_20142015_monthly = bransfield_cluster_20142015_monthly.values
## 2015-2016
idx_year_chl_start = np.argwhere((time_date_years == 2015) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2016) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20152016_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20152016 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20152016 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20152016_monthly = bransfield_cluster_20152016.resample('M').mean()
bransfield_cluster_20152016_monthly = bransfield_cluster_20152016_monthly.values
## 2016-2017
idx_year_chl_start = np.argwhere((time_date_years == 2016) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2017) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20162017_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20162017 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20162017 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20162017_monthly = bransfield_cluster_20162017.resample('M').mean()
bransfield_cluster_20162017_monthly = bransfield_cluster_20162017_monthly.values
## 2017-2018
idx_year_chl_start = np.argwhere((time_date_years == 2017) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2018) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20172018_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20172018 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20172018 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20172018_monthly = bransfield_cluster_20172018.resample('M').mean()
bransfield_cluster_20172018_monthly = bransfield_cluster_20172018_monthly.values
## 2018-2019
idx_year_chl_start = np.argwhere((time_date_years == 2018) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2019) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20182019_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20182019 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20182019 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20182019_monthly = bransfield_cluster_20182019.resample('M').mean()
bransfield_cluster_20182019_monthly = bransfield_cluster_20182019_monthly.values
## 2019-2020
idx_year_chl_start = np.argwhere((time_date_years == 2019) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2020) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20192020_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20192020 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20192020 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20192020_monthly = bransfield_cluster_20192020.resample('M').mean()
bransfield_cluster_20192020_monthly = bransfield_cluster_20192020_monthly.values
## 2020-2021
idx_year_chl_start = np.argwhere((time_date_years == 2020) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2021) & (time_date_months == 6)).ravel()[-1]+1
bransfield_cluster_20202021_daily = bransfield_cluster[idx_year_chl_start:idx_year_chl_end]
time_date20202021 = time_date[idx_year_chl_start:idx_year_chl_end]
bransfield_cluster_20202021 = pd.Series(bransfield_cluster[idx_year_chl_start:idx_year_chl_end], index=time_date[idx_year_chl_start:idx_year_chl_end])
bransfield_cluster_20202021_monthly = bransfield_cluster_20202021.resample('M').mean()
bransfield_cluster_20202021_monthly = bransfield_cluster_20202021_monthly.values
## 1997-2021 Mean cycle
bransfield_cluster_19972021_july =  np.nanmean(bransfield_cluster[time_date_months == 7])
bransfield_cluster_19972021_august =  np.nanmean(bransfield_cluster[time_date_months == 8])
bransfield_cluster_19972021_september =  np.nanmean(bransfield_cluster[time_date_months == 9])
bransfield_cluster_19972021_october =  np.nanmean(bransfield_cluster[time_date_months == 10])
bransfield_cluster_19972021_november =  np.nanmean(bransfield_cluster[time_date_months == 11])
bransfield_cluster_19972021_december =  np.nanmean(bransfield_cluster[time_date_months == 12])
bransfield_cluster_19972021_january =  np.nanmean(bransfield_cluster[time_date_months == 1])
bransfield_cluster_19972021_february =  np.nanmean(bransfield_cluster[time_date_months == 2])
bransfield_cluster_19972021_march =  np.nanmean(bransfield_cluster[time_date_months == 3])
bransfield_cluster_19972021_april =  np.nanmean(bransfield_cluster[time_date_months == 4])
bransfield_cluster_19972021_may =  np.nanmean(bransfield_cluster[time_date_months == 5])
bransfield_cluster_19972021_june =  np.nanmean(bransfield_cluster[time_date_months == 6])
bransfield_cluster_19972021_monthly = np.hstack((bransfield_cluster_19972021_july,
                                              bransfield_cluster_19972021_august,
                                              bransfield_cluster_19972021_september,
                                              bransfield_cluster_19972021_october,
                                              bransfield_cluster_19972021_november,
                                              bransfield_cluster_19972021_december,
                                              bransfield_cluster_19972021_january,
                                              bransfield_cluster_19972021_february,
                                              bransfield_cluster_19972021_march,
                                              bransfield_cluster_19972021_april,
                                              bransfield_cluster_19972021_may,
                                              bransfield_cluster_19972021_june
                                              ))
# Join yearly cicles
bransfield_cluster_allcicles = np.vstack([bransfield_cluster_19971998_monthly,
                                      bransfield_cluster_19981999_monthly,
                                      bransfield_cluster_19992000_monthly,
                                      bransfield_cluster_20002001_monthly,
                                      bransfield_cluster_20012002_monthly,
                                      bransfield_cluster_20022003_monthly,
                                      bransfield_cluster_20032004_monthly,
                                      bransfield_cluster_20042005_monthly,
                                      bransfield_cluster_20052006_monthly,
                                      bransfield_cluster_20062007_monthly,
                                      bransfield_cluster_20072008_monthly,
                                      bransfield_cluster_20082009_monthly,
                                      bransfield_cluster_20092010_monthly,
                                      bransfield_cluster_20102011_monthly,
                                      bransfield_cluster_20112012_monthly,
                                      bransfield_cluster_20122013_monthly,
                                      bransfield_cluster_20132014_monthly,
                                      bransfield_cluster_20142015_monthly,
                                      bransfield_cluster_20152016_monthly,
                                      bransfield_cluster_20162017_monthly,
                                      bransfield_cluster_20172018_monthly,
                                      bransfield_cluster_20182019_monthly,
                                      bransfield_cluster_20192020_monthly,
                                      bransfield_cluster_20202021_monthly])

bransfield_cluster_19982005 = np.nanmean(bransfield_cluster_allcicles[:8,:], axis=0)
bransfield_cluster_20062014 = np.nanmean(bransfield_cluster_allcicles[8:16,:], axis=0)
bransfield_cluster_20152021 = np.nanmean(bransfield_cluster_allcicles[16:,:], axis=0)
bransfield_yearlycicles_p90 = np.nanpercentile(bransfield_cluster_allcicles, 90, axis=0)
bransfield_yearlycicles_p10 = np.nanpercentile(bransfield_cluster_allcicles, 10, axis=0)
bransfield_yearlycicles_std = np.nanstd(bransfield_cluster_allcicles, axis=0)
bransfield_cluster_mean19972021 = np.nanmean(bransfield_cluster_allcicles, axis=0)
#%% bransfield Cluster Figure 1
f_cubic = interp1d(np.arange(3,11),bransfield_cluster_mean19972021[2:10], kind='cubic')
xnew = np.linspace(3, 10, num=50, endpoint=True)
plt.figure()
plt.plot(xnew, f_cubic(xnew), color = '#9800cb', linewidth = 5, label='1997-2021')

#plt.plot(np.arange(1,13),bransfield_cluster_mean19972021, color = [241/256, 180/256, 47/256, 1], linewidth = 4, label='1997-2021')
plt.plot(np.arange(1,13),bransfield_cluster_19982005, color = 'k', linewidth = 2, linestyle='--', label='1997-2005')
plt.plot(np.arange(1,13),bransfield_cluster_20062014, color = 'k', linewidth = 2, linestyle=':', label='2005-2014')
plt.plot(np.arange(1,13),bransfield_cluster_20152021, color = 'k', linewidth = 2, linestyle='-.', label='2015-2021')
plt.errorbar(np.arange(1,13),bransfield_cluster_mean19972021, bransfield_yearlycicles_std, linestyle='None', marker='None',
             color = '#9800cb', alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
#plt.fill_between(np.arange(1,13), bransfield_cluster_mean19972021, bransfield_cluster_mean19972021+bransfield_yearlycicles_std, color ='#9800cb', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), bransfield_cluster_mean19972021, bransfield_cluster_mean19972021-bransfield_yearlycicles_std, color ='#9800cb', alpha=.2, edgecolor = None)
plt.xticks([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
           ['Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr','May', 'Jun'], fontsize=12)
plt.xlim(3,10)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\bransfield_meancycle_errorbar.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate trends for each month or summer (Nov-Mar)
## Weddell
# Series for each month
weddell_cluster_sep = weddel_cluster_allcicles[:,2]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(weddell_cluster_sep)], weddell_cluster_sep[~np.isnan(weddell_cluster_sep)])
weddell_cluster_oct = weddel_cluster_allcicles[:,3]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(weddell_cluster_oct)], weddell_cluster_oct[~np.isnan(weddell_cluster_oct)])
weddell_cluster_nov = weddel_cluster_allcicles[:,4]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(weddell_cluster_nov)], weddell_cluster_nov[~np.isnan(weddell_cluster_nov)])
weddell_cluster_dec = weddel_cluster_allcicles[:,5]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(weddell_cluster_dec)], weddell_cluster_dec[~np.isnan(weddell_cluster_dec)])
weddell_cluster_jan = weddel_cluster_allcicles[:,6]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(weddell_cluster_jan)], weddell_cluster_jan[~np.isnan(weddell_cluster_jan)])
weddell_cluster_feb = weddel_cluster_allcicles[:,7]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(weddell_cluster_feb)], weddell_cluster_feb[~np.isnan(weddell_cluster_feb)])
weddell_cluster_mar = weddel_cluster_allcicles[:,8]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(weddell_cluster_mar)], weddell_cluster_mar[~np.isnan(weddell_cluster_mar)])
weddell_cluster_apr = weddel_cluster_allcicles[:,9]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(weddell_cluster_apr)], weddell_cluster_apr[~np.isnan(weddell_cluster_apr)])
# Series for each Spring-Summer
weddell_cluster_mean_allyears=np.nanmean(weddel_cluster_allcicles,1)
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), weddell_cluster_mean_allyears)
plt.scatter(np.arange(1998, 2022), weddell_cluster_mean_allyears)
plt.plot(np.arange(1998, 2022), np.arange(1998, 2022) * slope + intercept)
plt.xlabel('Years')
plt.ylabel('Chl')
#plt.scatter(np.arangeweddell_cluster_mean_allyears)
# Series for just Nov-Mar
weddell_cluster_summermean_allyears=np.nanmean(weddel_cluster_allcicles[:,4:9],1)
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), weddell_cluster_summermean_allyears)
## Gerlache
# Series for each month
gerlache_cluster_sep = gerlache_cluster_allcicles[:,2] # SIGNIFICATIVO
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_sep)], gerlache_cluster_sep[~np.isnan(gerlache_cluster_sep)])
gerlache_cluster_oct = gerlache_cluster_allcicles[:,3]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_oct)], gerlache_cluster_oct[~np.isnan(gerlache_cluster_oct)])
gerlache_cluster_nov = gerlache_cluster_allcicles[:,4]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_nov)], gerlache_cluster_nov[~np.isnan(gerlache_cluster_nov)])
gerlache_cluster_dec = gerlache_cluster_allcicles[:,5] #LIMIAR (0.059)
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_dec)], gerlache_cluster_dec[~np.isnan(gerlache_cluster_dec)])
gerlache_cluster_jan = gerlache_cluster_allcicles[:,6]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_jan)], gerlache_cluster_jan[~np.isnan(gerlache_cluster_jan)])
gerlache_cluster_feb = gerlache_cluster_allcicles[:,7]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_feb)], gerlache_cluster_feb[~np.isnan(gerlache_cluster_feb)])
gerlache_cluster_mar = gerlache_cluster_allcicles[:,8] # sIGNIFICATIVO
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_mar)], gerlache_cluster_mar[~np.isnan(gerlache_cluster_mar)])
gerlache_cluster_apr = gerlache_cluster_allcicles[:,9]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_apr)], gerlache_cluster_apr[~np.isnan(gerlache_cluster_apr)])
plt.scatter(np.arange(1998, 2022), gerlache_cluster_sep)
# Series for each Spring-Summer # P-VAL < 0.1
gerlache_cluster_mean_allyears=np.nanmean(gerlache_cluster_allcicles,1)
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), gerlache_cluster_mean_allyears)
# Series for just Nov-Mar
gerlache_cluster_summermean_allyears=np.nanmean(gerlache_cluster_allcicles[:,4:9],1)
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), gerlache_cluster_summermean_allyears)
plt.scatter(np.arange(1998, 2022), gerlache_cluster_mar)
plt.plot(np.arange(1998, 2022), np.arange(1998, 2022) * slope + intercept)
plt.xlabel('Years')
plt.ylabel('Chl')
## Oceanic
# Series for each month
oceanic_cluster_sep = oceanic_cluster_allcicles[:,2] 
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(oceanic_cluster_sep)], oceanic_cluster_sep[~np.isnan(oceanic_cluster_sep)])
oceanic_cluster_oct = oceanic_cluster_allcicles[:,3]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(oceanic_cluster_oct)], oceanic_cluster_oct[~np.isnan(oceanic_cluster_oct)])
oceanic_cluster_nov = oceanic_cluster_allcicles[:,4]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(oceanic_cluster_nov)], oceanic_cluster_nov[~np.isnan(oceanic_cluster_nov)])
oceanic_cluster_dec = oceanic_cluster_allcicles[:,5] 
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(oceanic_cluster_dec)], oceanic_cluster_dec[~np.isnan(oceanic_cluster_dec)])
oceanic_cluster_jan = oceanic_cluster_allcicles[:,6]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(oceanic_cluster_jan)], oceanic_cluster_jan[~np.isnan(oceanic_cluster_jan)])
oceanic_cluster_feb = oceanic_cluster_allcicles[:,7]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(oceanic_cluster_feb)], oceanic_cluster_feb[~np.isnan(oceanic_cluster_feb)])
oceanic_cluster_mar = oceanic_cluster_allcicles[:,8] # sIGNIFICATIVO
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(oceanic_cluster_mar)], oceanic_cluster_mar[~np.isnan(oceanic_cluster_mar)])
oceanic_cluster_apr = oceanic_cluster_allcicles[:,9]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(oceanic_cluster_apr)], oceanic_cluster_apr[~np.isnan(oceanic_cluster_apr)])
plt.scatter(np.arange(1998, 2022), oceanic_cluster_mar)
# Series for each Spring-Summer # P-VAL < 0.1
oceanic_cluster_mean_allyears=np.nanmean(oceanic_cluster_allcicles,1)
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), oceanic_cluster_mean_allyears)
# Series for just Nov-Mar
oceanic_cluster_summermean_allyears=np.nanmean(oceanic_cluster_allcicles[:,4:9],1)
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), oceanic_cluster_summermean_allyears)
plt.scatter(np.arange(1998, 2022), oceanic_cluster_mar)
plt.plot(np.arange(1998, 2022), np.arange(1998, 2022) * slope + intercept)
plt.xlabel('Years')
plt.ylabel('Chl')
## Bransfield
# Series for each month
bransfield_cluster_sep = bransfield_cluster_allcicles[:,2] #P-VAL <0.1
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(bransfield_cluster_sep)], bransfield_cluster_sep[~np.isnan(bransfield_cluster_sep)])
bransfield_cluster_oct = bransfield_cluster_allcicles[:,3]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(bransfield_cluster_oct)], bransfield_cluster_oct[~np.isnan(bransfield_cluster_oct)])
bransfield_cluster_nov = bransfield_cluster_allcicles[:,4]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(bransfield_cluster_nov)], bransfield_cluster_nov[~np.isnan(bransfield_cluster_nov)])
bransfield_cluster_dec = bransfield_cluster_allcicles[:,5] #LIMIAR 0.57 
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(bransfield_cluster_dec)], bransfield_cluster_dec[~np.isnan(bransfield_cluster_dec)])
bransfield_cluster_jan = bransfield_cluster_allcicles[:,6]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(bransfield_cluster_jan)], bransfield_cluster_jan[~np.isnan(bransfield_cluster_jan)])
bransfield_cluster_feb = bransfield_cluster_allcicles[:,7] # P-VAL <0.1
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(bransfield_cluster_feb)], bransfield_cluster_feb[~np.isnan(bransfield_cluster_feb)])
bransfield_cluster_mar = bransfield_cluster_allcicles[:,8] # SIGNIFICATIVO
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(bransfield_cluster_mar)], bransfield_cluster_mar[~np.isnan(bransfield_cluster_mar)])
bransfield_cluster_apr = bransfield_cluster_allcicles[:,9] #SIGNIFICATIVO
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(bransfield_cluster_apr)], bransfield_cluster_apr[~np.isnan(bransfield_cluster_apr)])
#plt.scatter(np.arange(1998, 2022), bransfield_cluster_summermean_allyears)
# Series for each Spring-Summer # SIGNIFICATIVO
bransfield_cluster_mean_allyears=np.nanmean(bransfield_cluster_allcicles,1)
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), bransfield_cluster_mean_allyears)
# Series for just Nov-Mar # SIGNIFICATIVO
bransfield_cluster_summermean_allyears=np.nanmean(bransfield_cluster_allcicles[:,4:9],1)
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), bransfield_cluster_summermean_allyears)
plt.scatter(np.arange(1998, 2022), bransfield_cluster_summermean_allyears)
plt.plot(np.arange(1998, 2022), np.arange(1998, 2022) * slope + intercept)
plt.xlabel('Years')
plt.ylabel('Chl')
#%% Weddell Plot comparison between 1997-2005, 2006-2014, 2015-2021
# Sep
weddell_cluster_sep_19982005 = weddel_cluster_allcicles[:8,2]
weddell_cluster_sep_19982005 = weddell_cluster_sep_19982005[~np.isnan(weddell_cluster_sep_19982005)]
weddell_cluster_sep_20062014 = weddel_cluster_allcicles[8:16,2]
weddell_cluster_sep_20152021 = weddel_cluster_allcicles[16:,2]
weddell_cluster_sep_20152021 = weddell_cluster_sep_20152021[~np.isnan(weddell_cluster_sep_20152021)]
plt.boxplot([weddell_cluster_sep_19982005, weddell_cluster_sep_20062014,
             weddell_cluster_sep_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_sep_19982005, weddell_cluster_sep_20062014, weddell_cluster_sep_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_sep_19982005, weddell_cluster_sep_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_sep_19982005, weddell_cluster_sep_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_sep_20062014, weddell_cluster_sep_20152021, nan_policy='omit')
# Oct
weddell_cluster_oct_19982005 = weddel_cluster_allcicles[:8,3]
weddell_cluster_oct_19982005 = weddell_cluster_oct_19982005[~np.isnan(weddell_cluster_oct_19982005)]
weddell_cluster_oct_20062014 = weddel_cluster_allcicles[8:16,3]
weddell_cluster_oct_20152021 = weddel_cluster_allcicles[16:,3]
plt.boxplot([weddell_cluster_oct_19982005, weddell_cluster_oct_20062014,
             weddell_cluster_oct_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_oct_19982005, weddell_cluster_oct_20062014, weddell_cluster_oct_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_oct_19982005, weddell_cluster_oct_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_oct_19982005, weddell_cluster_oct_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_oct_20062014, weddell_cluster_oct_20152021, nan_policy='omit')
# Nov
weddell_cluster_nov_19982005 = weddel_cluster_allcicles[:8,4]
weddell_cluster_nov_20062014 = weddel_cluster_allcicles[8:16,4]
weddell_cluster_nov_20152021 = weddel_cluster_allcicles[16:,4]
plt.boxplot([weddell_cluster_nov_19982005, weddell_cluster_nov_20062014,
             weddell_cluster_nov_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_nov_19982005, weddell_cluster_nov_20062014, weddell_cluster_nov_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_nov_19982005, weddell_cluster_nov_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_nov_19982005, weddell_cluster_nov_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_nov_20062014, weddell_cluster_nov_20152021, nan_policy='omit')
# Dec
weddell_cluster_dec_19982005 = weddel_cluster_allcicles[:8,5]
weddell_cluster_dec_20062014 = weddel_cluster_allcicles[8:16,5]
weddell_cluster_dec_20152021 = weddel_cluster_allcicles[16:,5]
plt.boxplot([weddell_cluster_dec_19982005, weddell_cluster_dec_20062014,
             weddell_cluster_dec_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_dec_19982005, weddell_cluster_dec_20062014, weddell_cluster_dec_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_dec_19982005, weddell_cluster_dec_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_dec_19982005, weddell_cluster_dec_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_dec_20062014, weddell_cluster_dec_20152021, nan_policy='omit')
# Jan
weddell_cluster_jan_19982005 = weddel_cluster_allcicles[:8,6]
weddell_cluster_jan_20062014 = weddel_cluster_allcicles[8:16,6]
weddell_cluster_jan_20152021 = weddel_cluster_allcicles[16:,6]
plt.boxplot([weddell_cluster_jan_19982005, weddell_cluster_jan_20062014,
             weddell_cluster_jan_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_jan_19982005, weddell_cluster_jan_20062014, weddell_cluster_jan_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_jan_19982005, weddell_cluster_jan_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_jan_19982005, weddell_cluster_jan_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_jan_20062014, weddell_cluster_jan_20152021, nan_policy='omit')
# Feb
weddell_cluster_feb_19982005 = weddel_cluster_allcicles[:8,7]
weddell_cluster_feb_20062014 = weddel_cluster_allcicles[8:16,7]
weddell_cluster_feb_20062014 = weddell_cluster_feb_20062014[~np.isnan(weddell_cluster_feb_20062014)]
weddell_cluster_feb_20152021 = weddel_cluster_allcicles[16:,7]
plt.boxplot([weddell_cluster_feb_19982005, weddell_cluster_feb_20062014,
             weddell_cluster_feb_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_feb_19982005, weddell_cluster_feb_20062014, weddell_cluster_feb_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_feb_19982005, weddell_cluster_feb_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_feb_19982005, weddell_cluster_feb_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_feb_20062014, weddell_cluster_feb_20152021, nan_policy='omit')
# Mar
weddell_cluster_mar_19982005 = weddel_cluster_allcicles[:8,8]
weddell_cluster_mar_20062014 = weddel_cluster_allcicles[8:16,8]
weddell_cluster_mar_20062014 = weddell_cluster_mar_20062014[~np.isnan(weddell_cluster_mar_20062014)]
weddell_cluster_mar_20152021 = weddel_cluster_allcicles[16:,8]
plt.boxplot([weddell_cluster_mar_19982005, weddell_cluster_mar_20062014,
             weddell_cluster_mar_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_mar_19982005, weddell_cluster_mar_20062014, weddell_cluster_mar_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_mar_19982005, weddell_cluster_mar_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_mar_19982005, weddell_cluster_mar_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_mar_20062014, weddell_cluster_mar_20152021, nan_policy='omit')
# Apr
weddell_cluster_apr_19982005 = weddel_cluster_allcicles[:8,9]
weddell_cluster_apr_19982005 = weddell_cluster_apr_19982005[~np.isnan(weddell_cluster_apr_19982005)]
weddell_cluster_apr_20062014 = weddel_cluster_allcicles[8:16,9]
weddell_cluster_apr_20062014 = weddell_cluster_apr_20062014[~np.isnan(weddell_cluster_apr_20062014)]
#weddell_cluster_apr_20062014 = weddell_cluster_apr_20062014[~np.isnan(weddell_cluster_apr_20062014)]
weddell_cluster_apr_20152021 = weddel_cluster_allcicles[16:,9]
weddell_cluster_apr_20152021 = weddell_cluster_apr_20152021[~np.isnan(weddell_cluster_apr_20152021)]
plt.boxplot([weddell_cluster_apr_19982005, weddell_cluster_apr_20062014,
             weddell_cluster_apr_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_apr_19982005, weddell_cluster_apr_20062014, weddell_cluster_apr_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_apr_19982005, weddell_cluster_apr_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_apr_19982005, weddell_cluster_apr_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_apr_20062014, weddell_cluster_apr_20152021, nan_policy='omit')
# Spring Summer
weddell_cluster_springsummer_19982005 = np.nanmean(weddel_cluster_allcicles[:8,2:10],1)
weddell_cluster_springsummer_20062014 = np.nanmean(weddel_cluster_allcicles[8:16,2:10],1)
#weddell_cluster_apr_20062014 = weddell_cluster_apr_20062014[~np.isnan(weddell_cluster_apr_20062014)]
weddell_cluster_springsummer_20152021 = np.nanmean(weddel_cluster_allcicles[16:,2:10],1)
plt.boxplot([weddell_cluster_springsummer_19982005, weddell_cluster_springsummer_20062014,
             weddell_cluster_springsummer_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_springsummer_19982005, weddell_cluster_springsummer_20062014, weddell_cluster_springsummer_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_springsummer_19982005, weddell_cluster_springsummer_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_springsummer_19982005, weddell_cluster_springsummer_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_springsummer_20062014, weddell_cluster_springsummer_20152021, nan_policy='omit')
# Nov-Mar
weddell_cluster_novmar_19982005 = np.nanmean(weddel_cluster_allcicles[:8,4:9],1)
weddell_cluster_novmar_20062014 = np.nanmean(weddel_cluster_allcicles[8:16,4:9],1)
#weddell_cluster_apr_20062014 = weddell_cluster_apr_20062014[~np.isnan(weddell_cluster_apr_20062014)]
weddell_cluster_novmar_20152021 = np.nanmean(weddel_cluster_allcicles[16:,4:9],1)
plt.boxplot([weddell_cluster_novmar_19982005, weddell_cluster_novmar_20062014,
             weddell_cluster_novmar_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(weddell_cluster_novmar_19982005, weddell_cluster_novmar_20062014, weddell_cluster_novmar_20152021,
              nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_novmar_19982005, weddell_cluster_novmar_20062014, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_novmar_19982005, weddell_cluster_novmar_20152021, nan_policy='omit')
stats.mannwhitneyu(weddell_cluster_novmar_20062014, weddell_cluster_novmar_20152021, nan_policy='omit')

#%% Bransfield Plot comparison between 1997-2005, 2006-2014, 2015-2021
# Sep
bransfield_cluster_sep_19982005 = bransfield_cluster_allcicles[:8,2]
bransfield_cluster_sep_20062014 = bransfield_cluster_allcicles[8:16,2]
bransfield_cluster_sep_20152021 = bransfield_cluster_allcicles[16:,2]
plt.boxplot([bransfield_cluster_sep_19982005, bransfield_cluster_sep_20062014,
             bransfield_cluster_sep_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_sep_19982005, bransfield_cluster_sep_20062014, bransfield_cluster_sep_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_sep_19982005, bransfield_cluster_sep_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_sep_19982005, bransfield_cluster_sep_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_sep_20062014, bransfield_cluster_sep_20152021, nan_policy='omit')
# Oct
bransfield_cluster_oct_19982005 = bransfield_cluster_allcicles[:8,3]
bransfield_cluster_oct_20062014 = bransfield_cluster_allcicles[8:16,3]
bransfield_cluster_oct_20152021 = bransfield_cluster_allcicles[16:,3]
plt.boxplot([bransfield_cluster_oct_19982005, bransfield_cluster_oct_20062014,
             bransfield_cluster_oct_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_oct_19982005, bransfield_cluster_oct_20062014, bransfield_cluster_oct_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_oct_19982005, bransfield_cluster_oct_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_oct_19982005, bransfield_cluster_oct_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_oct_20062014, bransfield_cluster_oct_20152021, nan_policy='omit')
# Nov
bransfield_cluster_nov_19982005 = bransfield_cluster_allcicles[:8,4]
bransfield_cluster_nov_20062014 = bransfield_cluster_allcicles[8:16,4]
bransfield_cluster_nov_20152021 = bransfield_cluster_allcicles[16:,4]
plt.boxplot([bransfield_cluster_nov_19982005, bransfield_cluster_nov_20062014,
             bransfield_cluster_nov_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_nov_19982005, bransfield_cluster_nov_20062014, bransfield_cluster_nov_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_nov_19982005, bransfield_cluster_nov_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_nov_19982005, bransfield_cluster_nov_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_nov_20062014, bransfield_cluster_nov_20152021, nan_policy='omit')
# Dec
bransfield_cluster_dec_19982005 = bransfield_cluster_allcicles[:8,5]
bransfield_cluster_dec_20062014 = bransfield_cluster_allcicles[8:16,5]
bransfield_cluster_dec_20152021 = bransfield_cluster_allcicles[16:,5]
plt.boxplot([bransfield_cluster_dec_19982005, bransfield_cluster_dec_20062014,
             bransfield_cluster_dec_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_dec_19982005, bransfield_cluster_dec_20062014, bransfield_cluster_dec_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_dec_19982005, bransfield_cluster_dec_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_dec_19982005, bransfield_cluster_dec_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_dec_20062014, bransfield_cluster_dec_20152021, nan_policy='omit')
# Jan
bransfield_cluster_jan_19982005 = bransfield_cluster_allcicles[:8,6]
bransfield_cluster_jan_20062014 = bransfield_cluster_allcicles[8:16,6]
bransfield_cluster_jan_20152021 = bransfield_cluster_allcicles[16:,6]
plt.boxplot([bransfield_cluster_jan_19982005, bransfield_cluster_jan_20062014,
             bransfield_cluster_jan_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_jan_19982005, bransfield_cluster_jan_20062014, bransfield_cluster_jan_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_jan_19982005, bransfield_cluster_jan_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_jan_19982005, bransfield_cluster_jan_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_jan_20062014, bransfield_cluster_jan_20152021, nan_policy='omit')
# Feb
bransfield_cluster_feb_19982005 = bransfield_cluster_allcicles[:8,7]
bransfield_cluster_feb_20062014 = bransfield_cluster_allcicles[8:16,7]
bransfield_cluster_feb_20062014 = bransfield_cluster_feb_20062014[~np.isnan(bransfield_cluster_feb_20062014)]
bransfield_cluster_feb_20152021 = bransfield_cluster_allcicles[16:,7]
plt.boxplot([bransfield_cluster_feb_19982005, bransfield_cluster_feb_20062014,
             bransfield_cluster_feb_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_feb_19982005, bransfield_cluster_feb_20062014, bransfield_cluster_feb_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_feb_19982005, bransfield_cluster_feb_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_feb_19982005, bransfield_cluster_feb_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_feb_20062014, bransfield_cluster_feb_20152021, nan_policy='omit')
# Mar
bransfield_cluster_mar_19982005 = bransfield_cluster_allcicles[:8,8]
bransfield_cluster_mar_20062014 = bransfield_cluster_allcicles[8:16,8]
bransfield_cluster_mar_20062014 = bransfield_cluster_mar_20062014[~np.isnan(bransfield_cluster_mar_20062014)]
bransfield_cluster_mar_20152021 = bransfield_cluster_allcicles[16:,8]
plt.boxplot([bransfield_cluster_mar_19982005, bransfield_cluster_mar_20062014,
             bransfield_cluster_mar_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_mar_19982005, bransfield_cluster_mar_20062014, bransfield_cluster_mar_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_mar_19982005, bransfield_cluster_mar_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_mar_19982005, bransfield_cluster_mar_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_mar_20062014, bransfield_cluster_mar_20152021, nan_policy='omit')
# Apr
bransfield_cluster_apr_19982005 = bransfield_cluster_allcicles[:8,9]
bransfield_cluster_apr_20062014 = bransfield_cluster_allcicles[8:16,9]
#bransfield_cluster_apr_20062014 = bransfield_cluster_apr_20062014[~np.isnan(bransfield_cluster_apr_20062014)]
bransfield_cluster_apr_20152021 = bransfield_cluster_allcicles[16:,9]
plt.boxplot([bransfield_cluster_apr_19982005, bransfield_cluster_apr_20062014,
             bransfield_cluster_apr_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_apr_19982005, bransfield_cluster_apr_20062014, bransfield_cluster_apr_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_apr_19982005, bransfield_cluster_apr_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_apr_19982005, bransfield_cluster_apr_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_apr_20062014, bransfield_cluster_apr_20152021, nan_policy='omit')
# Spring Summer
bransfield_cluster_springsummer_19982005 = np.nanmean(bransfield_cluster_allcicles[:8,2:10],1)
bransfield_cluster_springsummer_20062014 = np.nanmean(bransfield_cluster_allcicles[8:16,2:10],1)
#bransfield_cluster_apr_20062014 = bransfield_cluster_apr_20062014[~np.isnan(bransfield_cluster_apr_20062014)]
bransfield_cluster_springsummer_20152021 = np.nanmean(bransfield_cluster_allcicles[16:,2:10],1)
plt.boxplot([bransfield_cluster_springsummer_19982005, bransfield_cluster_springsummer_20062014,
             bransfield_cluster_springsummer_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_springsummer_19982005, bransfield_cluster_springsummer_20062014, bransfield_cluster_springsummer_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_springsummer_19982005, bransfield_cluster_springsummer_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_springsummer_19982005, bransfield_cluster_springsummer_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_springsummer_20062014, bransfield_cluster_springsummer_20152021, nan_policy='omit')
# Nov-Mar
bransfield_cluster_novmar_19982005 = np.nanmean(bransfield_cluster_allcicles[:8,4:9],1)
bransfield_cluster_novmar_20062014 = np.nanmean(bransfield_cluster_allcicles[8:16,4:9],1)
#bransfield_cluster_apr_20062014 = bransfield_cluster_apr_20062014[~np.isnan(bransfield_cluster_apr_20062014)]
bransfield_cluster_novmar_20152021 = np.nanmean(bransfield_cluster_allcicles[16:,4:9],1)
plt.boxplot([bransfield_cluster_novmar_19982005, bransfield_cluster_novmar_20062014,
             bransfield_cluster_novmar_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(bransfield_cluster_novmar_19982005, bransfield_cluster_novmar_20062014, bransfield_cluster_novmar_20152021,
              nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_novmar_19982005, bransfield_cluster_novmar_20062014, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_novmar_19982005, bransfield_cluster_novmar_20152021, nan_policy='omit')
stats.mannwhitneyu(bransfield_cluster_novmar_20062014, bransfield_cluster_novmar_20152021, nan_policy='omit')
#%% Oceanic Plot comparison between 1997-2005, 2006-2014, 2015-2021
# Sep
oceanic_cluster_sep_19982005 = oceanic_cluster_allcicles[:8,2]
oceanic_cluster_sep_20062014 = oceanic_cluster_allcicles[8:16,2]
oceanic_cluster_sep_20152021 = oceanic_cluster_allcicles[16:,2]
plt.boxplot([oceanic_cluster_sep_19982005, oceanic_cluster_sep_20062014,
             oceanic_cluster_sep_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_sep_19982005, oceanic_cluster_sep_20062014, oceanic_cluster_sep_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_sep_19982005, oceanic_cluster_sep_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_sep_19982005, oceanic_cluster_sep_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_sep_20062014, oceanic_cluster_sep_20152021, nan_policy='omit')
# Oct
oceanic_cluster_oct_19982005 = oceanic_cluster_allcicles[:8,3]
oceanic_cluster_oct_20062014 = oceanic_cluster_allcicles[8:16,3]
oceanic_cluster_oct_20152021 = oceanic_cluster_allcicles[16:,3]
plt.boxplot([oceanic_cluster_oct_19982005, oceanic_cluster_oct_20062014,
             oceanic_cluster_oct_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_oct_19982005, oceanic_cluster_oct_20062014, oceanic_cluster_oct_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_oct_19982005, oceanic_cluster_oct_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_oct_19982005, oceanic_cluster_oct_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_oct_20062014, oceanic_cluster_oct_20152021, nan_policy='omit')
# Nov
oceanic_cluster_nov_19982005 = oceanic_cluster_allcicles[:8,4]
oceanic_cluster_nov_20062014 = oceanic_cluster_allcicles[8:16,4]
oceanic_cluster_nov_20152021 = oceanic_cluster_allcicles[16:,4]
plt.boxplot([oceanic_cluster_nov_19982005, oceanic_cluster_nov_20062014,
             oceanic_cluster_nov_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_nov_19982005, oceanic_cluster_nov_20062014, oceanic_cluster_nov_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_nov_19982005, oceanic_cluster_nov_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_nov_19982005, oceanic_cluster_nov_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_nov_20062014, oceanic_cluster_nov_20152021, nan_policy='omit')
# Dec
oceanic_cluster_dec_19982005 = oceanic_cluster_allcicles[:8,5]
oceanic_cluster_dec_20062014 = oceanic_cluster_allcicles[8:16,5]
oceanic_cluster_dec_20152021 = oceanic_cluster_allcicles[16:,5]
plt.boxplot([oceanic_cluster_dec_19982005, oceanic_cluster_dec_20062014,
             oceanic_cluster_dec_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_dec_19982005, oceanic_cluster_dec_20062014, oceanic_cluster_dec_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_dec_19982005, oceanic_cluster_dec_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_dec_19982005, oceanic_cluster_dec_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_dec_20062014, oceanic_cluster_dec_20152021, nan_policy='omit')
# Jan
oceanic_cluster_jan_19982005 = oceanic_cluster_allcicles[:8,6]
oceanic_cluster_jan_20062014 = oceanic_cluster_allcicles[8:16,6]
oceanic_cluster_jan_20152021 = oceanic_cluster_allcicles[16:,6]
plt.boxplot([oceanic_cluster_jan_19982005, oceanic_cluster_jan_20062014,
             oceanic_cluster_jan_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_jan_19982005, oceanic_cluster_jan_20062014, oceanic_cluster_jan_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_jan_19982005, oceanic_cluster_jan_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_jan_19982005, oceanic_cluster_jan_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_jan_20062014, oceanic_cluster_jan_20152021, nan_policy='omit')
# Feb
oceanic_cluster_feb_19982005 = oceanic_cluster_allcicles[:8,7]
oceanic_cluster_feb_20062014 = oceanic_cluster_allcicles[8:16,7]
oceanic_cluster_feb_20062014 = oceanic_cluster_feb_20062014[~np.isnan(oceanic_cluster_feb_20062014)]
oceanic_cluster_feb_20152021 = oceanic_cluster_allcicles[16:,7]
plt.boxplot([oceanic_cluster_feb_19982005, oceanic_cluster_feb_20062014,
             oceanic_cluster_feb_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_feb_19982005, oceanic_cluster_feb_20062014, oceanic_cluster_feb_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_feb_19982005, oceanic_cluster_feb_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_feb_19982005, oceanic_cluster_feb_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_feb_20062014, oceanic_cluster_feb_20152021, nan_policy='omit')
# Mar
oceanic_cluster_mar_19982005 = oceanic_cluster_allcicles[:8,8]
oceanic_cluster_mar_20062014 = oceanic_cluster_allcicles[8:16,8]
oceanic_cluster_mar_20062014 = oceanic_cluster_mar_20062014[~np.isnan(oceanic_cluster_mar_20062014)]
oceanic_cluster_mar_20152021 = oceanic_cluster_allcicles[16:,8]
plt.boxplot([oceanic_cluster_mar_19982005, oceanic_cluster_mar_20062014,
             oceanic_cluster_mar_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_mar_19982005, oceanic_cluster_mar_20062014, oceanic_cluster_mar_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_mar_19982005, oceanic_cluster_mar_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_mar_19982005, oceanic_cluster_mar_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_mar_20062014, oceanic_cluster_mar_20152021, nan_policy='omit')
# Apr
oceanic_cluster_apr_19982005 = oceanic_cluster_allcicles[:8,9]
oceanic_cluster_apr_20062014 = oceanic_cluster_allcicles[8:16,9]
#oceanic_cluster_apr_20062014 = oceanic_cluster_apr_20062014[~np.isnan(oceanic_cluster_apr_20062014)]
oceanic_cluster_apr_20152021 = oceanic_cluster_allcicles[16:,9]
plt.boxplot([oceanic_cluster_apr_19982005, oceanic_cluster_apr_20062014,
             oceanic_cluster_apr_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_apr_19982005, oceanic_cluster_apr_20062014, oceanic_cluster_apr_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_apr_19982005, oceanic_cluster_apr_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_apr_19982005, oceanic_cluster_apr_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_apr_20062014, oceanic_cluster_apr_20152021, nan_policy='omit')
# Spring Summer
oceanic_cluster_springsummer_19982005 = np.nanmean(oceanic_cluster_allcicles[:8,2:10],1)
oceanic_cluster_springsummer_20062014 = np.nanmean(oceanic_cluster_allcicles[8:16,2:10],1)
#oceanic_cluster_apr_20062014 = oceanic_cluster_apr_20062014[~np.isnan(oceanic_cluster_apr_20062014)]
oceanic_cluster_springsummer_20152021 = np.nanmean(oceanic_cluster_allcicles[16:,2:10],1)
plt.boxplot([oceanic_cluster_springsummer_19982005, oceanic_cluster_springsummer_20062014,
             oceanic_cluster_springsummer_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_springsummer_19982005, oceanic_cluster_springsummer_20062014, oceanic_cluster_springsummer_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_springsummer_19982005, oceanic_cluster_springsummer_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_springsummer_19982005, oceanic_cluster_springsummer_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_springsummer_20062014, oceanic_cluster_springsummer_20152021, nan_policy='omit')
# Nov-Mar
oceanic_cluster_novmar_19982005 = np.nanmean(oceanic_cluster_allcicles[:8,4:9],1)
oceanic_cluster_novmar_20062014 = np.nanmean(oceanic_cluster_allcicles[8:16,4:9],1)
#oceanic_cluster_apr_20062014 = oceanic_cluster_apr_20062014[~np.isnan(oceanic_cluster_apr_20062014)]
oceanic_cluster_novmar_20152021 = np.nanmean(oceanic_cluster_allcicles[16:,4:9],1)
plt.boxplot([oceanic_cluster_novmar_19982005, oceanic_cluster_novmar_20062014,
             oceanic_cluster_novmar_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(oceanic_cluster_novmar_19982005, oceanic_cluster_novmar_20062014, oceanic_cluster_novmar_20152021,
              nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_novmar_19982005, oceanic_cluster_novmar_20062014, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_novmar_19982005, oceanic_cluster_novmar_20152021, nan_policy='omit')
stats.mannwhitneyu(oceanic_cluster_novmar_20062014, oceanic_cluster_novmar_20152021, nan_policy='omit')
#%% Weddell Cycle comparison between 1997-2005, 2006-2014, 2015-2021
# Spring Summer
weddell_cluster_springsummer_19982005 = np.nanmean(weddel_cluster_allcicles[:8,2:10],1)
weddell_cluster_springsummer_20062014 = np.nanmean(weddel_cluster_allcicles[8:16,2:10],1)
#weddell_cluster_apr_20062014 = weddell_cluster_apr_20062014[~np.isnan(weddell_cluster_apr_20062014)]
weddell_cluster_springsummer_20152021 = np.nanmean(weddel_cluster_allcicles[16:,2:10],1)
plt.plot(np.arange(1,9), weddell_cluster_springsummer_19982005, label='1998-2005')
plt.plot(np.arange(1,9), weddell_cluster_springsummer_20062014, label='2006-2014')
plt.plot(np.arange(1,9), weddell_cluster_springsummer_20152021, label='2015-2021')
plt.xticks(ticks=np.arange(1,9), labels=['S', 'O', 'N', 'D', 'J', 'F', 'M', 'A'])
plt.legend(loc=0)
#%% Bransfield Cycle comparison between 1997-2005, 2006-2014, 2015-2021
# Spring Summer
bransfield_cluster_springsummer_19982005 = np.nanmean(bransfield_cluster_allcicles[:8,2:10],1)
bransfield_cluster_springsummer_20062014 = np.nanmean(bransfield_cluster_allcicles[8:16,2:10],1)
#bransfield_cluster_apr_20062014 = bransfield_cluster_apr_20062014[~np.isnan(bransfield_cluster_apr_20062014)]
bransfield_cluster_springsummer_20152021 = np.nanmean(bransfield_cluster_allcicles[16:,2:10],1)
plt.plot(np.arange(1,9), bransfield_cluster_springsummer_19982005, label='1998-2005')
plt.plot(np.arange(1,9), bransfield_cluster_springsummer_20062014, label='2006-2014')
plt.plot(np.arange(1,9), bransfield_cluster_springsummer_20152021, label='2015-2021')
plt.xticks(ticks=np.arange(1,9), labels=['S', 'O', 'N', 'D', 'J', 'F', 'M', 'A'])
plt.legend(loc=0)
#%% Oceanic Cycle comparison between 1997-2005, 2006-2014, 2015-2021
# Spring Summer
oceanic_cluster_springsummer_19982005 = np.nanmean(oceanic_cluster_allcicles[:8,2:10],1)
oceanic_cluster_springsummer_20062014 = np.nanmean(oceanic_cluster_allcicles[8:16,2:10],1)
#oceanic_cluster_apr_20062014 = oceanic_cluster_apr_20062014[~np.isnan(oceanic_cluster_apr_20062014)]
oceanic_cluster_springsummer_20152021 = np.nanmean(oceanic_cluster_allcicles[16:,2:10],1)
plt.plot(np.arange(1,9), oceanic_cluster_springsummer_19982005, label='1998-2005')
plt.plot(np.arange(1,9), oceanic_cluster_springsummer_20062014, label='2006-2014')
plt.plot(np.arange(1,9), oceanic_cluster_springsummer_20152021, label='2015-2021')
plt.xticks(ticks=np.arange(1,9), labels=['S', 'O', 'N', 'D', 'J', 'F', 'M', 'A'])
plt.legend(loc=0)
#%% Gerlache Cycle comparison between 1997-2005, 2006-2014, 2015-2021
# Spring Summer
gerlache_cluster_springsummer_19982005 = np.nanmean(gerlache_cluster_allcicles[:8,2:10],1)
gerlache_cluster_springsummer_20062014 = np.nanmean(gerlache_cluster_allcicles[8:16,2:10],1)
#gerlache_cluster_apr_20062014 = gerlache_cluster_apr_20062014[~np.isnan(gerlache_cluster_apr_20062014)]
gerlache_cluster_springsummer_20152021 = np.nanmean(gerlache_cluster_allcicles[16:,2:10],1)
plt.plot(np.arange(1,9), gerlache_cluster_springsummer_19982005, label='1998-2005')
plt.plot(np.arange(1,9), gerlache_cluster_springsummer_20062014, label='2006-2014')
plt.plot(np.arange(1,9), gerlache_cluster_springsummer_20152021, label='2015-2021')
plt.xticks(ticks=np.arange(1,9), labels=['S', 'O', 'N', 'D', 'J', 'F', 'M', 'A'])
plt.legend(loc=0)