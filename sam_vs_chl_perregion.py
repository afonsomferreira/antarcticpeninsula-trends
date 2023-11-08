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
#%%
# DRA DJF
dra_cluster = chl[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
for i in np.arange(1998, 2023):
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        dra_chl_DJF = np.nanmean(yeartemp_DJF)
    else:
        dra_chl_DJF = np.hstack((dra_chl_DJF, np.nanmean(yeartemp_DJF)))
# BRS DJF
brs_cluster = chl[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
for i in np.arange(1998, 2023):
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        brs_chl_DJF = np.nanmean(yeartemp_DJF)
    else:
        brs_chl_DJF = np.hstack((brs_chl_DJF, np.nanmean(yeartemp_DJF)))
# WEDN DJF
wedn_cluster = chl[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
for i in np.arange(1998, 2023):
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        wedn_chl_DJF = np.nanmean(yeartemp_DJF)
    else:
        wedn_chl_DJF = np.hstack((wedn_chl_DJF, np.nanmean(yeartemp_DJF)))
# GES DJF
ges_cluster = chl[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
for i in np.arange(1998, 2023):
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        ges_chl_DJF = np.nanmean(yeartemp_DJF)
    else:
        ges_chl_DJF = np.hstack((ges_chl_DJF, np.nanmean(yeartemp_DJF)))
# WEDS DJF
weds_cluster = chl[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
for i in np.arange(1998, 2023):
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        weds_chl_DJF = np.nanmean(yeartemp_DJF)
    else:
        weds_chl_DJF = np.hstack((weds_chl_DJF, np.nanmean(yeartemp_DJF)))
# SAM
for i in np.arange(1998, 2023):
    yeartemp_sepdec = sam_pd_monthly.values[(sam_pd_monthly.index.year == i-1) & (sam_pd_monthly.index.month == 12)]
    yeartemp_janapr = sam_pd_monthly.values[(sam_pd_monthly.index.year == i) & ((sam_pd_monthly.index.month == 1) | (sam_pd_monthly.index.month == 2))]
    yeartemp_DJF = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        sam_DJF = np.nanmean(yeartemp_DJF)
    else:
        sam_DJF = np.hstack((sam_DJF, np.nanmean(yeartemp_DJF)))
#%% Correlations DJF
stats.spearmanr(dra_chl_DJF, sam_DJF, nan_policy='omit')
stats.spearmanr(brs_chl_DJF, sam_DJF, nan_policy='omit')
stats.spearmanr(wedn_chl_DJF, sam_DJF, nan_policy='omit')
stats.spearmanr(ges_chl_DJF, sam_DJF, nan_policy='omit') # *
stats.spearmanr(weds_chl_DJF, sam_DJF, nan_policy='omit')
#%% SEP-APR
# DRA SEP-APR
dra_cluster = chl[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_SEPAPR = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov,
                                 yeartemp_dec, yeartemp_jan, yeartemp_feb,
                                 yeartemp_mar, yeartemp_apr))
    if i == 1998:
        dra_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        dra_chl_SEPAPR = np.hstack((dra_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# BRS SEP-APR
brs_cluster = chl[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_SEPAPR = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov,
                                 yeartemp_dec, yeartemp_jan, yeartemp_feb,
                                 yeartemp_mar, yeartemp_apr))
    if i == 1998:
        brs_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        brs_chl_SEPAPR = np.hstack((brs_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDN SEP-APR
wedn_cluster = chl[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_SEPAPR = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov,
                                 yeartemp_dec, yeartemp_jan, yeartemp_feb,
                                 yeartemp_mar, yeartemp_apr))
    if i == 1998:
        wedn_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        wedn_chl_SEPAPR = np.hstack((wedn_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# GES SEP-APR
ges_cluster = chl[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_SEPAPR = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov,
                                 yeartemp_dec, yeartemp_jan, yeartemp_feb,
                                 yeartemp_mar, yeartemp_apr))
    if i == 1998:
        ges_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        ges_chl_SEPAPR = np.hstack((ges_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDS SEP-APR
weds_cluster = chl[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_SEPAPR = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov,
                                 yeartemp_dec, yeartemp_jan, yeartemp_feb,
                                 yeartemp_mar, yeartemp_apr))
    if i == 1998:
        weds_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        weds_chl_SEPAPR = np.hstack((weds_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# SAM SEP-APR
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
#%% Correlations DJF
stats.spearmanr(dra_chl_SEPAPR, sam_SEPAPR, nan_policy='omit') # **
stats.spearmanr(brs_chl_SEPAPR, sam_SEPAPR, nan_policy='omit') # *
stats.spearmanr(wedn_chl_SEPAPR, sam_SEPAPR, nan_policy='omit') # **
stats.spearmanr(ges_chl_SEPAPR, sam_SEPAPR, nan_policy='omit') #
stats.spearmanr(weds_chl_SEPAPR, sam_SEPAPR, nan_policy='omit') # *




