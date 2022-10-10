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
import matplotlib.patches as mpatches
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
weddell_cluster = np.where(weddell_cluster > np.nanmedian(weddell_cluster)-np.nanstd(weddell_cluster)*3, weddell_cluster, np.nan)
weddell_cluster = np.where(weddell_cluster < np.nanmedian(weddell_cluster)+np.nanstd(weddell_cluster)*3, weddell_cluster, np.nan)
### Calculate summer (November-February means for 1998-2005)
for i in np.arange(1998, 2006):
    yeartemp_nov = weddell_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weddell_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weddell_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weddell_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_19982005 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 1998:
        years_summermeans_19982005 = yeartemp_summermean_19982005
    else:
        years_summermeans_19982005 = np.hstack((years_summermeans_19982005, yeartemp_summermean_19982005))
### Calculate summer (November-February means for 2006-2014)
for i in np.arange(2006, 2014):
    yeartemp_nov = weddell_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weddell_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weddell_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weddell_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_20062013 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 2006:
        years_summermeans_20062013 = yeartemp_summermean_20062013
    else:
        years_summermeans_20062013 = np.hstack((years_summermeans_20062013, yeartemp_summermean_20062013))
### Calculate summer (November-February means for 2006-2014)
for i in np.arange(2014, 2022):
    yeartemp_nov = weddell_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weddell_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weddell_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weddell_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_20142021 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 2014:
        years_summermeans_20142021 = yeartemp_summermean_20142021
    else:
        years_summermeans_20142021 = np.hstack((years_summermeans_20142021, yeartemp_summermean_20142021))
years_summermeans_1998_2021 = np.hstack((years_summermeans_19982005, years_summermeans_20062013, years_summermeans_20142021))
stats.kruskal(years_summermeans_19982005, years_summermeans_20062013, yeartemp_summermean_20142021)
# standardize between 0 and 1
years_summermeans_19982005_norm_weddell = (years_summermeans_19982005 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_19982005_norm_weddell_mean = np.nanmean(years_summermeans_19982005_norm_weddell)
years_summermeans_19982005_norm_weddell_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_19982005_norm_weddell)-1, loc=np.mean(years_summermeans_19982005_norm_weddell), scale=stats.sem(years_summermeans_19982005_norm_weddell)) 
years_summermeans_20062013_norm_weddell = (years_summermeans_20062013 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_20062013_norm_weddell_mean = np.nanmean(years_summermeans_20062013_norm_weddell)
years_summermeans_20062013_norm_weddell_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_20062013_norm_weddell)-1, loc=np.mean(years_summermeans_20062013_norm_weddell), scale=stats.sem(years_summermeans_20062013_norm_weddell)) 
years_summermeans_20142021_norm_weddell = (years_summermeans_20142021 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_20142021_norm_weddell_mean = np.nanmean(years_summermeans_20142021_norm_weddell)
years_summermeans_20142021_norm_weddell_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_20142021_norm_weddell)-1, loc=np.mean(years_summermeans_20142021_norm_weddell), scale=stats.sem(years_summermeans_20142021_norm_weddell)) 
#%% Separar para o cluster 2 (Gerlache)
gerlache_cluster = chl[clusters == 2,:]
gerlache_cluster = np.nanmean(gerlache_cluster,0)
gerlache_cluster = np.where(gerlache_cluster > np.nanmedian(gerlache_cluster)-np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
gerlache_cluster = np.where(gerlache_cluster < np.nanmedian(gerlache_cluster)+np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
### Calculate summer (November-February means for 1998-2005)
for i in np.arange(1998, 2006):
    yeartemp_nov = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = gerlache_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = gerlache_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_19982005 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 1998:
        years_summermeans_19982005 = yeartemp_summermean_19982005
    else:
        years_summermeans_19982005 = np.hstack((years_summermeans_19982005, yeartemp_summermean_19982005))
### Calculate summer (November-February means for 2006-2014)
for i in np.arange(2006, 2014):
    yeartemp_nov = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = gerlache_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = gerlache_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_20062013 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 2006:
        years_summermeans_20062013 = yeartemp_summermean_20062013
    else:
        years_summermeans_20062013 = np.hstack((years_summermeans_20062013, yeartemp_summermean_20062013))
### Calculate summer (November-February means for 2006-2014)
for i in np.arange(2014, 2022):
    yeartemp_nov = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = gerlache_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = gerlache_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_20142021 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 2014:
        years_summermeans_20142021 = yeartemp_summermean_20142021
    else:
        years_summermeans_20142021 = np.hstack((years_summermeans_20142021, yeartemp_summermean_20142021))
years_summermeans_1998_2021 = np.hstack((years_summermeans_19982005, years_summermeans_20062013, years_summermeans_20142021))
stats.kruskal(years_summermeans_19982005, years_summermeans_20062013, yeartemp_summermean_20142021)
# standardize between 0 and 1
years_summermeans_19982005_norm_gerlache = (years_summermeans_19982005 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_19982005_norm_gerlache_mean = np.nanmean(years_summermeans_19982005_norm_gerlache)
years_summermeans_19982005_norm_gerlache_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_19982005_norm_gerlache)-1, loc=np.mean(years_summermeans_19982005_norm_gerlache), scale=stats.sem(years_summermeans_19982005_norm_gerlache)) 
years_summermeans_20062013_norm_gerlache = (years_summermeans_20062013 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_20062013_norm_gerlache_mean = np.nanmean(years_summermeans_20062013_norm_gerlache)
years_summermeans_20062013_norm_gerlache_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_20062013_norm_gerlache)-1, loc=np.mean(years_summermeans_20062013_norm_gerlache), scale=stats.sem(years_summermeans_20062013_norm_gerlache)) 
years_summermeans_20142021_norm_gerlache = (years_summermeans_20142021 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_20142021_norm_gerlache_mean = np.nanmean(years_summermeans_20142021_norm_gerlache)
years_summermeans_20142021_norm_gerlache_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_20142021_norm_gerlache)-1, loc=np.mean(years_summermeans_20142021_norm_gerlache), scale=stats.sem(years_summermeans_20142021_norm_gerlache)) 
#%% Separar para o cluster 4 (Bransfield)
bransfield_cluster = chl[clusters == 4,:]
bransfield_cluster = np.nanmean(bransfield_cluster,0)
bransfield_cluster = np.where(bransfield_cluster > np.nanmedian(bransfield_cluster)-np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
bransfield_cluster = np.where(bransfield_cluster < np.nanmedian(bransfield_cluster)+np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
### Calculate summer (November-February means for 1998-2005)
for i in np.arange(1998, 2006):
    yeartemp_nov = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = bransfield_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = bransfield_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_19982005 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 1998:
        years_summermeans_19982005 = yeartemp_summermean_19982005
    else:
        years_summermeans_19982005 = np.hstack((years_summermeans_19982005, yeartemp_summermean_19982005))
### Calculate summer (November-February means for 2006-2014)
for i in np.arange(2006, 2014):
    yeartemp_nov = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = bransfield_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = bransfield_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_20062013 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 2006:
        years_summermeans_20062013 = yeartemp_summermean_20062013
    else:
        years_summermeans_20062013 = np.hstack((years_summermeans_20062013, yeartemp_summermean_20062013))
### Calculate summer (November-February means for 2006-2014)
for i in np.arange(2014, 2022):
    yeartemp_nov = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = bransfield_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = bransfield_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_20142021 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 2014:
        years_summermeans_20142021 = yeartemp_summermean_20142021
    else:
        years_summermeans_20142021 = np.hstack((years_summermeans_20142021, yeartemp_summermean_20142021))
years_summermeans_1998_2021 = np.hstack((years_summermeans_19982005, years_summermeans_20062013, years_summermeans_20142021))
stats.kruskal(years_summermeans_19982005, years_summermeans_20062013, yeartemp_summermean_20142021)

# standardize between 0 and 1
years_summermeans_19982005_norm_bransfield = (years_summermeans_19982005 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_19982005_norm_bransfield_mean = np.nanmean(years_summermeans_19982005_norm_bransfield)
years_summermeans_19982005_norm_bransfield_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_19982005_norm_bransfield)-1, loc=np.mean(years_summermeans_19982005_norm_bransfield), scale=stats.sem(years_summermeans_19982005_norm_bransfield)) 
years_summermeans_20062013_norm_bransfield = (years_summermeans_20062013 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_20062013_norm_bransfield_mean = np.nanmean(years_summermeans_20062013_norm_bransfield)
years_summermeans_20062013_norm_bransfield_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_20062013_norm_bransfield)-1, loc=np.mean(years_summermeans_20062013_norm_bransfield), scale=stats.sem(years_summermeans_20062013_norm_bransfield)) 
years_summermeans_20142021_norm_bransfield = (years_summermeans_20142021 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_20142021_norm_bransfield_mean = np.nanmean(years_summermeans_20142021_norm_bransfield)
years_summermeans_20142021_norm_bransfield_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_20142021_norm_bransfield)-1, loc=np.mean(years_summermeans_20142021_norm_bransfield), scale=stats.sem(years_summermeans_20142021_norm_bransfield)) 
#%% Separar para o cluster 3 (Oceanic)
oceanic_cluster = chl[clusters == 3,:]
oceanic_cluster = np.nanmean(oceanic_cluster,0)
oceanic_cluster = np.where(oceanic_cluster > np.nanmedian(oceanic_cluster)-np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
oceanic_cluster = np.where(oceanic_cluster < np.nanmedian(oceanic_cluster)+np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
### Calculate summer (November-February means for 1998-2005)
for i in np.arange(1998, 2006):
    yeartemp_nov = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = oceanic_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = oceanic_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_19982005 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 1998:
        years_summermeans_19982005 = yeartemp_summermean_19982005
    else:
        years_summermeans_19982005 = np.hstack((years_summermeans_19982005, yeartemp_summermean_19982005))
### Calculate summer (November-February means for 2006-2014)
for i in np.arange(2006, 2014):
    yeartemp_nov = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = oceanic_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = oceanic_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_20062013 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 2006:
        years_summermeans_20062013 = yeartemp_summermean_20062013
    else:
        years_summermeans_20062013 = np.hstack((years_summermeans_20062013, yeartemp_summermean_20062013))
### Calculate summer (November-February means for 2006-2014)
for i in np.arange(2014, 2022):
    yeartemp_nov = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = oceanic_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = oceanic_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean_20142021 = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    if i == 2014:
        years_summermeans_20142021 = yeartemp_summermean_20142021
    else:
        years_summermeans_20142021 = np.hstack((years_summermeans_20142021, yeartemp_summermean_20142021))
years_summermeans_1998_2021 = np.hstack((years_summermeans_19982005, years_summermeans_20062013, years_summermeans_20142021))
stats.kruskal(years_summermeans_19982005, years_summermeans_20062013, yeartemp_summermean_20142021)

# standardize between 0 and 1
years_summermeans_19982005_norm_oceanic = (years_summermeans_19982005 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_19982005_norm_oceanic_mean = np.nanmean(years_summermeans_19982005_norm_oceanic)
years_summermeans_19982005_norm_oceanic_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_19982005_norm_oceanic)-1, loc=np.mean(years_summermeans_19982005_norm_oceanic), scale=stats.sem(years_summermeans_19982005_norm_oceanic)) 
years_summermeans_20062013_norm_oceanic = (years_summermeans_20062013 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_20062013_norm_oceanic_mean = np.nanmean(years_summermeans_20062013_norm_oceanic)
years_summermeans_20062013_norm_oceanic_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_20062013_norm_oceanic)-1, loc=np.mean(years_summermeans_20062013_norm_oceanic), scale=stats.sem(years_summermeans_20062013_norm_oceanic)) 
years_summermeans_20142021_norm_oceanic = (years_summermeans_20142021 - np.nanmin(years_summermeans_1998_2021)) / (np.nanmax(years_summermeans_1998_2021) - np.nanmin(years_summermeans_1998_2021))
years_summermeans_20142021_norm_oceanic_mean = np.nanmean(years_summermeans_20142021_norm_oceanic)
years_summermeans_20142021_norm_oceanic_mean_confinterval95 = stats.t.interval(alpha=0.95, df=len(years_summermeans_20142021_norm_oceanic)-1, loc=np.mean(years_summermeans_20142021_norm_oceanic), scale=stats.sem(years_summermeans_20142021_norm_oceanic)) 
#%% Calculate kruskal wallis



#%%
# Weddell
plt.plot((years_summermeans_19982005_norm_weddell_mean_confinterval95[0], years_summermeans_19982005_norm_weddell_mean_confinterval95[1]), (2.1, 2.1), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_19982005_norm_weddell_mean, 2.1, marker='o', c=[43/256, 131/256, 186/256, 1], zorder=1, s=150)
plt.plot((years_summermeans_20062013_norm_weddell_mean_confinterval95[0], years_summermeans_20062013_norm_weddell_mean_confinterval95[1]), (2, 2), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20062013_norm_weddell_mean, 2, marker='^', c=[43/256, 131/256, 186/256, 1], zorder=1, s=150)
plt.plot((years_summermeans_20142021_norm_weddell_mean_confinterval95[0], years_summermeans_20142021_norm_weddell_mean_confinterval95[1]), (1.9, 1.9), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20142021_norm_weddell_mean, 1.9, marker='s', c=[43/256, 131/256, 186/256, 1], zorder=1, s=150)
#Gerlache
plt.plot((years_summermeans_19982005_norm_gerlache_mean_confinterval95[0], years_summermeans_19982005_norm_gerlache_mean_confinterval95[1]), (1.6, 1.6), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_19982005_norm_gerlache_mean, 1.6, marker='o', c=[215/256, 25/256, 28/256, 1], zorder=1, s=150)
plt.plot((years_summermeans_20062013_norm_gerlache_mean_confinterval95[0], years_summermeans_20062013_norm_gerlache_mean_confinterval95[1]), (1.5, 1.5), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20062013_norm_gerlache_mean, 1.5, marker='^', c=[215/256, 25/256, 28/256, 1], zorder=1, s=150)
plt.plot((years_summermeans_20142021_norm_gerlache_mean_confinterval95[0], years_summermeans_20142021_norm_gerlache_mean_confinterval95[1]), (1.4, 1.4), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20142021_norm_gerlache_mean, 1.4, marker='s', c=[215/256, 25/256, 28/256, 1], zorder=1, s=150)
#Bransfield
plt.plot((years_summermeans_19982005_norm_bransfield_mean_confinterval95[0], years_summermeans_19982005_norm_bransfield_mean_confinterval95[1]), (1.1, 1.1), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_19982005_norm_bransfield_mean, 1.1, marker='o', c='#9800cb', zorder=1, s=150)
plt.plot((years_summermeans_20062013_norm_bransfield_mean_confinterval95[0], years_summermeans_20062013_norm_bransfield_mean_confinterval95[1]), (1, 1), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20062013_norm_bransfield_mean, 1, marker='^', c='#9800cb', zorder=1, s=150)
plt.plot((years_summermeans_20142021_norm_bransfield_mean_confinterval95[0], years_summermeans_20142021_norm_bransfield_mean_confinterval95[1]), (.9, .9), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20142021_norm_bransfield_mean, .9, marker='s', c='#9800cb', zorder=1, s=150)
#Oceanic
plt.plot((years_summermeans_19982005_norm_oceanic_mean_confinterval95[0], years_summermeans_19982005_norm_oceanic_mean_confinterval95[1]), (.6, .6), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_19982005_norm_oceanic_mean, .6, marker='o', c='#d09c26', zorder=1, s=150)
plt.plot((years_summermeans_20062013_norm_oceanic_mean_confinterval95[0], years_summermeans_20062013_norm_oceanic_mean_confinterval95[1]), (.5, .5), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20062013_norm_oceanic_mean, .5, marker='^', c='#d09c26', zorder=1, s=150)
plt.plot((years_summermeans_20142021_norm_oceanic_mean_confinterval95[0], years_summermeans_20142021_norm_oceanic_mean_confinterval95[1]), (.4, .4), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20142021_norm_oceanic_mean, .4, marker='s', c='#d09c26', zorder=1, s=150)
# Customizing figure
from matplotlib.lines import Line2D
plt.xlim(0,.8)
plt.yticks(ticks = [.5, 1, 1.5, 2],
           labels = ['OCE', 'BRA', 'GER', 'WED'], fontsize=14)
plt.xticks(ticks= [0, .4, .8])
legend_elements = [Line2D([0], [0], marker='o', color='w', label='1998-2005',
       markerfacecolor='k', markersize=8, alpha=0.7) ,
                   Line2D([0], [0], marker='^', color='w', label='2006-2013',
                          markerfacecolor='k', markersize=8, alpha=0.7),
                   Line2D([0], [0], marker='s', color='w', label='2014-2021',
                          markerfacecolor='k', markersize=8, alpha=0.7)]
plt.ylim(.1, 2.2)
plt.legend(handles=legend_elements, loc=8, fontsize=12, ncol=3, columnspacing=0.1,
           borderpad = 0.2, labelspacing=0.1, handletextpad=.01)
plt.xlabel('Standardized Chl-$\it{a}$ (November-February)', fontsize=16)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\comparisonbetween8yearsperiods.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()