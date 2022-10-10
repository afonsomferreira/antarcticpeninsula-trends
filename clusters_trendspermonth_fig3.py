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
for i in np.arange(1998, 2022):
    yeartemp_sep = weddell_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weddell_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weddell_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weddell_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weddell_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weddell_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weddell_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weddell_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_novfeb = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    yeartemp_sepapr = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                     yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr)))
    if i == 1998:
        years_sep_19982021 = np.nanmean(yeartemp_sep)
        years_oct_19982021 = np.nanmean(yeartemp_oct)
        years_nov_19982021 = np.nanmean(yeartemp_nov)
        years_dec_19982021 = np.nanmean(yeartemp_dec)
        years_jan_19982021 = np.nanmean(yeartemp_jan)
        years_feb_19982021 = np.nanmean(yeartemp_feb)
        years_mar_19982021 = np.nanmean(yeartemp_mar)
        years_apr_19982021 = np.nanmean(yeartemp_apr)
        years_novfeb_19982021 = yeartemp_novfeb
        years_sepapr_19982021 = yeartemp_sepapr
    else:
        years_sep_19982021 = np.hstack((years_sep_19982021, np.nanmean(yeartemp_sep)))
        years_oct_19982021 = np.hstack((years_oct_19982021, np.nanmean(yeartemp_oct)))
        years_nov_19982021 = np.hstack((years_nov_19982021, np.nanmean(yeartemp_nov)))
        years_dec_19982021 = np.hstack((years_dec_19982021, np.nanmean(yeartemp_dec)))
        years_jan_19982021 = np.hstack((years_jan_19982021, np.nanmean(yeartemp_jan)))
        years_feb_19982021 = np.hstack((years_feb_19982021, np.nanmean(yeartemp_feb)))
        years_mar_19982021 = np.hstack((years_mar_19982021, np.nanmean(yeartemp_mar)))
        years_apr_19982021 = np.hstack((years_apr_19982021, np.nanmean(yeartemp_apr)))
        years_novfeb_19982021 = np.hstack((years_novfeb_19982021, yeartemp_novfeb))
        years_sepapr_19982021 = np.hstack((years_sepapr_19982021, yeartemp_sepapr))
# Calculate linear trends for each period
# September
slope_weddell_sep, _, rvalue_weddell_sep, pvalue_weddell_sep, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_sep_19982021)], years_sep_19982021[~np.isnan(years_sep_19982021)])
# October
slope_weddell_oct, _, rvalue_weddell_oct, pvalue_weddell_oct, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_oct_19982021)], years_oct_19982021[~np.isnan(years_oct_19982021)])
# November
slope_weddell_nov, _, rvalue_weddell_nov, pvalue_weddell_nov, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_nov_19982021)], years_nov_19982021[~np.isnan(years_nov_19982021)])
# December
slope_weddell_dec, _, rvalue_weddell_dec, pvalue_weddell_dec, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_dec_19982021)], years_dec_19982021[~np.isnan(years_dec_19982021)])
# January
slope_weddell_jan, _, rvalue_weddell_jan, pvalue_weddell_jan, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_jan_19982021)], years_jan_19982021[~np.isnan(years_jan_19982021)])
# February
slope_weddell_feb, _, rvalue_weddell_feb, pvalue_weddell_feb, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_feb_19982021)], years_feb_19982021[~np.isnan(years_feb_19982021)])
# March
slope_weddell_mar, _, rvalue_weddell_mar, pvalue_weddell_mar, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_mar_19982021)], years_mar_19982021[~np.isnan(years_mar_19982021)])
# April # More than 5 years with nans
#slope_weddell_apr, _, rvalue_weddell_apr, pvalue_weddell_apr, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_apr_19982021)], years_apr_19982021[~np.isnan(years_apr_19982021)])
slope_weddell_apr = np.nan
# November-February
slope_weddell_novfeb, _, rvalue_weddell_novfeb, pvalue_weddell_novfeb, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_novfeb_19982021)], years_novfeb_19982021[~np.isnan(years_novfeb_19982021)])
# September-April
slope_weddell_sepapr, _, rvalue_weddell_sepapr, pvalue_weddell_sepapr, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_sepapr_19982021)], years_sepapr_19982021[~np.isnan(years_sepapr_19982021)])
# Prepare data for heatmap
weddell_slopes = np.hstack((slope_weddell_sep, slope_weddell_oct, slope_weddell_nov, slope_weddell_dec,
                            slope_weddell_jan, slope_weddell_feb, slope_weddell_mar, slope_weddell_apr,
                            slope_weddell_novfeb, slope_weddell_sepapr))
weddell_slopes = np.around(weddell_slopes, 3)
#%% Separar para o cluster 2 (Gerlache)
gerlache_cluster = chl[clusters == 2,:]
gerlache_cluster = np.nanmean(gerlache_cluster,0)
gerlache_cluster = np.where(gerlache_cluster > np.nanmedian(gerlache_cluster)-np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
gerlache_cluster = np.where(gerlache_cluster < np.nanmedian(gerlache_cluster)+np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
for i in np.arange(1998, 2022):
    yeartemp_sep = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = gerlache_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = gerlache_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = gerlache_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = gerlache_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = gerlache_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_novfeb = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    yeartemp_sepapr = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                     yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr)))
    if i == 1998:
        years_sep_19982021 = np.nanmean(yeartemp_sep)
        years_oct_19982021 = np.nanmean(yeartemp_oct)
        years_nov_19982021 = np.nanmean(yeartemp_nov)
        years_dec_19982021 = np.nanmean(yeartemp_dec)
        years_jan_19982021 = np.nanmean(yeartemp_jan)
        years_feb_19982021 = np.nanmean(yeartemp_feb)
        years_mar_19982021 = np.nanmean(yeartemp_mar)
        years_apr_19982021 = np.nanmean(yeartemp_apr)
        years_novfeb_19982021 = yeartemp_novfeb
        years_sepapr_19982021 = yeartemp_sepapr
    else:
        years_sep_19982021 = np.hstack((years_sep_19982021, np.nanmean(yeartemp_sep)))
        years_oct_19982021 = np.hstack((years_oct_19982021, np.nanmean(yeartemp_oct)))
        years_nov_19982021 = np.hstack((years_nov_19982021, np.nanmean(yeartemp_nov)))
        years_dec_19982021 = np.hstack((years_dec_19982021, np.nanmean(yeartemp_dec)))
        years_jan_19982021 = np.hstack((years_jan_19982021, np.nanmean(yeartemp_jan)))
        years_feb_19982021 = np.hstack((years_feb_19982021, np.nanmean(yeartemp_feb)))
        years_mar_19982021 = np.hstack((years_mar_19982021, np.nanmean(yeartemp_mar)))
        years_apr_19982021 = np.hstack((years_apr_19982021, np.nanmean(yeartemp_apr)))
        years_novfeb_19982021 = np.hstack((years_novfeb_19982021, yeartemp_novfeb))
        years_sepapr_19982021 = np.hstack((years_sepapr_19982021, yeartemp_sepapr))
# Calculate linear trends for each period
# September
slope_gerlache_sep, _, rvalue_gerlache_sep, pvalue_gerlache_sep, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_sep_19982021)], years_sep_19982021[~np.isnan(years_sep_19982021)])
# October
slope_gerlache_oct, _, rvalue_gerlache_oct, pvalue_gerlache_oct, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_oct_19982021)], years_oct_19982021[~np.isnan(years_oct_19982021)])
# November
slope_gerlache_nov, _, rvalue_gerlache_nov, pvalue_gerlache_nov, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_nov_19982021)], years_nov_19982021[~np.isnan(years_nov_19982021)])
# December
slope_gerlache_dec, _, rvalue_gerlache_dec, pvalue_gerlache_dec, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_dec_19982021)], years_dec_19982021[~np.isnan(years_dec_19982021)])
# January
slope_gerlache_jan, _, rvalue_gerlache_jan, pvalue_gerlache_jan, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_jan_19982021)], years_jan_19982021[~np.isnan(years_jan_19982021)])
# February
slope_gerlache_feb, _, rvalue_gerlache_feb, pvalue_gerlache_feb, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_feb_19982021)], years_feb_19982021[~np.isnan(years_feb_19982021)])
# March
slope_gerlache_mar, _, rvalue_gerlache_mar, pvalue_gerlache_mar, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_mar_19982021)], years_mar_19982021[~np.isnan(years_mar_19982021)])
# April # More than 5 years with nans
#slope_gerlache_apr, _, rvalue_gerlache_apr, pvalue_gerlache_apr, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_apr_19982021)], years_apr_19982021[~np.isnan(years_apr_19982021)])
slope_gerlache_apr = np.nan
# November-February
slope_gerlache_novfeb, _, rvalue_gerlache_novfeb, pvalue_gerlache_novfeb, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_novfeb_19982021)], years_novfeb_19982021[~np.isnan(years_novfeb_19982021)])
# September-April
slope_gerlache_sepapr, _, rvalue_gerlache_sepapr, pvalue_gerlache_sepapr, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_sepapr_19982021)], years_sepapr_19982021[~np.isnan(years_sepapr_19982021)])
# Prepare data for heatmap
gerlache_slopes = np.hstack((slope_gerlache_sep, slope_gerlache_oct, slope_gerlache_nov, slope_gerlache_dec,
                            slope_gerlache_jan, slope_gerlache_feb, slope_gerlache_mar, slope_gerlache_apr,
                            slope_gerlache_novfeb, slope_gerlache_sepapr))
gerlache_slopes = np.around(gerlache_slopes, 3)
#%% Separar para o cluster 4 (Bransfield)
bransfield_cluster = chl[clusters == 4,:]
bransfield_cluster = np.nanmean(bransfield_cluster,0)
bransfield_cluster = np.where(bransfield_cluster > np.nanmedian(bransfield_cluster)-np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
bransfield_cluster = np.where(bransfield_cluster < np.nanmedian(bransfield_cluster)+np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
for i in np.arange(1998, 2022):
    yeartemp_sep = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = bransfield_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = bransfield_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = bransfield_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = bransfield_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = bransfield_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_novfeb = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    yeartemp_sepapr = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                     yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr)))
    if i == 1998:
        years_sep_19982021 = np.nanmean(yeartemp_sep)
        years_oct_19982021 = np.nanmean(yeartemp_oct)
        years_nov_19982021 = np.nanmean(yeartemp_nov)
        years_dec_19982021 = np.nanmean(yeartemp_dec)
        years_jan_19982021 = np.nanmean(yeartemp_jan)
        years_feb_19982021 = np.nanmean(yeartemp_feb)
        years_mar_19982021 = np.nanmean(yeartemp_mar)
        years_apr_19982021 = np.nanmean(yeartemp_apr)
        years_novfeb_19982021 = yeartemp_novfeb
        years_sepapr_19982021 = yeartemp_sepapr
    else:
        years_sep_19982021 = np.hstack((years_sep_19982021, np.nanmean(yeartemp_sep)))
        years_oct_19982021 = np.hstack((years_oct_19982021, np.nanmean(yeartemp_oct)))
        years_nov_19982021 = np.hstack((years_nov_19982021, np.nanmean(yeartemp_nov)))
        years_dec_19982021 = np.hstack((years_dec_19982021, np.nanmean(yeartemp_dec)))
        years_jan_19982021 = np.hstack((years_jan_19982021, np.nanmean(yeartemp_jan)))
        years_feb_19982021 = np.hstack((years_feb_19982021, np.nanmean(yeartemp_feb)))
        years_mar_19982021 = np.hstack((years_mar_19982021, np.nanmean(yeartemp_mar)))
        years_apr_19982021 = np.hstack((years_apr_19982021, np.nanmean(yeartemp_apr)))
        years_novfeb_19982021 = np.hstack((years_novfeb_19982021, yeartemp_novfeb))
        years_sepapr_19982021 = np.hstack((years_sepapr_19982021, yeartemp_sepapr))
# Calculate linear trends for each period
# September
slope_bransfield_sep, _, rvalue_bransfield_sep, pvalue_bransfield_sep, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_sep_19982021)], years_sep_19982021[~np.isnan(years_sep_19982021)])
# October
slope_bransfield_oct, _, rvalue_bransfield_oct, pvalue_bransfield_oct, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_oct_19982021)], years_oct_19982021[~np.isnan(years_oct_19982021)])
# November
slope_bransfield_nov, _, rvalue_bransfield_nov, pvalue_bransfield_nov, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_nov_19982021)], years_nov_19982021[~np.isnan(years_nov_19982021)])
# December
slope_bransfield_dec, _, rvalue_bransfield_dec, pvalue_bransfield_dec, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_dec_19982021)], years_dec_19982021[~np.isnan(years_dec_19982021)])
# January
slope_bransfield_jan, _, rvalue_bransfield_jan, pvalue_bransfield_jan, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_jan_19982021)], years_jan_19982021[~np.isnan(years_jan_19982021)])
# February
slope_bransfield_feb, _, rvalue_bransfield_feb, pvalue_bransfield_feb, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_feb_19982021)], years_feb_19982021[~np.isnan(years_feb_19982021)])
# March
slope_bransfield_mar, _, rvalue_bransfield_mar, pvalue_bransfield_mar, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_mar_19982021)], years_mar_19982021[~np.isnan(years_mar_19982021)])
# April # More than 5 years with nans
slope_bransfield_apr, _, rvalue_bransfield_apr, pvalue_bransfield_apr, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_apr_19982021)], years_apr_19982021[~np.isnan(years_apr_19982021)])
#slope_bransfield_apr = np.nan
# November-February
slope_bransfield_novfeb, _, rvalue_bransfield_novfeb, pvalue_bransfield_novfeb, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_novfeb_19982021)], years_novfeb_19982021[~np.isnan(years_novfeb_19982021)])
# September-April
slope_bransfield_sepapr, _, rvalue_bransfield_sepapr, pvalue_bransfield_sepapr, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_sepapr_19982021)], years_sepapr_19982021[~np.isnan(years_sepapr_19982021)])
# Prepare data for heatmap
bransfield_slopes = np.hstack((slope_bransfield_sep, slope_bransfield_oct, slope_bransfield_nov, slope_bransfield_dec,
                            slope_bransfield_jan, slope_bransfield_feb, slope_bransfield_mar, slope_bransfield_apr,
                            slope_bransfield_novfeb, slope_bransfield_sepapr))
bransfield_slopes = np.around(bransfield_slopes, 3)
#%% Separar para o cluster 3 (Oceanic)
oceanic_cluster = chl[clusters == 3,:]
oceanic_cluster = np.nanmean(oceanic_cluster,0)
oceanic_cluster = np.where(oceanic_cluster > np.nanmedian(oceanic_cluster)-np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
oceanic_cluster = np.where(oceanic_cluster < np.nanmedian(oceanic_cluster)+np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
for i in np.arange(1998, 2022):
    yeartemp_sep = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = oceanic_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = oceanic_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = oceanic_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = oceanic_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = oceanic_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_novfeb = np.nanmean(np.hstack((yeartemp_nov, yeartemp_dec, yeartemp_jan, yeartemp_feb)))
    yeartemp_sepapr = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                     yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr)))
    if i == 1998:
        years_sep_19982021 = np.nanmean(yeartemp_sep)
        years_oct_19982021 = np.nanmean(yeartemp_oct)
        years_nov_19982021 = np.nanmean(yeartemp_nov)
        years_dec_19982021 = np.nanmean(yeartemp_dec)
        years_jan_19982021 = np.nanmean(yeartemp_jan)
        years_feb_19982021 = np.nanmean(yeartemp_feb)
        years_mar_19982021 = np.nanmean(yeartemp_mar)
        years_apr_19982021 = np.nanmean(yeartemp_apr)
        years_novfeb_19982021 = yeartemp_novfeb
        years_sepapr_19982021 = yeartemp_sepapr
    else:
        years_sep_19982021 = np.hstack((years_sep_19982021, np.nanmean(yeartemp_sep)))
        years_oct_19982021 = np.hstack((years_oct_19982021, np.nanmean(yeartemp_oct)))
        years_nov_19982021 = np.hstack((years_nov_19982021, np.nanmean(yeartemp_nov)))
        years_dec_19982021 = np.hstack((years_dec_19982021, np.nanmean(yeartemp_dec)))
        years_jan_19982021 = np.hstack((years_jan_19982021, np.nanmean(yeartemp_jan)))
        years_feb_19982021 = np.hstack((years_feb_19982021, np.nanmean(yeartemp_feb)))
        years_mar_19982021 = np.hstack((years_mar_19982021, np.nanmean(yeartemp_mar)))
        years_apr_19982021 = np.hstack((years_apr_19982021, np.nanmean(yeartemp_apr)))
        years_novfeb_19982021 = np.hstack((years_novfeb_19982021, yeartemp_novfeb))
        years_sepapr_19982021 = np.hstack((years_sepapr_19982021, yeartemp_sepapr))
# Calculate linear trends for each period
# September
slope_oceanic_sep, _, rvalue_oceanic_sep, pvalue_oceanic_sep, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_sep_19982021)], years_sep_19982021[~np.isnan(years_sep_19982021)])
# October
slope_oceanic_oct, _, rvalue_oceanic_oct, pvalue_oceanic_oct, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_oct_19982021)], years_oct_19982021[~np.isnan(years_oct_19982021)])
# November
slope_oceanic_nov, _, rvalue_oceanic_nov, pvalue_oceanic_nov, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_nov_19982021)], years_nov_19982021[~np.isnan(years_nov_19982021)])
# December
slope_oceanic_dec, _, rvalue_oceanic_dec, pvalue_oceanic_dec, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_dec_19982021)], years_dec_19982021[~np.isnan(years_dec_19982021)])
# January
slope_oceanic_jan, _, rvalue_oceanic_jan, pvalue_oceanic_jan, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_jan_19982021)], years_jan_19982021[~np.isnan(years_jan_19982021)])
# February
slope_oceanic_feb, _, rvalue_oceanic_feb, pvalue_oceanic_feb, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_feb_19982021)], years_feb_19982021[~np.isnan(years_feb_19982021)])
# March
slope_oceanic_mar, _, rvalue_oceanic_mar, pvalue_oceanic_mar, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_mar_19982021)], years_mar_19982021[~np.isnan(years_mar_19982021)])
# April # More than 5 years with nans
slope_oceanic_apr, _, rvalue_oceanic_apr, pvalue_oceanic_apr, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_apr_19982021)], years_apr_19982021[~np.isnan(years_apr_19982021)])
#slope_oceanic_apr = np.nan
# November-February
slope_oceanic_novfeb, _, rvalue_oceanic_novfeb, pvalue_oceanic_novfeb, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_novfeb_19982021)], years_novfeb_19982021[~np.isnan(years_novfeb_19982021)])
# September-April
slope_oceanic_sepapr, _, rvalue_oceanic_sepapr, pvalue_oceanic_sepapr, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(years_sepapr_19982021)], years_sepapr_19982021[~np.isnan(years_sepapr_19982021)])
# Prepare data for heatmap
oceanic_slopes = np.hstack((slope_oceanic_sep, slope_oceanic_oct, slope_oceanic_nov, slope_oceanic_dec,
                            slope_oceanic_jan, slope_oceanic_feb, slope_oceanic_mar, slope_oceanic_apr,
                            slope_oceanic_novfeb, slope_oceanic_sepapr))
oceanic_slopes = np.around(oceanic_slopes, 3)

#%%
annot_ori = [[f"{val:.3f}"  for val in row] for row in [weddell_slopes, gerlache_slopes, bransfield_slopes, oceanic_slopes]]
annot_ori[0][0] = annot_ori[0][0] + '\n★'
annot_ori[1][0] = annot_ori[1][0] + '\n★' + '★'
annot_ori[1][3] = annot_ori[1][3] + '\n★'
annot_ori[1][6] = annot_ori[1][6] + '\n★' + '★'
annot_ori[1][9] = annot_ori[1][9] + '\n★'
annot_ori[2][0] = annot_ori[2][0] + '\n★'
annot_ori[2][3] = annot_ori[2][3] + '\n★'
annot_ori[2][5] = annot_ori[2][5] + '\n★'
annot_ori[2][6] = annot_ori[2][6] + '\n★' + '★' + '★'
annot_ori[2][7] = annot_ori[2][7] + '\n★' + '★' + '★'
annot_ori[2][8] = annot_ori[2][8] + '\n★'
annot_ori[2][9] = annot_ori[2][9] + '\n★' + '★' + '★'
annot_ori[3][6] = annot_ori[3][6] + '\n★' + '★'
#%%
fig, ax = plt.subplots(figsize=(25, 3))
sns.heatmap([weddell_slopes, gerlache_slopes, bransfield_slopes, oceanic_slopes], square=True, annot=annot_ori, fmt='',
            vmin=-0.05, vmax=.05, cmap=plt.cm.seismic, cbar_kws={"fraction":0.019, "pad":0.05})
plt.yticks(ticks=np.arange(0.5, 4), labels=['WED', 'GER', 'BRA', 'OCE'], fontsize=12, rotation = 360)
plt.xticks(ticks=np.arange(0.5,10), labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'NOVFEB', 'SEPAPR'], fontsize=12)
plt.xlabel('Period', fontsize=14)
plt.ylabel('Region', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\trendspermonth_heatmap_612.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%






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
plt.scatter(years_summermeans_19982005_norm_weddell_mean, 2.1, marker='o', c=[43/256, 131/256, 186/256, 1], zorder=1, s=90)
plt.plot((years_summermeans_20062013_norm_weddell_mean_confinterval95[0], years_summermeans_20062013_norm_weddell_mean_confinterval95[1]), (2, 2), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20062013_norm_weddell_mean, 2, marker='^', c=[43/256, 131/256, 186/256, 1], zorder=1, s=90)
plt.plot((years_summermeans_20142021_norm_weddell_mean_confinterval95[0], years_summermeans_20142021_norm_weddell_mean_confinterval95[1]), (1.9, 1.9), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20142021_norm_weddell_mean, 1.9, marker='s', c=[43/256, 131/256, 186/256, 1], zorder=1, s=90)
#Gerlache
plt.plot((years_summermeans_19982005_norm_gerlache_mean_confinterval95[0], years_summermeans_19982005_norm_gerlache_mean_confinterval95[1]), (1.6, 1.6), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_19982005_norm_gerlache_mean, 1.6, marker='o', c=[215/256, 25/256, 28/256, 1], zorder=1, s=90)
plt.plot((years_summermeans_20062013_norm_gerlache_mean_confinterval95[0], years_summermeans_20062013_norm_gerlache_mean_confinterval95[1]), (1.5, 1.5), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20062013_norm_gerlache_mean, 1.5, marker='^', c=[215/256, 25/256, 28/256, 1], zorder=1, s=90)
plt.plot((years_summermeans_20142021_norm_gerlache_mean_confinterval95[0], years_summermeans_20142021_norm_gerlache_mean_confinterval95[1]), (1.4, 1.4), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20142021_norm_gerlache_mean, 1.4, marker='s', c=[215/256, 25/256, 28/256, 1], zorder=1, s=90)
#Bransfield
plt.plot((years_summermeans_19982005_norm_bransfield_mean_confinterval95[0], years_summermeans_19982005_norm_bransfield_mean_confinterval95[1]), (1.1, 1.1), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_19982005_norm_bransfield_mean, 1.1, marker='o', c='#9800cb', zorder=1, s=90)
plt.plot((years_summermeans_20062013_norm_bransfield_mean_confinterval95[0], years_summermeans_20062013_norm_bransfield_mean_confinterval95[1]), (1, 1), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20062013_norm_bransfield_mean, 1, marker='^', c='#9800cb', zorder=1, s=90)
plt.plot((years_summermeans_20142021_norm_bransfield_mean_confinterval95[0], years_summermeans_20142021_norm_bransfield_mean_confinterval95[1]), (.9, .9), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20142021_norm_bransfield_mean, .9, marker='s', c='#9800cb', zorder=1, s=90)
#Oceanic
plt.plot((years_summermeans_19982005_norm_oceanic_mean_confinterval95[0], years_summermeans_19982005_norm_oceanic_mean_confinterval95[1]), (.6, .6), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_19982005_norm_oceanic_mean, .6, marker='o', c='#d09c26', zorder=1, s=90)
plt.plot((years_summermeans_20062013_norm_oceanic_mean_confinterval95[0], years_summermeans_20062013_norm_oceanic_mean_confinterval95[1]), (.5, .5), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20062013_norm_oceanic_mean, .5, marker='^', c='#d09c26', zorder=1, s=90)
plt.plot((years_summermeans_20142021_norm_oceanic_mean_confinterval95[0], years_summermeans_20142021_norm_oceanic_mean_confinterval95[1]), (.4, .4), 'k-', zorder=0, alpha=0.5)
plt.scatter(years_summermeans_20142021_norm_oceanic_mean, .4, marker='s', c='#d09c26', zorder=1, s=90)
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
plt.xlabel('Standardized Chl-a', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\comparisonbetween8yearsperiods.png'
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