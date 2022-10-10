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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia')
### Load data 1998-2020
fh = np.load('seaice_19972021_10km.npz', allow_pickle=True)
lat = fh['lat'][100:]
lon = fh['lon'][30:250]
seaice = fh['seaice'][100:, 30:250, :]
seaice = seaice*100
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
#seaice[seaice > 50] = 50
# Load clusters
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('antarcticpeninsula_cluster.npz',allow_pickle = True)
clusters = fh['clusters']
#%% Separar para o cluster 1 (Weddell)
gerlache_cluster = seaice[clusters == 2,:]
gerlache_cluster = np.nanmean(gerlache_cluster,0)
#gerlache_cluster = np.where(gerlache_cluster > np.nanmedian(gerlache_cluster)-np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
#gerlache_cluster = np.where(gerlache_cluster < np.nanmedian(gerlache_cluster)+np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
seaice_persistence_gerlache = np.empty((np.size(np.arange(1997, 2021))))*np.nan
seaice_days_gerlache = np.empty((np.size(np.arange(1997, 2021))))*np.nan
seaice_init_gerlache = np.empty((np.size(np.arange(1997, 2021))))*np.nan
seaice_term_gerlache = np.empty((np.size(np.arange(1997, 2021))))*np.nan
for i, item in enumerate(np.arange(1997, 2021)):
    yeartemp_mardec = gerlache_cluster[(time_date_years == item) & ((time_date_months == 3) | (time_date_months == 4) |
                                                                (time_date_months == 5) | (time_date_months == 6) | (time_date_months == 7) |
                                                                (time_date_months == 8) | (time_date_months == 9) | (time_date_months == 10)
                                                                | (time_date_months == 11) | (time_date_months == 12)
                                                                )]
    yeartemp_janmar = gerlache_cluster[(time_date_years == item+1) & ((time_date_months == 1) |
                                                                (time_date_months == 2) | (time_date_months == 3))]
    yeartemp_data = np.hstack((yeartemp_mardec[14:], yeartemp_janmar[:-17]))
    yeartimetemp_mardec = time_date[(time_date_years == item) & ((time_date_months == 3) | (time_date_months == 4) |
                                                                (time_date_months == 5) | (time_date_months == 6) | (time_date_months == 7) |
                                                                (time_date_months == 8) | (time_date_months == 9) | (time_date_months == 10)
                                                                | (time_date_months == 11) | (time_date_months == 12)
                                                                )]
    yeartimetemp_janmar = time_date[(time_date_years == item+1) & ((time_date_months == 1) |
                                                                (time_date_months == 2) | (time_date_months == 3))]
    yeartimetemp_data = np.hstack((yeartimetemp_mardec[14:], yeartimetemp_janmar[:-17]))
    # Check if there is sea ice throughout the year
#    if (any(x > 15 for x in yeartemp_febapr) == False) | (np.size(yeartemp_febapr[yeartemp_febapr > 15]) < 5):
#        seaice_duration_gerlache[i] = 0
#        continue
#    if all(x > 15 for x in yeartemp_febapr) == True:
#        seaice_duration_gerlache[i] = 365
#        continue
    # Check sea ice initiation/advance (>15%)
    ice_index = yeartemp_data > 15
    breaker_a = 0
    for l, ind in enumerate(ice_index):
        if ind == False:
            continue
        elif (breaker_a == 0) & (all(x == True for x in ice_index[l:l+5])):
            seaice_init_index = l
            seaice_init_date = yeartimetemp_data[l]
            breaker_a=1
    # Check final/retraction of  sea ice (>15%)
    ice_index_fin = ice_index[::-1]
    breaker_b = 0
    for l, ind in enumerate(ice_index_fin):
        if ind == False:
            continue
        elif (breaker_b == 0) & (all(x == True for x in ice_index_fin[l:l+5])):
            seaice_fin_index = l
            seaice_fin_date = yeartimetemp_data[-l-1]
            breaker_b=1
    # Calculate duration of sea ice
    if breaker_a == 1 & breaker_b == 1:
        seaice_fin_date_index = np.where(yeartimetemp_data == seaice_fin_date)[0][0]
        seaice_init_date_index = np.where(yeartimetemp_data == seaice_init_date)[0][0]
        seaice_days_gerlache[i] = np.sum(ice_index[seaice_init_date_index:seaice_fin_date_index])
        seaice_persistence_gerlache[i] = (np.sum(ice_index[seaice_init_date_index:seaice_fin_date_index])/(seaice_fin_date - seaice_init_date).days)*100
        seaice_init_gerlache[i] = seaice_init_date.timetuple().tm_yday
        seaice_term_gerlache[i] = seaice_fin_date.timetuple().tm_yday   
#%%
seaice_term_gerlache_fake = np.empty_like(seaice_term_gerlache)
for i in range(0, len(seaice_term_gerlache_fake)):
    if seaice_term_gerlache[i] <80:
        seaice_term_gerlache_fake[i] = seaice_term_gerlache[i] + 365
# Calculate linear trends for each period
# Sea Ice Init
slope_gerlache_seaiceinit, _, rvalue_gerlache_seaiceinit, pvalue_gerlache_seaiceinit, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_init_gerlache)], seaice_init_gerlache[~np.isnan(seaice_init_gerlache)])
# Sea Ice Term
slope_gerlache_seaiceterm, _, rvalue_gerlache_seaiceterm, pvalue_gerlache_seaiceterm, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_term_gerlache_fake)], seaice_term_gerlache_fake[~np.isnan(seaice_term_gerlache_fake)])
# November
slope_gerlache_seaicedays, _, rvalue_gerlache_seaicedays, pvalue_gerlache_seaicedays, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_days_gerlache)], seaice_days_gerlache[~np.isnan(seaice_days_gerlache)])
# December
slope_gerlache_seaicepers, _, rvalue_gerlache_seaicepers, pvalue_gerlache_seaicepers, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_persistence_gerlache)], seaice_persistence_gerlache[~np.isnan(seaice_persistence_gerlache)])
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
#%% Separar para o cluster 2 (Gerlache)
gerlache_cluster = seaice[clusters == 2,:]
gerlache_cluster = np.nanmean(gerlache_cluster,0)
#gerlache_cluster = np.where(gerlache_cluster > np.nanmedian(gerlache_cluster)-np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
#gerlache_cluster = np.where(gerlache_cluster < np.nanmedian(gerlache_cluster)+np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
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
bransfield_cluster = seaice[clusters == 4,:]
bransfield_cluster = np.nanmean(bransfield_cluster,0)
#bransfield_cluster = np.where(bransfield_cluster > np.nanmedian(bransfield_cluster)-np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
#bransfield_cluster = np.where(bransfield_cluster < np.nanmedian(bransfield_cluster)+np.nanstd(bransfield_cluster)*3, bransfield_cluster, np.nan)
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
oceanic_cluster = seaice[clusters == 3,:]
oceanic_cluster = np.nanmean(oceanic_cluster,0)
#oceanic_cluster = np.where(oceanic_cluster > np.nanmedian(oceanic_cluster)-np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
#oceanic_cluster = np.where(oceanic_cluster < np.nanmedian(oceanic_cluster)+np.nanstd(oceanic_cluster)*3, oceanic_cluster, np.nan)
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
annot_ori[0][0] = annot_ori[0][0] + '\n★' + '★' + '★'
annot_ori[0][6] = annot_ori[0][2] + '\n★'
#%%
fig, ax = plt.subplots(figsize=(25, 3))
sns.heatmap([weddell_slopes, gerlache_slopes, bransfield_slopes, oceanic_slopes], square=True, annot=annot_ori, fmt='',
            vmin=-1, vmax=1, cmap=plt.cm.seismic, cbar_kws={"fraction":0.019, "pad":0.05})
plt.yticks(ticks=np.arange(0.5, 4), labels=['WED', 'GER', 'BRA', 'OCE'], fontsize=12, rotation = 360)
plt.xticks(ticks=np.arange(0.5,10), labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'NOVFEB', 'SEPAPR'], fontsize=12)
plt.xlabel('Period', fontsize=14)
plt.ylabel('Region', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\seaice\\trendspermonth_heatmap.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()