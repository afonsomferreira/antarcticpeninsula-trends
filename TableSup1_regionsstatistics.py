# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:33:06 2020

@author: Afonso
"""
import os
import statistics
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
from scipy import integrate
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
def check_for_bloominit(yearly_timeseries):
    arr = yearly_timeseries.values.copy()                   # avoid mutating the original list
    counting = []                      # keep track of True indexes, to count them later
    for i in range(len(arr)):          # cycle by index
        is_last = i + 1 >= len(arr)    # True if this is the last index in the array
        if arr[i] == True:
            counting.append(i)         # add value to list if True
        if is_last or arr[i] == False: # when we are at the last entry, or find a False
            if len(counting) < 2:      # check the length of our True indexes, and if less than 6
                for j in counting:
                    arr[j] = False     # make each False
            counting = []
    return arr
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
# Load upscaled 4km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_4km.npz',allow_pickle = True)
clusters = fh['clusters']
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
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2023):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        dra_decfeb = yeartemp_decfeb
    else:
        dra_decfeb = np.hstack((dra_decfeb,yeartemp_decfeb))
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
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
    b_peak_temp = np.argmax(yeartemp_sepapr_pd_8day)
    chl_max_temp = np.nanmax(yeartemp_sepapr_pd_8day)
    b_dur_temp = b_term_temp - b_init_temp + 1
    b_magnitude_temp = integrate.simpson(yeartemp_sepapr_pd_8day.values[b_init_temp:b_term_temp+1])
    # Average for 8 days
    if i == 1998:
        dra_b_init = b_init_temp
        dra_b_term = b_term_temp
        dra_b_peak = b_peak_temp
        dra_chlmax = chl_max_temp
        dra_b_dur = b_dur_temp
        dra_b_magnitude = b_magnitude_temp
        
    else:
        dra_b_init = np.hstack((dra_b_init, b_init_temp))    
        dra_b_term = np.hstack((dra_b_term, b_term_temp))    
        dra_b_peak = np.hstack((dra_b_peak, b_peak_temp))    
        dra_chlmax = np.hstack((dra_chlmax, chl_max_temp))    
        dra_b_dur = np.hstack((dra_b_dur, b_dur_temp))    
        dra_b_magnitude = np.hstack((dra_b_magnitude, b_magnitude_temp))   
#%% DRA statistics for table
# Chl-a
np.nanmean(dra_decfeb)
np.nanmin(dra_decfeb)
np.nanmax(dra_decfeb)
np.nanstd(dra_decfeb)
np.nanpercentile(dra_decfeb,10)
np.nanpercentile(dra_decfeb,90)
# Bloom Peak
np.nanmean(dra_b_peak)
np.nanmin(dra_b_peak)
np.nanmax(dra_b_peak)
np.nanstd(dra_b_peak)
# Bloom Init
np.nanmean(dra_b_init)
np.nanmin(dra_b_init)
np.nanmax(dra_b_init)
np.nanstd(dra_b_init)
# Bloom Term
np.nanmean(dra_b_term)
np.nanmin(dra_b_term)
np.nanmax(dra_b_term)
np.nanstd(dra_b_term)
yeartemp_sepapr_pd_8day.index[26]
# Bloom Dur
np.nanmean(dra_b_dur)
np.nanmin(dra_b_dur)
np.nanmax(dra_b_dur)
np.nanstd(dra_b_dur)
np.nanpercentile(dra_b_dur,10)
np.nanpercentile(dra_b_dur,90)
# Bloom Magnitude
np.nanmean(dra_b_magnitude)
np.nanmin(dra_b_magnitude)
np.nanmax(dra_b_magnitude)
np.nanstd(dra_b_magnitude)
np.nanpercentile(dra_b_magnitude,10)
np.nanpercentile(dra_b_magnitude,90)
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
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2023):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        brs_decfeb = yeartemp_decfeb
    else:
        brs_decfeb = np.hstack((brs_decfeb,yeartemp_decfeb))
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
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
    b_peak_temp = np.argmax(yeartemp_sepapr_pd_8day)
    chl_max_temp = np.nanmax(yeartemp_sepapr_pd_8day)
    b_dur_temp = b_term_temp - b_init_temp + 1
    b_magnitude_temp = integrate.simpson(yeartemp_sepapr_pd_8day.values[b_init_temp:b_term_temp+1])
    # Average for 8 days
    if i == 1998:
        brs_b_init = b_init_temp
        brs_b_term = b_term_temp
        brs_b_peak = b_peak_temp
        brs_chlmax = chl_max_temp
        brs_b_dur = b_dur_temp
        brs_b_magnitude = b_magnitude_temp
        
    else:
        brs_b_init = np.hstack((brs_b_init, b_init_temp))    
        brs_b_term = np.hstack((brs_b_term, b_term_temp))    
        brs_b_peak = np.hstack((brs_b_peak, b_peak_temp))    
        brs_chlmax = np.hstack((brs_chlmax, chl_max_temp))    
        brs_b_dur = np.hstack((brs_b_dur, b_dur_temp))    
        brs_b_magnitude = np.hstack((brs_b_magnitude, b_magnitude_temp))
#%% BRS statistics for table
# Chl-a
np.nanmean(brs_decfeb)
np.nanmin(brs_decfeb)
np.nanmax(brs_decfeb)
np.nanstd(brs_decfeb)
np.nanpercentile(brs_decfeb,10)
np.nanpercentile(brs_decfeb,90)
# Bloom Peak
np.nanmean(brs_b_peak)
np.nanmin(brs_b_peak)
np.nanmax(brs_b_peak)
np.nanstd(brs_b_peak)
# Bloom Init
np.nanmean(brs_b_init)
np.nanmin(brs_b_init)
np.nanmax(brs_b_init)
np.nanstd(brs_b_init)
# Bloom Term
np.nanmean(brs_b_term)
np.nanmin(brs_b_term)
np.nanmax(brs_b_term)
np.nanstd(brs_b_term)
yeartemp_sepapr_pd_8day.index[26]
# Bloom Dur
np.nanmean(brs_b_dur)
np.nanmin(brs_b_dur)
np.nanmax(brs_b_dur)
np.nanstd(brs_b_dur)
np.nanpercentile(brs_b_dur,10)
np.nanpercentile(brs_b_dur,90)
# Bloom Magnitude
np.nanmean(brs_b_magnitude)
np.nanmin(brs_b_magnitude)
np.nanmax(brs_b_magnitude)
np.nanstd(brs_b_magnitude)
np.nanpercentile(brs_b_magnitude,10)
np.nanpercentile(brs_b_magnitude,90)
#%% Separar para o cluster 5 (WEDN)
wedn_cluster = chl[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
np.nanmedian(wedn_cluster)
np.nanmax(wedn_cluster)
np.nanmin(wedn_cluster)
np.nanstd(wedn_cluster)*3
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2023):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        wedn_decfeb = yeartemp_decfeb
    else:
        wedn_decfeb = np.hstack((wedn_decfeb,yeartemp_decfeb))
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
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
    b_peak_temp = np.argmax(yeartemp_sepapr_pd_8day)
    chl_max_temp = np.nanmax(yeartemp_sepapr_pd_8day)
    b_dur_temp = b_term_temp - b_init_temp + 1
    b_magnitude_temp = integrate.simpson(yeartemp_sepapr_pd_8day.values[b_init_temp:b_term_temp+1])
    # Average for 8 days
    if i == 1998:
        wedn_b_init = b_init_temp
        wedn_b_term = b_term_temp
        wedn_b_peak = b_peak_temp
        wedn_chlmax = chl_max_temp
        wedn_b_dur = b_dur_temp
        wedn_b_magnitude = b_magnitude_temp
        
    else:
        wedn_b_init = np.hstack((wedn_b_init, b_init_temp))    
        wedn_b_term = np.hstack((wedn_b_term, b_term_temp))    
        wedn_b_peak = np.hstack((wedn_b_peak, b_peak_temp))    
        wedn_chlmax = np.hstack((wedn_chlmax, chl_max_temp))    
        wedn_b_dur = np.hstack((wedn_b_dur, b_dur_temp))    
        wedn_b_magnitude = np.hstack((wedn_b_magnitude, b_magnitude_temp))
#%% WEDN statistics for table
# Chl-a
np.nanmean(wedn_decfeb)
np.nanmin(wedn_decfeb)
np.nanmax(wedn_decfeb)
np.nanstd(wedn_decfeb)
np.nanpercentile(wedn_decfeb,10)
np.nanpercentile(wedn_decfeb,90)
# Bloom Peak
np.nanmean(wedn_b_peak)
np.nanmin(wedn_b_peak)
np.nanmax(wedn_b_peak)
np.nanstd(wedn_b_peak)
# Bloom Init
np.nanmean(wedn_b_init)
np.nanmin(wedn_b_init)
np.nanmax(wedn_b_init)
np.nanstd(wedn_b_init)
# Bloom Term
np.nanmean(wedn_b_term)
np.nanmin(wedn_b_term)
np.nanmax(wedn_b_term)
np.nanstd(wedn_b_term)
yeartemp_sepapr_pd_8day.index[26]
# Bloom Dur
np.nanmean(wedn_b_dur)
np.nanmin(wedn_b_dur)
np.nanmax(wedn_b_dur)
np.nanstd(wedn_b_dur)
np.nanpercentile(wedn_b_dur,10)
np.nanpercentile(wedn_b_dur,90)
# Bloom Magnitude
np.nanmean(wedn_b_magnitude)
np.nanmin(wedn_b_magnitude)
np.nanmax(wedn_b_magnitude)
np.nanstd(wedn_b_magnitude)
np.nanpercentile(wedn_b_magnitude,10)
np.nanpercentile(wedn_b_magnitude,90)
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
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2023):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        ges_decfeb = yeartemp_decfeb
    else:
        ges_decfeb = np.hstack((ges_decfeb,yeartemp_decfeb))
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
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
    b_peak_temp = np.argmax(yeartemp_sepapr_pd_8day)
    chl_max_temp = np.nanmax(yeartemp_sepapr_pd_8day)
    b_dur_temp = b_term_temp - b_init_temp + 1
    b_magnitude_temp = integrate.simpson(yeartemp_sepapr_pd_8day.values[b_init_temp:b_term_temp+1])
    # Average for 8 days
    if i == 1998:
        ges_b_init = b_init_temp
        ges_b_term = b_term_temp
        ges_b_peak = b_peak_temp
        ges_chlmax = chl_max_temp
        ges_b_dur = b_dur_temp
        ges_b_magnitude = b_magnitude_temp
        
    else:
        ges_b_init = np.hstack((ges_b_init, b_init_temp))    
        ges_b_term = np.hstack((ges_b_term, b_term_temp))    
        ges_b_peak = np.hstack((ges_b_peak, b_peak_temp))    
        ges_chlmax = np.hstack((ges_chlmax, chl_max_temp))    
        ges_b_dur = np.hstack((ges_b_dur, b_dur_temp))    
        ges_b_magnitude = np.hstack((ges_b_magnitude, b_magnitude_temp))
#%% GES statistics for table
# Chl-a
np.nanmean(ges_decfeb)
np.nanmin(ges_decfeb)
np.nanmax(ges_decfeb)
np.nanstd(ges_decfeb)
np.nanpercentile(ges_decfeb,10)
np.nanpercentile(ges_decfeb,90)
# Bloom Peak
np.nanmean(ges_b_peak)
np.nanmin(ges_b_peak)
np.nanmax(ges_b_peak)
np.nanstd(ges_b_peak)
# Bloom Init
np.nanmean(ges_b_init)
np.nanmin(ges_b_init)
np.nanmax(ges_b_init)
np.nanstd(ges_b_init)
# Bloom Term
np.nanmean(ges_b_term)
np.nanmin(ges_b_term)
np.nanmax(ges_b_term)
np.nanstd(ges_b_term)
yeartemp_sepapr_pd_8day.index[26]
# Bloom Dur
np.nanmean(ges_b_dur)
np.nanmin(ges_b_dur)
np.nanmax(ges_b_dur)
np.nanstd(ges_b_dur)
np.nanpercentile(ges_b_dur,10)
np.nanpercentile(ges_b_dur,90)
# Bloom Magnitude
np.nanmean(ges_b_magnitude)
np.nanmin(ges_b_magnitude)
np.nanmax(ges_b_magnitude)
np.nanstd(ges_b_magnitude)
np.nanpercentile(ges_b_magnitude,10)
np.nanpercentile(ges_b_magnitude,90)
#%% Separar para o cluster 1 (WEDS)
weds_cluster = chl[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
np.nanmedian(weds_cluster)
np.nanmax(weds_cluster)
np.nanmin(weds_cluster)
np.nanstd(weds_cluster)*3
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2023):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        weds_decfeb = yeartemp_decfeb
    else:
        weds_decfeb = np.hstack((weds_decfeb,yeartemp_decfeb))
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
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
    b_peak_temp = np.argmax(yeartemp_sepapr_pd_8day)
    chl_max_temp = np.nanmax(yeartemp_sepapr_pd_8day)
    b_dur_temp = b_term_temp - b_init_temp + 1
    b_magnitude_temp = integrate.simpson(yeartemp_sepapr_pd_8day.values[b_init_temp:b_term_temp+1])
    # Average for 8 days
    if i == 1998:
        weds_b_init = b_init_temp
        weds_b_term = b_term_temp
        weds_b_peak = b_peak_temp
        weds_chlmax = chl_max_temp
        weds_b_dur = b_dur_temp
        weds_b_magnitude = b_magnitude_temp
        
    else:
        weds_b_init = np.hstack((weds_b_init, b_init_temp))    
        weds_b_term = np.hstack((weds_b_term, b_term_temp))    
        weds_b_peak = np.hstack((weds_b_peak, b_peak_temp))    
        weds_chlmax = np.hstack((weds_chlmax, chl_max_temp))    
        weds_b_dur = np.hstack((weds_b_dur, b_dur_temp))    
        weds_b_magnitude = np.hstack((weds_b_magnitude, b_magnitude_temp))
#%% WEDS statistics for table
# Chl-a
np.nanmean(weds_decfeb)
np.nanmin(weds_decfeb)
np.nanmax(weds_decfeb)
np.nanstd(weds_decfeb)
np.nanpercentile(weds_decfeb,10)
np.nanpercentile(weds_decfeb,90)
# Bloom Peak
np.nanmean(weds_b_peak)
np.nanmin(weds_b_peak)
np.nanmax(weds_b_peak)
np.nanstd(weds_b_peak)
# Bloom Init
np.nanmean(weds_b_init)
np.nanmin(weds_b_init)
np.nanmax(weds_b_init)
np.nanstd(weds_b_init)
# Bloom Term
np.nanmean(weds_b_term)
np.nanmin(weds_b_term)
np.nanmax(weds_b_term)
np.nanstd(weds_b_term)
yeartemp_sepapr_pd_8day.index[26]
# Bloom Dur
np.nanmean(weds_b_dur)
np.nanmin(weds_b_dur)
np.nanmax(weds_b_dur)
np.nanstd(weds_b_dur)
np.nanpercentile(weds_b_dur,10)
np.nanpercentile(weds_b_dur,90)
# Bloom Magnitude
np.nanmean(weds_b_magnitude)
np.nanmin(weds_b_magnitude)
np.nanmax(weds_b_magnitude)
np.nanstd(weds_b_magnitude)
np.nanpercentile(weds_b_magnitude,10)
np.nanpercentile(weds_b_magnitude,90)








