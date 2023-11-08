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
#%% Load data 1998-2022
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
### Load data 1998-2020
fh = np.load('sst-seaice_19972021.npz', allow_pickle=True)
lat_sst = fh['lat']
lon_sst = fh['lon']
sst = fh['sst']
seaice = fh['seaice']
seaice = seaice * 100
time_date_sst = fh['time_date']
time_date_years_sst = np.empty_like(time_date_sst)
time_date_months_sst = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years_sst[i] = time_date_sst[i].year
    time_date_months_sst[i] = time_date_sst[i].month
# Load upscaled 4km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_sst.npz',allow_pickle = True)
clusters_sst = fh['clusters'] 
#%% Load data 1998-2022
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\par\\')
### Load data 1998-2020
fh = np.load('par_19972021_new.npz', allow_pickle=True)
lat_par = fh['lat']
lon_par = fh['lon']
par = fh['par']
par = par.astype(dtype=np.float64)
time_date_par = fh['time_date']
time_date_years_par = np.empty_like(time_date_par)
time_date_months_par = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years_par[i] = time_date_par[i].year
    time_date_months_par[i] = time_date_par[i].month
# Load upscaled 4km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_par.npz',allow_pickle = True)
clusters_par = fh['clusters']
#%% Load data 1998-2022
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\winds\\')
### Load data 1998-2020
fh = np.load('winds_19972022_era5.npz', allow_pickle=True)
lat_winds = fh['lat']
lon_winds = fh['lon']
winds_u = fh['wind_u']
winds_v = fh['wind_v']
windspeed = np.sqrt(winds_u**2 + winds_v**2)
time_date_winds = fh['time_date']
time_date_years_winds = np.empty_like(time_date_winds)
time_date_months_winds = np.empty_like(time_date_winds)
for i in range(0, len(time_date_winds)):
    time_date_years_winds[i] = time_date_winds[i].year
    time_date_months_winds[i] = time_date_winds[i].month
# Load upscaled 4km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_winds.npz',allow_pickle = True)
clusters_winds = fh['clusters']
del(winds_u, winds_v)
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
#%% Load El Niño
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\elnino\\')
elnino_pd = pd.read_csv('meiv2.csv', sep=';', header=None)
for i in range(0, len(elnino_pd)):
    temp_elnino = np.hstack((elnino_pd[1][i], elnino_pd[2][i], elnino_pd[3][i], elnino_pd[4][i],
                             elnino_pd[5][i], elnino_pd[6][i], elnino_pd[7][i], elnino_pd[8][i],
                             elnino_pd[9][i], elnino_pd[10][i], elnino_pd[11][i], elnino_pd[12][i]))
    temp_elnino = temp_elnino.astype(np.float)
    # Join
    if i == 0:
        meiv2 = temp_elnino
        meiv2_months = np.arange(1,13)
        meiv2_years = np.repeat(elnino_pd[0][i], 12)
    else:
        meiv2 = np.hstack((meiv2, temp_elnino))
        meiv2_months = np.hstack((meiv2_months, np.arange(1,13)))
        meiv2_years = np.hstack((meiv2_years, np.repeat(elnino_pd[0][i], 12)))
#%% For Spring (Sep-Nov), summer (DJF), autumn (MA), calculate average and separate positive and negative
# DRA
# Spring SAM
for i in np.arange(1998, 2023):
    yeartemp_sep = sam_daily[(time_date_years_sam == i-1) & (time_date_months_sam == 9)]
    yeartemp_oct = sam_daily[(time_date_years_sam == i-1) & (time_date_months_sam == 10)]
    yeartemp_nov = sam_daily[(time_date_years_sam == i-1) & (time_date_months_sam == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        sam_SON = np.nanmean(yeartemp_SON)
    else:
        sam_SON = np.hstack((sam_SON, np.nanmean(yeartemp_SON)))
# Spring MEI
for i in np.arange(1998, 2023):
    yeartemp_sep = meiv2[(meiv2_years == i-1) & (meiv2_months == 9)]
    yeartemp_oct = meiv2[(meiv2_years == i-1) & (meiv2_months == 10)]
    yeartemp_nov = meiv2[(meiv2_years == i-1) & (meiv2_months == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        mei_SON = np.nanmean(yeartemp_SON)
    else:
        mei_SON = np.hstack((mei_SON, np.nanmean(yeartemp_SON)))
# DRA Spring Chl-a
dra_cluster = chl[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        dra_chl_SON = np.nanmean(yeartemp_SON)
    else:
        dra_chl_SON = np.hstack((dra_chl_SON, np.nanmean(yeartemp_SON)))
# DRA Spring SST
dra_cluster = sst[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        dra_sst_SON = np.nanmean(yeartemp_SON)
    else:
        dra_sst_SON = np.hstack((dra_sst_SON, np.nanmean(yeartemp_SON)))
# DRA Spring Sea Ice
dra_cluster = seaice[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        dra_seaice_SON = np.nanmean(yeartemp_SON)
    else:
        dra_seaice_SON = np.hstack((dra_seaice_SON, np.nanmean(yeartemp_SON)))
# DRA Spring PAR
dra_cluster = par[clusters_par == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = dra_cluster[(time_date_years_par == i-1) & (time_date_months_par == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_par == i-1) & (time_date_months_par == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_par == i-1) & (time_date_months_par == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        dra_par_SON = np.nanmean(yeartemp_SON)
    else:
        dra_par_SON = np.hstack((dra_par_SON, np.nanmean(yeartemp_SON)))
# DRA Spring Winds
dra_cluster = windspeed[clusters_winds == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        dra_windspeed_SON = np.nanmean(yeartemp_SON)
    else:
        dra_windspeed_SON = np.hstack((dra_windspeed_SON, np.nanmean(yeartemp_SON)))
#%% Calculate comparisons
## SAM
# Chl-a
dra_chl_SON_SAM_POS = dra_chl_SON[sam_SON > 0.5]
dra_chl_SON_SAM_NEG = dra_chl_SON[sam_SON < -0.5]
print(np.nanmean(dra_chl_SON_SAM_POS))
print(np.nanmean(dra_chl_SON_SAM_NEG))
stats.mannwhitneyu(dra_chl_SON_SAM_POS, dra_chl_SON_SAM_NEG, nan_policy = 'omit') # ***
# SST
dra_sst_SON_SAM_POS = dra_sst_SON[sam_SON > 0.5]
dra_sst_SON_SAM_NEG = dra_sst_SON[sam_SON < -0.5]
print(np.nanmean(dra_sst_SON_SAM_POS))
print(np.nanmean(dra_sst_SON_SAM_NEG))
stats.mannwhitneyu(dra_sst_SON_SAM_POS, dra_sst_SON_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
dra_seaice_SON_SAM_POS = dra_seaice_SON[sam_SON > 0.5]
dra_seaice_SON_SAM_NEG = dra_seaice_SON[sam_SON < -0.5]
print(np.nanmean(dra_seaice_SON_SAM_POS))
print(np.nanmean(dra_seaice_SON_SAM_NEG))
stats.mannwhitneyu(dra_seaice_SON_SAM_POS, dra_seaice_SON_SAM_NEG, nan_policy = 'omit') #
# PAR
dra_par_SON_SAM_POS = dra_par_SON[sam_SON > 0.5]
dra_par_SON_SAM_NEG = dra_par_SON[sam_SON < -0.5]
print(np.nanmean(dra_par_SON_SAM_POS))
print(np.nanmean(dra_par_SON_SAM_NEG))
stats.mannwhitneyu(dra_par_SON_SAM_POS, dra_par_SON_SAM_NEG, nan_policy = 'omit') # **
# Winds
dra_windspeed_SON_SAM_POS = dra_windspeed_SON[sam_SON > 0.5]
dra_windspeed_SON_SAM_NEG = dra_windspeed_SON[sam_SON < -0.5]
print(np.nanmean(dra_windspeed_SON_SAM_POS))
print(np.nanmean(dra_windspeed_SON_SAM_NEG))
stats.mannwhitneyu(dra_windspeed_SON_SAM_POS, dra_windspeed_SON_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
dra_chl_SON_MEI_POS = dra_chl_SON[mei_SON > 0.5]
dra_chl_SON_MEI_NEG = dra_chl_SON[mei_SON < -0.5]
print(np.nanmean(dra_chl_SON_MEI_POS))
print(np.nanmean(dra_chl_SON_MEI_NEG))
stats.mannwhitneyu(dra_chl_SON_MEI_POS, dra_chl_SON_MEI_NEG, nan_policy = 'omit') # ***
# SST
dra_sst_SON_MEI_POS = dra_sst_SON[mei_SON > 0.5]
dra_sst_SON_MEI_NEG = dra_sst_SON[mei_SON < -0.5]
print(np.nanmean(dra_sst_SON_MEI_POS))
print(np.nanmean(dra_sst_SON_MEI_NEG))
stats.mannwhitneyu(dra_sst_SON_MEI_POS, dra_sst_SON_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
dra_seaice_SON_MEI_POS = dra_seaice_SON[mei_SON > 0.5]
dra_seaice_SON_MEI_NEG = dra_seaice_SON[mei_SON < -0.5]
print(np.nanmean(dra_seaice_SON_MEI_POS))
print(np.nanmean(dra_seaice_SON_MEI_NEG))
stats.mannwhitneyu(dra_seaice_SON_MEI_POS, dra_seaice_SON_MEI_NEG, nan_policy = 'omit') #
# PAR
dra_par_SON_MEI_POS = dra_par_SON[mei_SON > 0.5]
dra_par_SON_MEI_NEG = dra_par_SON[mei_SON < -0.5]
print(np.nanmean(dra_par_SON_MEI_POS))
print(np.nanmean(dra_par_SON_MEI_NEG))
stats.mannwhitneyu(dra_par_SON_MEI_POS, dra_par_SON_MEI_NEG, nan_policy = 'omit') # **
# Winds
dra_windspeed_SON_MEI_POS = dra_windspeed_SON[mei_SON > 0.5]
dra_windspeed_SON_MEI_NEG = dra_windspeed_SON[mei_SON < -0.5]
print(np.nanmean(dra_windspeed_SON_MEI_POS))
print(np.nanmean(dra_windspeed_SON_MEI_NEG))
stats.mannwhitneyu(dra_windspeed_SON_MEI_POS, dra_windspeed_SON_MEI_NEG, nan_policy = 'omit') # **
#%% # DRA
# Summer SAM
for i in np.arange(1998, 2023):
    yeartemp_dec = sam_daily[(time_date_years_sam == i-1) & (time_date_months_sam == 12)]
    yeartemp_jan = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 1)]
    yeartemp_feb = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        sam_DJF = np.nanmean(yeartemp_DJF)
    else:
        sam_DJF = np.hstack((sam_DJF, np.nanmean(yeartemp_DJF)))
# Summer MEI
for i in np.arange(1998, 2023):
    yeartemp_dec = meiv2[(meiv2_years == i-1) & (meiv2_months == 12)]
    yeartemp_jan = meiv2[(meiv2_years == i) & (meiv2_months == 1)]
    yeartemp_feb = meiv2[(meiv2_years == i) & (meiv2_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        mei_DJF = np.nanmean(yeartemp_DJF)
    else:
        mei_DJF = np.hstack((mei_DJF, np.nanmean(yeartemp_DJF)))
# DRA Summer Chl-a
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
# DRA Summer SST
dra_cluster = sst[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        dra_sst_DJF = np.nanmean(yeartemp_DJF)
    else:
        dra_sst_DJF = np.hstack((dra_sst_DJF, np.nanmean(yeartemp_DJF)))
# DRA Summer Sea Ice
dra_cluster = seaice[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        dra_seaice_DJF = np.nanmean(yeartemp_DJF)
    else:
        dra_seaice_DJF = np.hstack((dra_seaice_DJF, np.nanmean(yeartemp_DJF)))
# DRA Summer PAR
dra_cluster = par[clusters_par == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = dra_cluster[(time_date_years_par == i-1) & (time_date_months_par == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_par == i) & (time_date_months_par == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_par == i) & (time_date_months_par == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        dra_par_DJF = np.nanmean(yeartemp_DJF)
    else:
        dra_par_DJF = np.hstack((dra_par_DJF, np.nanmean(yeartemp_DJF)))
# DRA Summer Winds
dra_cluster = windspeed[clusters_winds == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        dra_windspeed_DJF = np.nanmean(yeartemp_DJF)
    else:
        dra_windspeed_DJF = np.hstack((dra_windspeed_DJF, np.nanmean(yeartemp_DJF)))
#%% Calculate comparisons
## SAM
# Chl-a
dra_chl_DJF_SAM_POS = dra_chl_DJF[sam_DJF > 0.5]
dra_chl_DJF_SAM_NEG = dra_chl_DJF[sam_DJF < -0.5]
print(np.nanmean(dra_chl_DJF_SAM_POS))
print(np.nanmean(dra_chl_DJF_SAM_NEG))
stats.mannwhitneyu(dra_chl_DJF_SAM_POS, dra_chl_DJF_SAM_NEG, nan_policy = 'omit') # ***
# SST
dra_sst_DJF_SAM_POS = dra_sst_DJF[sam_DJF > 0.5]
dra_sst_DJF_SAM_NEG = dra_sst_DJF[sam_DJF < -0.5]
print(np.nanmean(dra_sst_DJF_SAM_POS))
print(np.nanmean(dra_sst_DJF_SAM_NEG))
stats.mannwhitneyu(dra_sst_DJF_SAM_POS, dra_sst_DJF_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
dra_seaice_DJF_SAM_POS = dra_seaice_DJF[sam_DJF > 0.5]
dra_seaice_DJF_SAM_NEG = dra_seaice_DJF[sam_DJF < -0.5]
print(np.nanmean(dra_seaice_DJF_SAM_POS))
print(np.nanmean(dra_seaice_DJF_SAM_NEG))
stats.mannwhitneyu(dra_seaice_DJF_SAM_POS, dra_seaice_DJF_SAM_NEG, nan_policy = 'omit') #
# PAR
dra_par_DJF_SAM_POS = dra_par_DJF[sam_DJF > 0.5]
dra_par_DJF_SAM_NEG = dra_par_DJF[sam_DJF < -0.5]
print(np.nanmean(dra_par_DJF_SAM_POS))
print(np.nanmean(dra_par_DJF_SAM_NEG))
stats.mannwhitneyu(dra_par_DJF_SAM_POS, dra_par_DJF_SAM_NEG, nan_policy = 'omit') # **
# Winds
dra_windspeed_DJF_SAM_POS = dra_windspeed_DJF[sam_DJF > 0.5]
dra_windspeed_DJF_SAM_NEG = dra_windspeed_DJF[sam_DJF < -0.5]
print(np.nanmean(dra_windspeed_DJF_SAM_POS))
print(np.nanmean(dra_windspeed_DJF_SAM_NEG))
stats.mannwhitneyu(dra_windspeed_DJF_SAM_POS, dra_windspeed_DJF_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
dra_chl_DJF_MEI_POS = dra_chl_DJF[mei_DJF > 0.5]
dra_chl_DJF_MEI_NEG = dra_chl_DJF[mei_DJF < -0.5]
print(np.nanmean(dra_chl_DJF_MEI_POS))
print(np.nanmean(dra_chl_DJF_MEI_NEG))
stats.mannwhitneyu(dra_chl_DJF_MEI_POS, dra_chl_DJF_MEI_NEG, nan_policy = 'omit') # ***
# SST
dra_sst_DJF_MEI_POS = dra_sst_DJF[mei_DJF > 0.5]
dra_sst_DJF_MEI_NEG = dra_sst_DJF[mei_DJF < -0.5]
print(np.nanmean(dra_sst_DJF_MEI_POS))
print(np.nanmean(dra_sst_DJF_MEI_NEG))
stats.mannwhitneyu(dra_sst_DJF_MEI_POS, dra_sst_DJF_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
dra_seaice_DJF_MEI_POS = dra_seaice_DJF[mei_DJF > 0.5]
dra_seaice_DJF_MEI_NEG = dra_seaice_DJF[mei_DJF < -0.5]
print(np.nanmean(dra_seaice_DJF_MEI_POS))
print(np.nanmean(dra_seaice_DJF_MEI_NEG))
stats.mannwhitneyu(dra_seaice_DJF_MEI_POS, dra_seaice_DJF_MEI_NEG, nan_policy = 'omit') #
# PAR
dra_par_DJF_MEI_POS = dra_par_DJF[mei_DJF > 0.5]
dra_par_DJF_MEI_NEG = dra_par_DJF[mei_DJF < -0.5]
print(np.nanmean(dra_par_DJF_MEI_POS))
print(np.nanmean(dra_par_DJF_MEI_NEG))
stats.mannwhitneyu(dra_par_DJF_MEI_POS, dra_par_DJF_MEI_NEG, nan_policy = 'omit') # **
# Winds
dra_windspeed_DJF_MEI_POS = dra_windspeed_DJF[mei_DJF > 0.5]
dra_windspeed_DJF_MEI_NEG = dra_windspeed_DJF[mei_DJF < -0.5]
print(np.nanmean(dra_windspeed_DJF_MEI_POS))
print(np.nanmean(dra_windspeed_DJF_MEI_NEG))
stats.mannwhitneyu(dra_windspeed_DJF_MEI_POS, dra_windspeed_DJF_MEI_NEG, nan_policy = 'omit') # **
#%% # DRA
# Autumn SAM
for i in np.arange(1998, 2023):
    yeartemp_mar = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 3)]
    yeartemp_apr = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        sam_MA = np.nanmean(yeartemp_MA)
    else:
        sam_MA = np.hstack((sam_MA, np.nanmean(yeartemp_MA)))
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_mar = meiv2[(meiv2_years == i) & (meiv2_months == 3)]
    yeartemp_apr = meiv2[(meiv2_years == i) & (meiv2_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        mei_MA = np.nanmean(yeartemp_MA)
    else:
        mei_MA = np.hstack((mei_MA, np.nanmean(yeartemp_MA)))
# DRA Autumn Chl-a
dra_cluster = chl[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
for i in np.arange(1998, 2023):
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        dra_chl_MA = np.nanmean(yeartemp_MA)
    else:
        dra_chl_MA = np.hstack((dra_chl_MA, np.nanmean(yeartemp_MA)))
# DRA Autumn SST
dra_cluster = sst[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        dra_sst_MA = np.nanmean(yeartemp_MA)
    else:
        dra_sst_MA = np.hstack((dra_sst_MA, np.nanmean(yeartemp_MA)))
# DRA Autumn Sea Ice
dra_cluster = seaice[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        dra_seaice_MA = np.nanmean(yeartemp_MA)
    else:
        dra_seaice_MA = np.hstack((dra_seaice_MA, np.nanmean(yeartemp_MA)))
# DRA Autumn PAR
dra_cluster = par[clusters_par == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = dra_cluster[(time_date_years_par == i) & (time_date_months_par == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_par == i) & (time_date_months_par == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        dra_par_MA = np.nanmean(yeartemp_MA)
    else:
        dra_par_MA = np.hstack((dra_par_MA, np.nanmean(yeartemp_MA)))
# DRA Autumn Winds
dra_cluster = windspeed[clusters_winds == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        dra_windspeed_MA = np.nanmean(yeartemp_MA)
    else:
        dra_windspeed_MA = np.hstack((dra_windspeed_MA, np.nanmean(yeartemp_MA)))
#%% Calculate comparisons
## SAM
# Chl-a
dra_chl_MA_SAM_POS = dra_chl_MA[sam_MA > 0.5]
dra_chl_MA_SAM_NEG = dra_chl_MA[sam_MA < -0.5]
print(np.nanmean(dra_chl_MA_SAM_POS))
print(np.nanmean(dra_chl_MA_SAM_NEG))
stats.mannwhitneyu(dra_chl_MA_SAM_POS, dra_chl_MA_SAM_NEG, nan_policy = 'omit') # ***
# SST
dra_sst_MA_SAM_POS = dra_sst_MA[sam_MA > 0.5]
dra_sst_MA_SAM_NEG = dra_sst_MA[sam_MA < -0.5]
print(np.nanmean(dra_sst_MA_SAM_POS))
print(np.nanmean(dra_sst_MA_SAM_NEG))
stats.mannwhitneyu(dra_sst_MA_SAM_POS, dra_sst_MA_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
dra_seaice_MA_SAM_POS = dra_seaice_MA[sam_MA > 0.5]
dra_seaice_MA_SAM_NEG = dra_seaice_MA[sam_MA < -0.5]
print(np.nanmean(dra_seaice_MA_SAM_POS))
print(np.nanmean(dra_seaice_MA_SAM_NEG))
stats.mannwhitneyu(dra_seaice_MA_SAM_POS, dra_seaice_MA_SAM_NEG, nan_policy = 'omit') #
# PAR
dra_par_MA_SAM_POS = dra_par_MA[sam_MA > 0.5]
dra_par_MA_SAM_NEG = dra_par_MA[sam_MA < -0.5]
print(np.nanmean(dra_par_MA_SAM_POS))
print(np.nanmean(dra_par_MA_SAM_NEG))
stats.mannwhitneyu(dra_par_MA_SAM_POS, dra_par_MA_SAM_NEG, nan_policy = 'omit') # **
# Winds
dra_windspeed_MA_SAM_POS = dra_windspeed_MA[sam_MA > 0.5]
dra_windspeed_MA_SAM_NEG = dra_windspeed_MA[sam_MA < -0.5]
print(np.nanmean(dra_windspeed_MA_SAM_POS))
print(np.nanmean(dra_windspeed_MA_SAM_NEG))
stats.mannwhitneyu(dra_windspeed_MA_SAM_POS, dra_windspeed_MA_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
dra_chl_MA_MEI_POS = dra_chl_MA[mei_MA > 0.5]
dra_chl_MA_MEI_NEG = dra_chl_MA[mei_MA < -0.5]
print(np.nanmean(dra_chl_MA_MEI_POS))
print(np.nanmean(dra_chl_MA_MEI_NEG))
stats.mannwhitneyu(dra_chl_MA_MEI_POS, dra_chl_MA_MEI_NEG, nan_policy = 'omit') # ***
# SST
dra_sst_MA_MEI_POS = dra_sst_MA[mei_MA > 0.5]
dra_sst_MA_MEI_NEG = dra_sst_MA[mei_MA < -0.5]
print(np.nanmean(dra_sst_MA_MEI_POS))
print(np.nanmean(dra_sst_MA_MEI_NEG))
stats.mannwhitneyu(dra_sst_MA_MEI_POS, dra_sst_MA_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
dra_seaice_MA_MEI_POS = dra_seaice_MA[mei_MA > 0.5]
dra_seaice_MA_MEI_NEG = dra_seaice_MA[mei_MA < -0.5]
print(np.nanmean(dra_seaice_MA_MEI_POS))
print(np.nanmean(dra_seaice_MA_MEI_NEG))
stats.mannwhitneyu(dra_seaice_MA_MEI_POS, dra_seaice_MA_MEI_NEG, nan_policy = 'omit') #
# PAR
dra_par_MA_MEI_POS = dra_par_MA[mei_MA > 0.5]
dra_par_MA_MEI_NEG = dra_par_MA[mei_MA < -0.5]
print(np.nanmean(dra_par_MA_MEI_POS))
print(np.nanmean(dra_par_MA_MEI_NEG))
stats.mannwhitneyu(dra_par_MA_MEI_POS, dra_par_MA_MEI_NEG, nan_policy = 'omit') # **
# Winds
dra_windspeed_MA_MEI_POS = dra_windspeed_MA[mei_MA > 0.5]
dra_windspeed_MA_MEI_NEG = dra_windspeed_MA[mei_MA < -0.5]
print(np.nanmean(dra_windspeed_MA_MEI_POS))
print(np.nanmean(dra_windspeed_MA_MEI_NEG))
stats.mannwhitneyu(dra_windspeed_MA_MEI_POS, dra_windspeed_MA_MEI_NEG, nan_policy = 'omit') # **
#%% September - April!
# Autumn SAM
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
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_sepdec = meiv2[(meiv2_years == i-1) & ((meiv2_months == 9) | (meiv2_months == 10)
                                                                | (meiv2_months == 11) | (meiv2_months == 12))]
    yeartemp_janapr = meiv2[(meiv2_years == i) & ((meiv2_months == 1) | (meiv2_months == 2)
                                                                | (meiv2_months == 3) | (meiv2_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))

    if i == 1998:
        mei_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        mei_SEPAPR = np.hstack((mei_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# DRA Autumn Chl-a
dra_cluster = chl[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sepdec = dra_cluster[(time_date_years == i-1) & ((time_date_months == 9) | (time_date_months == 10)
                                                                | (time_date_months == 11) | (time_date_months == 12))]
    yeartemp_janapr = dra_cluster[(time_date_years == i) & ((time_date_months == 1) | (time_date_months == 2)
                                                                | (time_date_months == 3) | (time_date_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        dra_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        dra_chl_SEPAPR = np.hstack((dra_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# DRA Autumn SST
dra_cluster = sst[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = dra_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = dra_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        dra_sst_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        dra_sst_SEPAPR = np.hstack((dra_sst_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# DRA Autumn Sea Ice
dra_cluster = seaice[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = dra_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = dra_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        dra_seaice_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        dra_seaice_SEPAPR = np.hstack((dra_seaice_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# DRA Autumn PAR
dra_cluster = par[clusters_par == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = dra_cluster[(time_date_years_par == i-1) & ((time_date_months_par == 9) | (time_date_months_par == 10)
                                                                | (time_date_months_par == 11) | (time_date_months_par == 12))]
    yeartemp_janapr = dra_cluster[(time_date_years_par == i) & ((time_date_months_par == 1) | (time_date_months_par == 2)
                                                                | (time_date_months_par == 3) | (time_date_months_par == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        dra_par_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        dra_par_SEPAPR = np.hstack((dra_par_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# DRA Autumn Winds
dra_cluster = windspeed[clusters_winds == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = dra_cluster[(time_date_years_winds == i-1) & ((time_date_months_winds == 9) | (time_date_months_winds == 10)
                                                                | (time_date_months_winds == 11) | (time_date_months_winds == 12))]
    yeartemp_janapr = dra_cluster[(time_date_years_winds == i) & ((time_date_months_winds == 1) | (time_date_months_winds == 2)
                                                                | (time_date_months_winds == 3) | (time_date_months_winds == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        dra_windspeed_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        dra_windspeed_SEPAPR = np.hstack((dra_windspeed_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate comparisons
## SAM
# Chl-a
dra_chl_SEPAPR_SAM_POS = dra_chl_SEPAPR[sam_SEPAPR > 0.5]
dra_chl_SEPAPR_SAM_NEG = dra_chl_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(dra_chl_SEPAPR_SAM_POS))
print(np.nanmean(dra_chl_SEPAPR_SAM_NEG))
stats.mannwhitneyu(dra_chl_SEPAPR_SAM_POS, dra_chl_SEPAPR_SAM_NEG, nan_policy = 'omit') # ***
# SST
dra_sst_SEPAPR_SAM_POS = dra_sst_SEPAPR[sam_SEPAPR > 0.5]
dra_sst_SEPAPR_SAM_NEG = dra_sst_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(dra_sst_SEPAPR_SAM_POS))
print(np.nanmean(dra_sst_SEPAPR_SAM_NEG))
stats.mannwhitneyu(dra_sst_SEPAPR_SAM_POS, dra_sst_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
dra_seaice_SEPAPR_SAM_POS = dra_seaice_SEPAPR[sam_SEPAPR > 0.5]
dra_seaice_SEPAPR_SAM_NEG = dra_seaice_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(dra_seaice_SEPAPR_SAM_POS))
print(np.nanmean(dra_seaice_SEPAPR_SAM_NEG))
stats.mannwhitneyu(dra_seaice_SEPAPR_SAM_POS, dra_seaice_SEPAPR_SAM_NEG, nan_policy = 'omit') #
# PAR
dra_par_SEPAPR_SAM_POS = dra_par_SEPAPR[sam_SEPAPR > 0.5]
dra_par_SEPAPR_SAM_NEG = dra_par_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(dra_par_SEPAPR_SAM_POS))
print(np.nanmean(dra_par_SEPAPR_SAM_NEG))
stats.mannwhitneyu(dra_par_SEPAPR_SAM_POS, dra_par_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Winds
dra_windspeed_SEPAPR_SAM_POS = dra_windspeed_SEPAPR[sam_SEPAPR > 0.5]
dra_windspeed_SEPAPR_SAM_NEG = dra_windspeed_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(dra_windspeed_SEPAPR_SAM_POS))
print(np.nanmean(dra_windspeed_SEPAPR_SAM_NEG))
stats.mannwhitneyu(dra_windspeed_SEPAPR_SAM_POS, dra_windspeed_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
dra_chl_SEPAPR_MEI_POS = dra_chl_SEPAPR[mei_SEPAPR > 0.5]
dra_chl_SEPAPR_MEI_NEG = dra_chl_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(dra_chl_SEPAPR_MEI_POS))
print(np.nanmean(dra_chl_SEPAPR_MEI_NEG))
stats.mannwhitneyu(dra_chl_SEPAPR_MEI_POS, dra_chl_SEPAPR_MEI_NEG, nan_policy = 'omit') # ***
# SST
dra_sst_SEPAPR_MEI_POS = dra_sst_SEPAPR[mei_SEPAPR > 0.5]
dra_sst_SEPAPR_MEI_NEG = dra_sst_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(dra_sst_SEPAPR_MEI_POS))
print(np.nanmean(dra_sst_SEPAPR_MEI_NEG))
stats.mannwhitneyu(dra_sst_SEPAPR_MEI_POS, dra_sst_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
dra_seaice_SEPAPR_MEI_POS = dra_seaice_SEPAPR[mei_SEPAPR > 0.5]
dra_seaice_SEPAPR_MEI_NEG = dra_seaice_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(dra_seaice_SEPAPR_MEI_POS))
print(np.nanmean(dra_seaice_SEPAPR_MEI_NEG))
stats.mannwhitneyu(dra_seaice_SEPAPR_MEI_POS, dra_seaice_SEPAPR_MEI_NEG, nan_policy = 'omit') #
# PAR
dra_par_SEPAPR_MEI_POS = dra_par_SEPAPR[mei_SEPAPR > 0.5]
dra_par_SEPAPR_MEI_NEG = dra_par_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(dra_par_SEPAPR_MEI_POS))
print(np.nanmean(dra_par_SEPAPR_MEI_NEG))
stats.mannwhitneyu(dra_par_SEPAPR_MEI_POS, dra_par_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Winds
dra_windspeed_SEPAPR_MEI_POS = dra_windspeed_SEPAPR[mei_SEPAPR > 0.5]
dra_windspeed_SEPAPR_MEI_NEG = dra_windspeed_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(dra_windspeed_SEPAPR_MEI_POS))
print(np.nanmean(dra_windspeed_SEPAPR_MEI_NEG))
stats.mannwhitneyu(dra_windspeed_SEPAPR_MEI_POS, dra_windspeed_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
#%% BRS
# Spring SAM
# BRS Spring Chl-a
brs_cluster = chl[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        brs_chl_SON = np.nanmean(yeartemp_SON)
    else:
        brs_chl_SON = np.hstack((brs_chl_SON, np.nanmean(yeartemp_SON)))
# BRS Spring SST
brs_cluster = sst[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        brs_sst_SON = np.nanmean(yeartemp_SON)
    else:
        brs_sst_SON = np.hstack((brs_sst_SON, np.nanmean(yeartemp_SON)))
# BRS Spring Sea Ice
brs_cluster = seaice[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        brs_seaice_SON = np.nanmean(yeartemp_SON)
    else:
        brs_seaice_SON = np.hstack((brs_seaice_SON, np.nanmean(yeartemp_SON)))
# BRS Spring PAR
brs_cluster = par[clusters_par == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = brs_cluster[(time_date_years_par == i-1) & (time_date_months_par == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_par == i-1) & (time_date_months_par == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_par == i-1) & (time_date_months_par == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        brs_par_SON = np.nanmean(yeartemp_SON)
    else:
        brs_par_SON = np.hstack((brs_par_SON, np.nanmean(yeartemp_SON)))
# BRS Spring Winds
brs_cluster = windspeed[clusters_winds == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        brs_windspeed_SON = np.nanmean(yeartemp_SON)
    else:
        brs_windspeed_SON = np.hstack((brs_windspeed_SON, np.nanmean(yeartemp_SON)))
#%% Calculate comparisons
## SAM
# Chl-a
brs_chl_SON_SAM_POS = brs_chl_SON[sam_SON > 0.5]
brs_chl_SON_SAM_NEG = brs_chl_SON[sam_SON < -0.5]
print(np.nanmean(brs_chl_SON_SAM_POS))
print(np.nanmean(brs_chl_SON_SAM_NEG))
stats.mannwhitneyu(brs_chl_SON_SAM_POS, brs_chl_SON_SAM_NEG, nan_policy = 'omit') # ***
# SST
brs_sst_SON_SAM_POS = brs_sst_SON[sam_SON > 0.5]
brs_sst_SON_SAM_NEG = brs_sst_SON[sam_SON < -0.5]
print(np.nanmean(brs_sst_SON_SAM_POS))
print(np.nanmean(brs_sst_SON_SAM_NEG))
stats.mannwhitneyu(brs_sst_SON_SAM_POS, brs_sst_SON_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
brs_seaice_SON_SAM_POS = brs_seaice_SON[sam_SON > 0.5]
brs_seaice_SON_SAM_NEG = brs_seaice_SON[sam_SON < -0.5]
print(np.nanmean(brs_seaice_SON_SAM_POS))
print(np.nanmean(brs_seaice_SON_SAM_NEG))
stats.mannwhitneyu(brs_seaice_SON_SAM_POS, brs_seaice_SON_SAM_NEG, nan_policy = 'omit') #
# PAR
brs_par_SON_SAM_POS = brs_par_SON[sam_SON > 0.5]
brs_par_SON_SAM_NEG = brs_par_SON[sam_SON < -0.5]
print(np.nanmean(brs_par_SON_SAM_POS))
print(np.nanmean(brs_par_SON_SAM_NEG))
stats.mannwhitneyu(brs_par_SON_SAM_POS, brs_par_SON_SAM_NEG, nan_policy = 'omit') # **
# Winds
brs_windspeed_SON_SAM_POS = brs_windspeed_SON[sam_SON > 0.5]
brs_windspeed_SON_SAM_NEG = brs_windspeed_SON[sam_SON < -0.5]
print(np.nanmean(brs_windspeed_SON_SAM_POS))
print(np.nanmean(brs_windspeed_SON_SAM_NEG))
stats.mannwhitneyu(brs_windspeed_SON_SAM_POS, brs_windspeed_SON_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
brs_chl_SON_MEI_POS = brs_chl_SON[mei_SON > 0.5]
brs_chl_SON_MEI_NEG = brs_chl_SON[mei_SON < -0.5]
print(np.nanmean(brs_chl_SON_MEI_POS))
print(np.nanmean(brs_chl_SON_MEI_NEG))
stats.mannwhitneyu(brs_chl_SON_MEI_POS, brs_chl_SON_MEI_NEG, nan_policy = 'omit') # ***
# SST
brs_sst_SON_MEI_POS = brs_sst_SON[mei_SON > 0.5]
brs_sst_SON_MEI_NEG = brs_sst_SON[mei_SON < -0.5]
print(np.nanmean(brs_sst_SON_MEI_POS))
print(np.nanmean(brs_sst_SON_MEI_NEG))
stats.mannwhitneyu(brs_sst_SON_MEI_POS, brs_sst_SON_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
brs_seaice_SON_MEI_POS = brs_seaice_SON[mei_SON > 0.5]
brs_seaice_SON_MEI_NEG = brs_seaice_SON[mei_SON < -0.5]
print(np.nanmean(brs_seaice_SON_MEI_POS))
print(np.nanmean(brs_seaice_SON_MEI_NEG))
stats.mannwhitneyu(brs_seaice_SON_MEI_POS, brs_seaice_SON_MEI_NEG, nan_policy = 'omit') #
# PAR
brs_par_SON_MEI_POS = brs_par_SON[mei_SON > 0.5]
brs_par_SON_MEI_NEG = brs_par_SON[mei_SON < -0.5]
print(np.nanmean(brs_par_SON_MEI_POS))
print(np.nanmean(brs_par_SON_MEI_NEG))
stats.mannwhitneyu(brs_par_SON_MEI_POS, brs_par_SON_MEI_NEG, nan_policy = 'omit') # **
# Winds
brs_windspeed_SON_MEI_POS = brs_windspeed_SON[mei_SON > 0.5]
brs_windspeed_SON_MEI_NEG = brs_windspeed_SON[mei_SON < -0.5]
print(np.nanmean(brs_windspeed_SON_MEI_POS))
print(np.nanmean(brs_windspeed_SON_MEI_NEG))
stats.mannwhitneyu(brs_windspeed_SON_MEI_POS, brs_windspeed_SON_MEI_NEG, nan_policy = 'omit') # **
#%% # BRS
# Summer SAM
for i in np.arange(1998, 2023):
    yeartemp_dec = sam_daily[(time_date_years_sam == i-1) & (time_date_months_sam == 12)]
    yeartemp_jan = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 1)]
    yeartemp_feb = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        sam_DJF = np.nanmean(yeartemp_DJF)
    else:
        sam_DJF = np.hstack((sam_DJF, np.nanmean(yeartemp_DJF)))
# Summer MEI
for i in np.arange(1998, 2023):
    yeartemp_dec = meiv2[(meiv2_years == i-1) & (meiv2_months == 12)]
    yeartemp_jan = meiv2[(meiv2_years == i) & (meiv2_months == 1)]
    yeartemp_feb = meiv2[(meiv2_years == i) & (meiv2_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        mei_DJF = np.nanmean(yeartemp_DJF)
    else:
        mei_DJF = np.hstack((mei_DJF, np.nanmean(yeartemp_DJF)))
# BRS Summer Chl-a
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
# BRS Summer SST
brs_cluster = sst[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        brs_sst_DJF = np.nanmean(yeartemp_DJF)
    else:
        brs_sst_DJF = np.hstack((brs_sst_DJF, np.nanmean(yeartemp_DJF)))
# BRS Summer Sea Ice
brs_cluster = seaice[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        brs_seaice_DJF = np.nanmean(yeartemp_DJF)
    else:
        brs_seaice_DJF = np.hstack((brs_seaice_DJF, np.nanmean(yeartemp_DJF)))
# BRS Summer PAR
brs_cluster = par[clusters_par == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = brs_cluster[(time_date_years_par == i-1) & (time_date_months_par == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_par == i) & (time_date_months_par == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_par == i) & (time_date_months_par == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        brs_par_DJF = np.nanmean(yeartemp_DJF)
    else:
        brs_par_DJF = np.hstack((brs_par_DJF, np.nanmean(yeartemp_DJF)))
# BRS Summer Winds
brs_cluster = windspeed[clusters_winds == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        brs_windspeed_DJF = np.nanmean(yeartemp_DJF)
    else:
        brs_windspeed_DJF = np.hstack((brs_windspeed_DJF, np.nanmean(yeartemp_DJF)))
#%% Calculate comparisons
## SAM
# Chl-a
brs_chl_DJF_SAM_POS = brs_chl_DJF[sam_DJF > 0.5]
brs_chl_DJF_SAM_NEG = brs_chl_DJF[sam_DJF < -0.5]
print(np.nanmean(brs_chl_DJF_SAM_POS))
print(np.nanmean(brs_chl_DJF_SAM_NEG))
stats.mannwhitneyu(brs_chl_DJF_SAM_POS, brs_chl_DJF_SAM_NEG, nan_policy = 'omit') # ***
# SST
brs_sst_DJF_SAM_POS = brs_sst_DJF[sam_DJF > 0.5]
brs_sst_DJF_SAM_NEG = brs_sst_DJF[sam_DJF < -0.5]
print(np.nanmean(brs_sst_DJF_SAM_POS))
print(np.nanmean(brs_sst_DJF_SAM_NEG))
stats.mannwhitneyu(brs_sst_DJF_SAM_POS, brs_sst_DJF_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
brs_seaice_DJF_SAM_POS = brs_seaice_DJF[sam_DJF > 0.5]
brs_seaice_DJF_SAM_NEG = brs_seaice_DJF[sam_DJF < -0.5]
print(np.nanmean(brs_seaice_DJF_SAM_POS))
print(np.nanmean(brs_seaice_DJF_SAM_NEG))
stats.mannwhitneyu(brs_seaice_DJF_SAM_POS, brs_seaice_DJF_SAM_NEG, nan_policy = 'omit') #
# PAR
brs_par_DJF_SAM_POS = brs_par_DJF[sam_DJF > 0.5]
brs_par_DJF_SAM_NEG = brs_par_DJF[sam_DJF < -0.5]
print(np.nanmean(brs_par_DJF_SAM_POS))
print(np.nanmean(brs_par_DJF_SAM_NEG))
stats.mannwhitneyu(brs_par_DJF_SAM_POS, brs_par_DJF_SAM_NEG, nan_policy = 'omit') # **
# Winds
brs_windspeed_DJF_SAM_POS = brs_windspeed_DJF[sam_DJF > 0.5]
brs_windspeed_DJF_SAM_NEG = brs_windspeed_DJF[sam_DJF < -0.5]
print(np.nanmean(brs_windspeed_DJF_SAM_POS))
print(np.nanmean(brs_windspeed_DJF_SAM_NEG))
stats.mannwhitneyu(brs_windspeed_DJF_SAM_POS, brs_windspeed_DJF_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
brs_chl_DJF_MEI_POS = brs_chl_DJF[mei_DJF > 0.5]
brs_chl_DJF_MEI_NEG = brs_chl_DJF[mei_DJF < -0.5]
print(np.nanmean(brs_chl_DJF_MEI_POS))
print(np.nanmean(brs_chl_DJF_MEI_NEG))
stats.mannwhitneyu(brs_chl_DJF_MEI_POS, brs_chl_DJF_MEI_NEG, nan_policy = 'omit') # ***
# SST
brs_sst_DJF_MEI_POS = brs_sst_DJF[mei_DJF > 0.5]
brs_sst_DJF_MEI_NEG = brs_sst_DJF[mei_DJF < -0.5]
print(np.nanmean(brs_sst_DJF_MEI_POS))
print(np.nanmean(brs_sst_DJF_MEI_NEG))
stats.mannwhitneyu(brs_sst_DJF_MEI_POS, brs_sst_DJF_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
brs_seaice_DJF_MEI_POS = brs_seaice_DJF[mei_DJF > 0.5]
brs_seaice_DJF_MEI_NEG = brs_seaice_DJF[mei_DJF < -0.5]
print(np.nanmean(brs_seaice_DJF_MEI_POS))
print(np.nanmean(brs_seaice_DJF_MEI_NEG))
stats.mannwhitneyu(brs_seaice_DJF_MEI_POS, brs_seaice_DJF_MEI_NEG, nan_policy = 'omit') #
# PAR
brs_par_DJF_MEI_POS = brs_par_DJF[mei_DJF > 0.5]
brs_par_DJF_MEI_NEG = brs_par_DJF[mei_DJF < -0.5]
print(np.nanmean(brs_par_DJF_MEI_POS))
print(np.nanmean(brs_par_DJF_MEI_NEG))
stats.mannwhitneyu(brs_par_DJF_MEI_POS, brs_par_DJF_MEI_NEG, nan_policy = 'omit') # **
# Winds
brs_windspeed_DJF_MEI_POS = brs_windspeed_DJF[mei_DJF > 0.5]
brs_windspeed_DJF_MEI_NEG = brs_windspeed_DJF[mei_DJF < -0.5]
print(np.nanmean(brs_windspeed_DJF_MEI_POS))
print(np.nanmean(brs_windspeed_DJF_MEI_NEG))
stats.mannwhitneyu(brs_windspeed_DJF_MEI_POS, brs_windspeed_DJF_MEI_NEG, nan_policy = 'omit') # **
#%% # BRS
# Autumn SAM
for i in np.arange(1998, 2023):
    yeartemp_mar = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 3)]
    yeartemp_apr = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        sam_MA = np.nanmean(yeartemp_MA)
    else:
        sam_MA = np.hstack((sam_MA, np.nanmean(yeartemp_MA)))
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_mar = meiv2[(meiv2_years == i) & (meiv2_months == 3)]
    yeartemp_apr = meiv2[(meiv2_years == i) & (meiv2_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        mei_MA = np.nanmean(yeartemp_MA)
    else:
        mei_MA = np.hstack((mei_MA, np.nanmean(yeartemp_MA)))
# BRS Autumn Chl-a
brs_cluster = chl[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
for i in np.arange(1998, 2023):
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        brs_chl_MA = np.nanmean(yeartemp_MA)
    else:
        brs_chl_MA = np.hstack((brs_chl_MA, np.nanmean(yeartemp_MA)))
# BRS Autumn SST
brs_cluster = sst[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        brs_sst_MA = np.nanmean(yeartemp_MA)
    else:
        brs_sst_MA = np.hstack((brs_sst_MA, np.nanmean(yeartemp_MA)))
# BRS Autumn Sea Ice
brs_cluster = seaice[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        brs_seaice_MA = np.nanmean(yeartemp_MA)
    else:
        brs_seaice_MA = np.hstack((brs_seaice_MA, np.nanmean(yeartemp_MA)))
# BRS Autumn PAR
brs_cluster = par[clusters_par == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = brs_cluster[(time_date_years_par == i) & (time_date_months_par == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_par == i) & (time_date_months_par == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        brs_par_MA = np.nanmean(yeartemp_MA)
    else:
        brs_par_MA = np.hstack((brs_par_MA, np.nanmean(yeartemp_MA)))
# BRS Autumn Winds
brs_cluster = windspeed[clusters_winds == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        brs_windspeed_MA = np.nanmean(yeartemp_MA)
    else:
        brs_windspeed_MA = np.hstack((brs_windspeed_MA, np.nanmean(yeartemp_MA)))
#%% Calculate comparisons
## SAM
# Chl-a
brs_chl_MA_SAM_POS = brs_chl_MA[sam_MA > 0.5]
brs_chl_MA_SAM_NEG = brs_chl_MA[sam_MA < -0.5]
print(np.nanmean(brs_chl_MA_SAM_POS))
print(np.nanmean(brs_chl_MA_SAM_NEG))
stats.mannwhitneyu(brs_chl_MA_SAM_POS, brs_chl_MA_SAM_NEG, nan_policy = 'omit') # ***
# SST
brs_sst_MA_SAM_POS = brs_sst_MA[sam_MA > 0.5]
brs_sst_MA_SAM_NEG = brs_sst_MA[sam_MA < -0.5]
print(np.nanmean(brs_sst_MA_SAM_POS))
print(np.nanmean(brs_sst_MA_SAM_NEG))
stats.mannwhitneyu(brs_sst_MA_SAM_POS, brs_sst_MA_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
brs_seaice_MA_SAM_POS = brs_seaice_MA[sam_MA > 0.5]
brs_seaice_MA_SAM_NEG = brs_seaice_MA[sam_MA < -0.5]
print(np.nanmean(brs_seaice_MA_SAM_POS))
print(np.nanmean(brs_seaice_MA_SAM_NEG))
stats.mannwhitneyu(brs_seaice_MA_SAM_POS, brs_seaice_MA_SAM_NEG, nan_policy = 'omit') #
# PAR
brs_par_MA_SAM_POS = brs_par_MA[sam_MA > 0.5]
brs_par_MA_SAM_NEG = brs_par_MA[sam_MA < -0.5]
print(np.nanmean(brs_par_MA_SAM_POS))
print(np.nanmean(brs_par_MA_SAM_NEG))
stats.mannwhitneyu(brs_par_MA_SAM_POS, brs_par_MA_SAM_NEG, nan_policy = 'omit') # **
# Winds
brs_windspeed_MA_SAM_POS = brs_windspeed_MA[sam_MA > 0.5]
brs_windspeed_MA_SAM_NEG = brs_windspeed_MA[sam_MA < -0.5]
print(np.nanmean(brs_windspeed_MA_SAM_POS))
print(np.nanmean(brs_windspeed_MA_SAM_NEG))
stats.mannwhitneyu(brs_windspeed_MA_SAM_POS, brs_windspeed_MA_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
brs_chl_MA_MEI_POS = brs_chl_MA[mei_MA > 0.5]
brs_chl_MA_MEI_NEG = brs_chl_MA[mei_MA < -0.5]
print(np.nanmean(brs_chl_MA_MEI_POS))
print(np.nanmean(brs_chl_MA_MEI_NEG))
stats.mannwhitneyu(brs_chl_MA_MEI_POS, brs_chl_MA_MEI_NEG, nan_policy = 'omit') # ***
# SST
brs_sst_MA_MEI_POS = brs_sst_MA[mei_MA > 0.5]
brs_sst_MA_MEI_NEG = brs_sst_MA[mei_MA < -0.5]
print(np.nanmean(brs_sst_MA_MEI_POS))
print(np.nanmean(brs_sst_MA_MEI_NEG))
stats.mannwhitneyu(brs_sst_MA_MEI_POS, brs_sst_MA_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
brs_seaice_MA_MEI_POS = brs_seaice_MA[mei_MA > 0.5]
brs_seaice_MA_MEI_NEG = brs_seaice_MA[mei_MA < -0.5]
print(np.nanmean(brs_seaice_MA_MEI_POS))
print(np.nanmean(brs_seaice_MA_MEI_NEG))
stats.mannwhitneyu(brs_seaice_MA_MEI_POS, brs_seaice_MA_MEI_NEG, nan_policy = 'omit') #
# PAR
brs_par_MA_MEI_POS = brs_par_MA[mei_MA > 0.5]
brs_par_MA_MEI_NEG = brs_par_MA[mei_MA < -0.5]
print(np.nanmean(brs_par_MA_MEI_POS))
print(np.nanmean(brs_par_MA_MEI_NEG))
stats.mannwhitneyu(brs_par_MA_MEI_POS, brs_par_MA_MEI_NEG, nan_policy = 'omit') # **
# Winds
brs_windspeed_MA_MEI_POS = brs_windspeed_MA[mei_MA > 0.5]
brs_windspeed_MA_MEI_NEG = brs_windspeed_MA[mei_MA < -0.5]
print(np.nanmean(brs_windspeed_MA_MEI_POS))
print(np.nanmean(brs_windspeed_MA_MEI_NEG))
stats.mannwhitneyu(brs_windspeed_MA_MEI_POS, brs_windspeed_MA_MEI_NEG, nan_policy = 'omit') # **
#%% September - April!
# Autumn SAM
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
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_sepdec = meiv2[(meiv2_years == i-1) & ((meiv2_months == 9) | (meiv2_months == 10)
                                                                | (meiv2_months == 11) | (meiv2_months == 12))]
    yeartemp_janapr = meiv2[(meiv2_years == i) & ((meiv2_months == 1) | (meiv2_months == 2)
                                                                | (meiv2_months == 3) | (meiv2_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))

    if i == 1998:
        mei_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        mei_SEPAPR = np.hstack((mei_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# BRS Autumn Chl-a
brs_cluster = chl[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sepdec = brs_cluster[(time_date_years == i-1) & ((time_date_months == 9) | (time_date_months == 10)
                                                                | (time_date_months == 11) | (time_date_months == 12))]
    yeartemp_janapr = brs_cluster[(time_date_years == i) & ((time_date_months == 1) | (time_date_months == 2)
                                                                | (time_date_months == 3) | (time_date_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        brs_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        brs_chl_SEPAPR = np.hstack((brs_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# BRS Autumn SST
brs_cluster = sst[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = brs_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = brs_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        brs_sst_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        brs_sst_SEPAPR = np.hstack((brs_sst_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# BRS Autumn Sea Ice
brs_cluster = seaice[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = brs_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = brs_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        brs_seaice_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        brs_seaice_SEPAPR = np.hstack((brs_seaice_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# BRS Autumn PAR
brs_cluster = par[clusters_par == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = brs_cluster[(time_date_years_par == i-1) & ((time_date_months_par == 9) | (time_date_months_par == 10)
                                                                | (time_date_months_par == 11) | (time_date_months_par == 12))]
    yeartemp_janapr = brs_cluster[(time_date_years_par == i) & ((time_date_months_par == 1) | (time_date_months_par == 2)
                                                                | (time_date_months_par == 3) | (time_date_months_par == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        brs_par_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        brs_par_SEPAPR = np.hstack((brs_par_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# BRS Autumn Winds
brs_cluster = windspeed[clusters_winds == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = brs_cluster[(time_date_years_winds == i-1) & ((time_date_months_winds == 9) | (time_date_months_winds == 10)
                                                                | (time_date_months_winds == 11) | (time_date_months_winds == 12))]
    yeartemp_janapr = brs_cluster[(time_date_years_winds == i) & ((time_date_months_winds == 1) | (time_date_months_winds == 2)
                                                                | (time_date_months_winds == 3) | (time_date_months_winds == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        brs_windspeed_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        brs_windspeed_SEPAPR = np.hstack((brs_windspeed_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate comparisons
## SAM
# Chl-a
brs_chl_SEPAPR_SAM_POS = brs_chl_SEPAPR[sam_SEPAPR > 0.5]
brs_chl_SEPAPR_SAM_NEG = brs_chl_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(brs_chl_SEPAPR_SAM_POS))
print(np.nanmean(brs_chl_SEPAPR_SAM_NEG))
stats.mannwhitneyu(brs_chl_SEPAPR_SAM_POS, brs_chl_SEPAPR_SAM_NEG, nan_policy = 'omit') # ***
# SST
brs_sst_SEPAPR_SAM_POS = brs_sst_SEPAPR[sam_SEPAPR > 0.5]
brs_sst_SEPAPR_SAM_NEG = brs_sst_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(brs_sst_SEPAPR_SAM_POS))
print(np.nanmean(brs_sst_SEPAPR_SAM_NEG))
stats.mannwhitneyu(brs_sst_SEPAPR_SAM_POS, brs_sst_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
brs_seaice_SEPAPR_SAM_POS = brs_seaice_SEPAPR[sam_SEPAPR > 0.5]
brs_seaice_SEPAPR_SAM_NEG = brs_seaice_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(brs_seaice_SEPAPR_SAM_POS))
print(np.nanmean(brs_seaice_SEPAPR_SAM_NEG))
stats.mannwhitneyu(brs_seaice_SEPAPR_SAM_POS, brs_seaice_SEPAPR_SAM_NEG, nan_policy = 'omit') #
# PAR
brs_par_SEPAPR_SAM_POS = brs_par_SEPAPR[sam_SEPAPR > 0.5]
brs_par_SEPAPR_SAM_NEG = brs_par_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(brs_par_SEPAPR_SAM_POS))
print(np.nanmean(brs_par_SEPAPR_SAM_NEG))
stats.mannwhitneyu(brs_par_SEPAPR_SAM_POS, brs_par_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Winds
brs_windspeed_SEPAPR_SAM_POS = brs_windspeed_SEPAPR[sam_SEPAPR > 0.5]
brs_windspeed_SEPAPR_SAM_NEG = brs_windspeed_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(brs_windspeed_SEPAPR_SAM_POS))
print(np.nanmean(brs_windspeed_SEPAPR_SAM_NEG))
stats.mannwhitneyu(brs_windspeed_SEPAPR_SAM_POS, brs_windspeed_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
brs_chl_SEPAPR_MEI_POS = brs_chl_SEPAPR[mei_SEPAPR > 0.5]
brs_chl_SEPAPR_MEI_NEG = brs_chl_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(brs_chl_SEPAPR_MEI_POS))
print(np.nanmean(brs_chl_SEPAPR_MEI_NEG))
stats.mannwhitneyu(brs_chl_SEPAPR_MEI_POS, brs_chl_SEPAPR_MEI_NEG, nan_policy = 'omit') # ***
# SST
brs_sst_SEPAPR_MEI_POS = brs_sst_SEPAPR[mei_SEPAPR > 0.5]
brs_sst_SEPAPR_MEI_NEG = brs_sst_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(brs_sst_SEPAPR_MEI_POS))
print(np.nanmean(brs_sst_SEPAPR_MEI_NEG))
stats.mannwhitneyu(brs_sst_SEPAPR_MEI_POS, brs_sst_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
brs_seaice_SEPAPR_MEI_POS = brs_seaice_SEPAPR[mei_SEPAPR > 0.5]
brs_seaice_SEPAPR_MEI_NEG = brs_seaice_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(brs_seaice_SEPAPR_MEI_POS))
print(np.nanmean(brs_seaice_SEPAPR_MEI_NEG))
stats.mannwhitneyu(brs_seaice_SEPAPR_MEI_POS, brs_seaice_SEPAPR_MEI_NEG, nan_policy = 'omit') #
# PAR
brs_par_SEPAPR_MEI_POS = brs_par_SEPAPR[mei_SEPAPR > 0.5]
brs_par_SEPAPR_MEI_NEG = brs_par_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(brs_par_SEPAPR_MEI_POS))
print(np.nanmean(brs_par_SEPAPR_MEI_NEG))
stats.mannwhitneyu(brs_par_SEPAPR_MEI_POS, brs_par_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Winds
brs_windspeed_SEPAPR_MEI_POS = brs_windspeed_SEPAPR[mei_SEPAPR > 0.5]
brs_windspeed_SEPAPR_MEI_NEG = brs_windspeed_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(brs_windspeed_SEPAPR_MEI_POS))
print(np.nanmean(brs_windspeed_SEPAPR_MEI_NEG))
stats.mannwhitneyu(brs_windspeed_SEPAPR_MEI_POS, brs_windspeed_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
#%% WEDN
# Spring SAM
# WEDN Spring Chl-a
wedn_cluster = chl[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        wedn_chl_SON = np.nanmean(yeartemp_SON)
    else:
        wedn_chl_SON = np.hstack((wedn_chl_SON, np.nanmean(yeartemp_SON)))
# WEDN Spring SST
wedn_cluster = sst[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        wedn_sst_SON = np.nanmean(yeartemp_SON)
    else:
        wedn_sst_SON = np.hstack((wedn_sst_SON, np.nanmean(yeartemp_SON)))
# WEDN Spring Sea Ice
wedn_cluster = seaice[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        wedn_seaice_SON = np.nanmean(yeartemp_SON)
    else:
        wedn_seaice_SON = np.hstack((wedn_seaice_SON, np.nanmean(yeartemp_SON)))
# WEDN Spring PAR
wedn_cluster = par[clusters_par == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = wedn_cluster[(time_date_years_par == i-1) & (time_date_months_par == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_par == i-1) & (time_date_months_par == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_par == i-1) & (time_date_months_par == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        wedn_par_SON = np.nanmean(yeartemp_SON)
    else:
        wedn_par_SON = np.hstack((wedn_par_SON, np.nanmean(yeartemp_SON)))
# WEDN Spring Winds
wedn_cluster = windspeed[clusters_winds == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        wedn_windspeed_SON = np.nanmean(yeartemp_SON)
    else:
        wedn_windspeed_SON = np.hstack((wedn_windspeed_SON, np.nanmean(yeartemp_SON)))
#%% Calculate comparisons
## SAM
# Chl-a
wedn_chl_SON_SAM_POS = wedn_chl_SON[sam_SON > 0.5]
wedn_chl_SON_SAM_NEG = wedn_chl_SON[sam_SON < -0.5]
print(np.nanmean(wedn_chl_SON_SAM_POS))
print(np.nanmean(wedn_chl_SON_SAM_NEG))
stats.mannwhitneyu(wedn_chl_SON_SAM_POS, wedn_chl_SON_SAM_NEG, nan_policy = 'omit') # ***
# SST
wedn_sst_SON_SAM_POS = wedn_sst_SON[sam_SON > 0.5]
wedn_sst_SON_SAM_NEG = wedn_sst_SON[sam_SON < -0.5]
print(np.nanmean(wedn_sst_SON_SAM_POS))
print(np.nanmean(wedn_sst_SON_SAM_NEG))
stats.mannwhitneyu(wedn_sst_SON_SAM_POS, wedn_sst_SON_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
wedn_seaice_SON_SAM_POS = wedn_seaice_SON[sam_SON > 0.5]
wedn_seaice_SON_SAM_NEG = wedn_seaice_SON[sam_SON < -0.5]
print(np.nanmean(wedn_seaice_SON_SAM_POS))
print(np.nanmean(wedn_seaice_SON_SAM_NEG))
stats.mannwhitneyu(wedn_seaice_SON_SAM_POS, wedn_seaice_SON_SAM_NEG, nan_policy = 'omit') #
# PAR
wedn_par_SON_SAM_POS = wedn_par_SON[sam_SON > 0.5]
wedn_par_SON_SAM_NEG = wedn_par_SON[sam_SON < -0.5]
print(np.nanmean(wedn_par_SON_SAM_POS))
print(np.nanmean(wedn_par_SON_SAM_NEG))
stats.mannwhitneyu(wedn_par_SON_SAM_POS, wedn_par_SON_SAM_NEG, nan_policy = 'omit') # **
# Winds
wedn_windspeed_SON_SAM_POS = wedn_windspeed_SON[sam_SON > 0.5]
wedn_windspeed_SON_SAM_NEG = wedn_windspeed_SON[sam_SON < -0.5]
print(np.nanmean(wedn_windspeed_SON_SAM_POS))
print(np.nanmean(wedn_windspeed_SON_SAM_NEG))
stats.mannwhitneyu(wedn_windspeed_SON_SAM_POS, wedn_windspeed_SON_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
wedn_chl_SON_MEI_POS = wedn_chl_SON[mei_SON > 0.5]
wedn_chl_SON_MEI_NEG = wedn_chl_SON[mei_SON < -0.5]
print(np.nanmean(wedn_chl_SON_MEI_POS))
print(np.nanmean(wedn_chl_SON_MEI_NEG))
stats.mannwhitneyu(wedn_chl_SON_MEI_POS, wedn_chl_SON_MEI_NEG, nan_policy = 'omit') # ***
# SST
wedn_sst_SON_MEI_POS = wedn_sst_SON[mei_SON > 0.5]
wedn_sst_SON_MEI_NEG = wedn_sst_SON[mei_SON < -0.5]
print(np.nanmean(wedn_sst_SON_MEI_POS))
print(np.nanmean(wedn_sst_SON_MEI_NEG))
stats.mannwhitneyu(wedn_sst_SON_MEI_POS, wedn_sst_SON_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
wedn_seaice_SON_MEI_POS = wedn_seaice_SON[mei_SON > 0.5]
wedn_seaice_SON_MEI_NEG = wedn_seaice_SON[mei_SON < -0.5]
print(np.nanmean(wedn_seaice_SON_MEI_POS))
print(np.nanmean(wedn_seaice_SON_MEI_NEG))
stats.mannwhitneyu(wedn_seaice_SON_MEI_POS, wedn_seaice_SON_MEI_NEG, nan_policy = 'omit') #
# PAR
wedn_par_SON_MEI_POS = wedn_par_SON[mei_SON > 0.5]
wedn_par_SON_MEI_NEG = wedn_par_SON[mei_SON < -0.5]
print(np.nanmean(wedn_par_SON_MEI_POS))
print(np.nanmean(wedn_par_SON_MEI_NEG))
stats.mannwhitneyu(wedn_par_SON_MEI_POS, wedn_par_SON_MEI_NEG, nan_policy = 'omit') # **
# Winds
wedn_windspeed_SON_MEI_POS = wedn_windspeed_SON[mei_SON > 0.5]
wedn_windspeed_SON_MEI_NEG = wedn_windspeed_SON[mei_SON < -0.5]
print(np.nanmean(wedn_windspeed_SON_MEI_POS))
print(np.nanmean(wedn_windspeed_SON_MEI_NEG))
stats.mannwhitneyu(wedn_windspeed_SON_MEI_POS, wedn_windspeed_SON_MEI_NEG, nan_policy = 'omit') # **
#%% # WEDN
# Summer SAM
for i in np.arange(1998, 2023):
    yeartemp_dec = sam_daily[(time_date_years_sam == i-1) & (time_date_months_sam == 12)]
    yeartemp_jan = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 1)]
    yeartemp_feb = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        sam_DJF = np.nanmean(yeartemp_DJF)
    else:
        sam_DJF = np.hstack((sam_DJF, np.nanmean(yeartemp_DJF)))
# Summer MEI
for i in np.arange(1998, 2023):
    yeartemp_dec = meiv2[(meiv2_years == i-1) & (meiv2_months == 12)]
    yeartemp_jan = meiv2[(meiv2_years == i) & (meiv2_months == 1)]
    yeartemp_feb = meiv2[(meiv2_years == i) & (meiv2_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        mei_DJF = np.nanmean(yeartemp_DJF)
    else:
        mei_DJF = np.hstack((mei_DJF, np.nanmean(yeartemp_DJF)))
# WEDN Summer Chl-a
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
# WEDN Summer SST
wedn_cluster = sst[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        wedn_sst_DJF = np.nanmean(yeartemp_DJF)
    else:
        wedn_sst_DJF = np.hstack((wedn_sst_DJF, np.nanmean(yeartemp_DJF)))
# WEDN Summer Sea Ice
wedn_cluster = seaice[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        wedn_seaice_DJF = np.nanmean(yeartemp_DJF)
    else:
        wedn_seaice_DJF = np.hstack((wedn_seaice_DJF, np.nanmean(yeartemp_DJF)))
# WEDN Summer PAR
wedn_cluster = par[clusters_par == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = wedn_cluster[(time_date_years_par == i-1) & (time_date_months_par == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_par == i) & (time_date_months_par == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_par == i) & (time_date_months_par == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        wedn_par_DJF = np.nanmean(yeartemp_DJF)
    else:
        wedn_par_DJF = np.hstack((wedn_par_DJF, np.nanmean(yeartemp_DJF)))
# WEDN Summer Winds
wedn_cluster = windspeed[clusters_winds == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        wedn_windspeed_DJF = np.nanmean(yeartemp_DJF)
    else:
        wedn_windspeed_DJF = np.hstack((wedn_windspeed_DJF, np.nanmean(yeartemp_DJF)))
#%% Calculate comparisons
## SAM
# Chl-a
wedn_chl_DJF_SAM_POS = wedn_chl_DJF[sam_DJF > 0.5]
wedn_chl_DJF_SAM_NEG = wedn_chl_DJF[sam_DJF < -0.5]
print(np.nanmean(wedn_chl_DJF_SAM_POS))
print(np.nanmean(wedn_chl_DJF_SAM_NEG))
stats.mannwhitneyu(wedn_chl_DJF_SAM_POS, wedn_chl_DJF_SAM_NEG, nan_policy = 'omit') # ***
# SST
wedn_sst_DJF_SAM_POS = wedn_sst_DJF[sam_DJF > 0.5]
wedn_sst_DJF_SAM_NEG = wedn_sst_DJF[sam_DJF < -0.5]
print(np.nanmean(wedn_sst_DJF_SAM_POS))
print(np.nanmean(wedn_sst_DJF_SAM_NEG))
stats.mannwhitneyu(wedn_sst_DJF_SAM_POS, wedn_sst_DJF_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
wedn_seaice_DJF_SAM_POS = wedn_seaice_DJF[sam_DJF > 0.5]
wedn_seaice_DJF_SAM_NEG = wedn_seaice_DJF[sam_DJF < -0.5]
print(np.nanmean(wedn_seaice_DJF_SAM_POS))
print(np.nanmean(wedn_seaice_DJF_SAM_NEG))
stats.mannwhitneyu(wedn_seaice_DJF_SAM_POS, wedn_seaice_DJF_SAM_NEG, nan_policy = 'omit') #
# PAR
wedn_par_DJF_SAM_POS = wedn_par_DJF[sam_DJF > 0.5]
wedn_par_DJF_SAM_NEG = wedn_par_DJF[sam_DJF < -0.5]
print(np.nanmean(wedn_par_DJF_SAM_POS))
print(np.nanmean(wedn_par_DJF_SAM_NEG))
stats.mannwhitneyu(wedn_par_DJF_SAM_POS, wedn_par_DJF_SAM_NEG, nan_policy = 'omit') # **
# Winds
wedn_windspeed_DJF_SAM_POS = wedn_windspeed_DJF[sam_DJF > 0.5]
wedn_windspeed_DJF_SAM_NEG = wedn_windspeed_DJF[sam_DJF < -0.5]
print(np.nanmean(wedn_windspeed_DJF_SAM_POS))
print(np.nanmean(wedn_windspeed_DJF_SAM_NEG))
stats.mannwhitneyu(wedn_windspeed_DJF_SAM_POS, wedn_windspeed_DJF_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
wedn_chl_DJF_MEI_POS = wedn_chl_DJF[mei_DJF > 0.5]
wedn_chl_DJF_MEI_NEG = wedn_chl_DJF[mei_DJF < -0.5]
print(np.nanmean(wedn_chl_DJF_MEI_POS))
print(np.nanmean(wedn_chl_DJF_MEI_NEG))
stats.mannwhitneyu(wedn_chl_DJF_MEI_POS, wedn_chl_DJF_MEI_NEG, nan_policy = 'omit') # ***
# SST
wedn_sst_DJF_MEI_POS = wedn_sst_DJF[mei_DJF > 0.5]
wedn_sst_DJF_MEI_NEG = wedn_sst_DJF[mei_DJF < -0.5]
print(np.nanmean(wedn_sst_DJF_MEI_POS))
print(np.nanmean(wedn_sst_DJF_MEI_NEG))
stats.mannwhitneyu(wedn_sst_DJF_MEI_POS, wedn_sst_DJF_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
wedn_seaice_DJF_MEI_POS = wedn_seaice_DJF[mei_DJF > 0.5]
wedn_seaice_DJF_MEI_NEG = wedn_seaice_DJF[mei_DJF < -0.5]
print(np.nanmean(wedn_seaice_DJF_MEI_POS))
print(np.nanmean(wedn_seaice_DJF_MEI_NEG))
stats.mannwhitneyu(wedn_seaice_DJF_MEI_POS, wedn_seaice_DJF_MEI_NEG, nan_policy = 'omit') #
# PAR
wedn_par_DJF_MEI_POS = wedn_par_DJF[mei_DJF > 0.5]
wedn_par_DJF_MEI_NEG = wedn_par_DJF[mei_DJF < -0.5]
print(np.nanmean(wedn_par_DJF_MEI_POS))
print(np.nanmean(wedn_par_DJF_MEI_NEG))
stats.mannwhitneyu(wedn_par_DJF_MEI_POS, wedn_par_DJF_MEI_NEG, nan_policy = 'omit') # **
# Winds
wedn_windspeed_DJF_MEI_POS = wedn_windspeed_DJF[mei_DJF > 0.5]
wedn_windspeed_DJF_MEI_NEG = wedn_windspeed_DJF[mei_DJF < -0.5]
print(np.nanmean(wedn_windspeed_DJF_MEI_POS))
print(np.nanmean(wedn_windspeed_DJF_MEI_NEG))
stats.mannwhitneyu(wedn_windspeed_DJF_MEI_POS, wedn_windspeed_DJF_MEI_NEG, nan_policy = 'omit') # **
#%% # WEDN
# Autumn SAM
for i in np.arange(1998, 2023):
    yeartemp_mar = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 3)]
    yeartemp_apr = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        sam_MA = np.nanmean(yeartemp_MA)
    else:
        sam_MA = np.hstack((sam_MA, np.nanmean(yeartemp_MA)))
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_mar = meiv2[(meiv2_years == i) & (meiv2_months == 3)]
    yeartemp_apr = meiv2[(meiv2_years == i) & (meiv2_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        mei_MA = np.nanmean(yeartemp_MA)
    else:
        mei_MA = np.hstack((mei_MA, np.nanmean(yeartemp_MA)))
# WEDN Autumn Chl-a
wedn_cluster = chl[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
for i in np.arange(1998, 2023):
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        wedn_chl_MA = np.nanmean(yeartemp_MA)
    else:
        wedn_chl_MA = np.hstack((wedn_chl_MA, np.nanmean(yeartemp_MA)))
# WEDN Autumn SST
wedn_cluster = sst[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        wedn_sst_MA = np.nanmean(yeartemp_MA)
    else:
        wedn_sst_MA = np.hstack((wedn_sst_MA, np.nanmean(yeartemp_MA)))
# WEDN Autumn Sea Ice
wedn_cluster = seaice[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        wedn_seaice_MA = np.nanmean(yeartemp_MA)
    else:
        wedn_seaice_MA = np.hstack((wedn_seaice_MA, np.nanmean(yeartemp_MA)))
# WEDN Autumn PAR
wedn_cluster = par[clusters_par == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = wedn_cluster[(time_date_years_par == i) & (time_date_months_par == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_par == i) & (time_date_months_par == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        wedn_par_MA = np.nanmean(yeartemp_MA)
    else:
        wedn_par_MA = np.hstack((wedn_par_MA, np.nanmean(yeartemp_MA)))
# WEDN Autumn Winds
wedn_cluster = windspeed[clusters_winds == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        wedn_windspeed_MA = np.nanmean(yeartemp_MA)
    else:
        wedn_windspeed_MA = np.hstack((wedn_windspeed_MA, np.nanmean(yeartemp_MA)))
#%% Calculate comparisons
## SAM
# Chl-a
wedn_chl_MA_SAM_POS = wedn_chl_MA[sam_MA > 0.5]
wedn_chl_MA_SAM_NEG = wedn_chl_MA[sam_MA < -0.5]
print(np.nanmean(wedn_chl_MA_SAM_POS))
print(np.nanmean(wedn_chl_MA_SAM_NEG))
stats.mannwhitneyu(wedn_chl_MA_SAM_POS, wedn_chl_MA_SAM_NEG, nan_policy = 'omit') # ***
# SST
wedn_sst_MA_SAM_POS = wedn_sst_MA[sam_MA > 0.5]
wedn_sst_MA_SAM_NEG = wedn_sst_MA[sam_MA < -0.5]
print(np.nanmean(wedn_sst_MA_SAM_POS))
print(np.nanmean(wedn_sst_MA_SAM_NEG))
stats.mannwhitneyu(wedn_sst_MA_SAM_POS, wedn_sst_MA_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
wedn_seaice_MA_SAM_POS = wedn_seaice_MA[sam_MA > 0.5]
wedn_seaice_MA_SAM_NEG = wedn_seaice_MA[sam_MA < -0.5]
print(np.nanmean(wedn_seaice_MA_SAM_POS))
print(np.nanmean(wedn_seaice_MA_SAM_NEG))
stats.mannwhitneyu(wedn_seaice_MA_SAM_POS, wedn_seaice_MA_SAM_NEG, nan_policy = 'omit') #
# PAR
wedn_par_MA_SAM_POS = wedn_par_MA[sam_MA > 0.5]
wedn_par_MA_SAM_NEG = wedn_par_MA[sam_MA < -0.5]
print(np.nanmean(wedn_par_MA_SAM_POS))
print(np.nanmean(wedn_par_MA_SAM_NEG))
stats.mannwhitneyu(wedn_par_MA_SAM_POS, wedn_par_MA_SAM_NEG, nan_policy = 'omit') # **
# Winds
wedn_windspeed_MA_SAM_POS = wedn_windspeed_MA[sam_MA > 0.5]
wedn_windspeed_MA_SAM_NEG = wedn_windspeed_MA[sam_MA < -0.5]
print(np.nanmean(wedn_windspeed_MA_SAM_POS))
print(np.nanmean(wedn_windspeed_MA_SAM_NEG))
stats.mannwhitneyu(wedn_windspeed_MA_SAM_POS, wedn_windspeed_MA_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
wedn_chl_MA_MEI_POS = wedn_chl_MA[mei_MA > 0.5]
wedn_chl_MA_MEI_NEG = wedn_chl_MA[mei_MA < -0.5]
print(np.nanmean(wedn_chl_MA_MEI_POS))
print(np.nanmean(wedn_chl_MA_MEI_NEG))
stats.mannwhitneyu(wedn_chl_MA_MEI_POS, wedn_chl_MA_MEI_NEG, nan_policy = 'omit') # ***
# SST
wedn_sst_MA_MEI_POS = wedn_sst_MA[mei_MA > 0.5]
wedn_sst_MA_MEI_NEG = wedn_sst_MA[mei_MA < -0.5]
print(np.nanmean(wedn_sst_MA_MEI_POS))
print(np.nanmean(wedn_sst_MA_MEI_NEG))
stats.mannwhitneyu(wedn_sst_MA_MEI_POS, wedn_sst_MA_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
wedn_seaice_MA_MEI_POS = wedn_seaice_MA[mei_MA > 0.5]
wedn_seaice_MA_MEI_NEG = wedn_seaice_MA[mei_MA < -0.5]
print(np.nanmean(wedn_seaice_MA_MEI_POS))
print(np.nanmean(wedn_seaice_MA_MEI_NEG))
stats.mannwhitneyu(wedn_seaice_MA_MEI_POS, wedn_seaice_MA_MEI_NEG, nan_policy = 'omit') #
# PAR
wedn_par_MA_MEI_POS = wedn_par_MA[mei_MA > 0.5]
wedn_par_MA_MEI_NEG = wedn_par_MA[mei_MA < -0.5]
print(np.nanmean(wedn_par_MA_MEI_POS))
print(np.nanmean(wedn_par_MA_MEI_NEG))
stats.mannwhitneyu(wedn_par_MA_MEI_POS, wedn_par_MA_MEI_NEG, nan_policy = 'omit') # **
# Winds
wedn_windspeed_MA_MEI_POS = wedn_windspeed_MA[mei_MA > 0.5]
wedn_windspeed_MA_MEI_NEG = wedn_windspeed_MA[mei_MA < -0.5]
print(np.nanmean(wedn_windspeed_MA_MEI_POS))
print(np.nanmean(wedn_windspeed_MA_MEI_NEG))
stats.mannwhitneyu(wedn_windspeed_MA_MEI_POS, wedn_windspeed_MA_MEI_NEG, nan_policy = 'omit') # **
#%% September - April!
# Autumn SAM
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
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_sepdec = meiv2[(meiv2_years == i-1) & ((meiv2_months == 9) | (meiv2_months == 10)
                                                                | (meiv2_months == 11) | (meiv2_months == 12))]
    yeartemp_janapr = meiv2[(meiv2_years == i) & ((meiv2_months == 1) | (meiv2_months == 2)
                                                                | (meiv2_months == 3) | (meiv2_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))

    if i == 1998:
        mei_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        mei_SEPAPR = np.hstack((mei_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDN Autumn Chl-a
wedn_cluster = chl[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sepdec = wedn_cluster[(time_date_years == i-1) & ((time_date_months == 9) | (time_date_months == 10)
                                                                | (time_date_months == 11) | (time_date_months == 12))]
    yeartemp_janapr = wedn_cluster[(time_date_years == i) & ((time_date_months == 1) | (time_date_months == 2)
                                                                | (time_date_months == 3) | (time_date_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        wedn_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        wedn_chl_SEPAPR = np.hstack((wedn_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDN Autumn SST
wedn_cluster = sst[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = wedn_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = wedn_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        wedn_sst_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        wedn_sst_SEPAPR = np.hstack((wedn_sst_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDN Autumn Sea Ice
wedn_cluster = seaice[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = wedn_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = wedn_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        wedn_seaice_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        wedn_seaice_SEPAPR = np.hstack((wedn_seaice_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDN Autumn PAR
wedn_cluster = par[clusters_par == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = wedn_cluster[(time_date_years_par == i-1) & ((time_date_months_par == 9) | (time_date_months_par == 10)
                                                                | (time_date_months_par == 11) | (time_date_months_par == 12))]
    yeartemp_janapr = wedn_cluster[(time_date_years_par == i) & ((time_date_months_par == 1) | (time_date_months_par == 2)
                                                                | (time_date_months_par == 3) | (time_date_months_par == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        wedn_par_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        wedn_par_SEPAPR = np.hstack((wedn_par_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDN Autumn Winds
wedn_cluster = windspeed[clusters_winds == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = wedn_cluster[(time_date_years_winds == i-1) & ((time_date_months_winds == 9) | (time_date_months_winds == 10)
                                                                | (time_date_months_winds == 11) | (time_date_months_winds == 12))]
    yeartemp_janapr = wedn_cluster[(time_date_years_winds == i) & ((time_date_months_winds == 1) | (time_date_months_winds == 2)
                                                                | (time_date_months_winds == 3) | (time_date_months_winds == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        wedn_windspeed_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        wedn_windspeed_SEPAPR = np.hstack((wedn_windspeed_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate comparisons
## SAM
# Chl-a
wedn_chl_SEPAPR_SAM_POS = wedn_chl_SEPAPR[sam_SEPAPR > 0.5]
wedn_chl_SEPAPR_SAM_NEG = wedn_chl_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(wedn_chl_SEPAPR_SAM_POS))
print(np.nanmean(wedn_chl_SEPAPR_SAM_NEG))
stats.mannwhitneyu(wedn_chl_SEPAPR_SAM_POS, wedn_chl_SEPAPR_SAM_NEG, nan_policy = 'omit') # ***
# SST
wedn_sst_SEPAPR_SAM_POS = wedn_sst_SEPAPR[sam_SEPAPR > 0.5]
wedn_sst_SEPAPR_SAM_NEG = wedn_sst_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(wedn_sst_SEPAPR_SAM_POS))
print(np.nanmean(wedn_sst_SEPAPR_SAM_NEG))
stats.mannwhitneyu(wedn_sst_SEPAPR_SAM_POS, wedn_sst_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
wedn_seaice_SEPAPR_SAM_POS = wedn_seaice_SEPAPR[sam_SEPAPR > 0.5]
wedn_seaice_SEPAPR_SAM_NEG = wedn_seaice_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(wedn_seaice_SEPAPR_SAM_POS))
print(np.nanmean(wedn_seaice_SEPAPR_SAM_NEG))
stats.mannwhitneyu(wedn_seaice_SEPAPR_SAM_POS, wedn_seaice_SEPAPR_SAM_NEG, nan_policy = 'omit') #
# PAR
wedn_par_SEPAPR_SAM_POS = wedn_par_SEPAPR[sam_SEPAPR > 0.5]
wedn_par_SEPAPR_SAM_NEG = wedn_par_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(wedn_par_SEPAPR_SAM_POS))
print(np.nanmean(wedn_par_SEPAPR_SAM_NEG))
stats.mannwhitneyu(wedn_par_SEPAPR_SAM_POS, wedn_par_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Winds
wedn_windspeed_SEPAPR_SAM_POS = wedn_windspeed_SEPAPR[sam_SEPAPR > 0.5]
wedn_windspeed_SEPAPR_SAM_NEG = wedn_windspeed_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(wedn_windspeed_SEPAPR_SAM_POS))
print(np.nanmean(wedn_windspeed_SEPAPR_SAM_NEG))
stats.mannwhitneyu(wedn_windspeed_SEPAPR_SAM_POS, wedn_windspeed_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
wedn_chl_SEPAPR_MEI_POS = wedn_chl_SEPAPR[mei_SEPAPR > 0.5]
wedn_chl_SEPAPR_MEI_NEG = wedn_chl_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(wedn_chl_SEPAPR_MEI_POS))
print(np.nanmean(wedn_chl_SEPAPR_MEI_NEG))
stats.mannwhitneyu(wedn_chl_SEPAPR_MEI_POS, wedn_chl_SEPAPR_MEI_NEG, nan_policy = 'omit') # ***
# SST
wedn_sst_SEPAPR_MEI_POS = wedn_sst_SEPAPR[mei_SEPAPR > 0.5]
wedn_sst_SEPAPR_MEI_NEG = wedn_sst_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(wedn_sst_SEPAPR_MEI_POS))
print(np.nanmean(wedn_sst_SEPAPR_MEI_NEG))
stats.mannwhitneyu(wedn_sst_SEPAPR_MEI_POS, wedn_sst_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
wedn_seaice_SEPAPR_MEI_POS = wedn_seaice_SEPAPR[mei_SEPAPR > 0.5]
wedn_seaice_SEPAPR_MEI_NEG = wedn_seaice_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(wedn_seaice_SEPAPR_MEI_POS))
print(np.nanmean(wedn_seaice_SEPAPR_MEI_NEG))
stats.mannwhitneyu(wedn_seaice_SEPAPR_MEI_POS, wedn_seaice_SEPAPR_MEI_NEG, nan_policy = 'omit') #
# PAR
wedn_par_SEPAPR_MEI_POS = wedn_par_SEPAPR[mei_SEPAPR > 0.5]
wedn_par_SEPAPR_MEI_NEG = wedn_par_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(wedn_par_SEPAPR_MEI_POS))
print(np.nanmean(wedn_par_SEPAPR_MEI_NEG))
stats.mannwhitneyu(wedn_par_SEPAPR_MEI_POS, wedn_par_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Winds
wedn_windspeed_SEPAPR_MEI_POS = wedn_windspeed_SEPAPR[mei_SEPAPR > 0.5]
wedn_windspeed_SEPAPR_MEI_NEG = wedn_windspeed_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(wedn_windspeed_SEPAPR_MEI_POS))
print(np.nanmean(wedn_windspeed_SEPAPR_MEI_NEG))
stats.mannwhitneyu(wedn_windspeed_SEPAPR_MEI_POS, wedn_windspeed_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
#%% GES
# Spring SAM
# GES Spring Chl-a
ges_cluster = chl[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        ges_chl_SON = np.nanmean(yeartemp_SON)
    else:
        ges_chl_SON = np.hstack((ges_chl_SON, np.nanmean(yeartemp_SON)))
# GES Spring SST
ges_cluster = sst[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        ges_sst_SON = np.nanmean(yeartemp_SON)
    else:
        ges_sst_SON = np.hstack((ges_sst_SON, np.nanmean(yeartemp_SON)))
# GES Spring Sea Ice
ges_cluster = seaice[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        ges_seaice_SON = np.nanmean(yeartemp_SON)
    else:
        ges_seaice_SON = np.hstack((ges_seaice_SON, np.nanmean(yeartemp_SON)))
# GES Spring PAR
ges_cluster = par[clusters_par == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = ges_cluster[(time_date_years_par == i-1) & (time_date_months_par == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_par == i-1) & (time_date_months_par == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_par == i-1) & (time_date_months_par == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        ges_par_SON = np.nanmean(yeartemp_SON)
    else:
        ges_par_SON = np.hstack((ges_par_SON, np.nanmean(yeartemp_SON)))
# GES Spring Winds
ges_cluster = windspeed[clusters_winds == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        ges_windspeed_SON = np.nanmean(yeartemp_SON)
    else:
        ges_windspeed_SON = np.hstack((ges_windspeed_SON, np.nanmean(yeartemp_SON)))
#%% Calculate comparisons
## SAM
# Chl-a
ges_chl_SON_SAM_POS = ges_chl_SON[sam_SON > 0.5]
ges_chl_SON_SAM_NEG = ges_chl_SON[sam_SON < -0.5]
print(np.nanmean(ges_chl_SON_SAM_POS))
print(np.nanmean(ges_chl_SON_SAM_NEG))
stats.mannwhitneyu(ges_chl_SON_SAM_POS, ges_chl_SON_SAM_NEG, nan_policy = 'omit') # ***
# SST
ges_sst_SON_SAM_POS = ges_sst_SON[sam_SON > 0.5]
ges_sst_SON_SAM_NEG = ges_sst_SON[sam_SON < -0.5]
print(np.nanmean(ges_sst_SON_SAM_POS))
print(np.nanmean(ges_sst_SON_SAM_NEG))
stats.mannwhitneyu(ges_sst_SON_SAM_POS, ges_sst_SON_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
ges_seaice_SON_SAM_POS = ges_seaice_SON[sam_SON > 0.5]
ges_seaice_SON_SAM_NEG = ges_seaice_SON[sam_SON < -0.5]
print(np.nanmean(ges_seaice_SON_SAM_POS))
print(np.nanmean(ges_seaice_SON_SAM_NEG))
stats.mannwhitneyu(ges_seaice_SON_SAM_POS, ges_seaice_SON_SAM_NEG, nan_policy = 'omit') #
# PAR
ges_par_SON_SAM_POS = ges_par_SON[sam_SON > 0.5]
ges_par_SON_SAM_NEG = ges_par_SON[sam_SON < -0.5]
print(np.nanmean(ges_par_SON_SAM_POS))
print(np.nanmean(ges_par_SON_SAM_NEG))
stats.mannwhitneyu(ges_par_SON_SAM_POS, ges_par_SON_SAM_NEG, nan_policy = 'omit') # **
# Winds
ges_windspeed_SON_SAM_POS = ges_windspeed_SON[sam_SON > 0.5]
ges_windspeed_SON_SAM_NEG = ges_windspeed_SON[sam_SON < -0.5]
print(np.nanmean(ges_windspeed_SON_SAM_POS))
print(np.nanmean(ges_windspeed_SON_SAM_NEG))
stats.mannwhitneyu(ges_windspeed_SON_SAM_POS, ges_windspeed_SON_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
ges_chl_SON_MEI_POS = ges_chl_SON[mei_SON > 0.5]
ges_chl_SON_MEI_NEG = ges_chl_SON[mei_SON < -0.5]
print(np.nanmean(ges_chl_SON_MEI_POS))
print(np.nanmean(ges_chl_SON_MEI_NEG))
stats.mannwhitneyu(ges_chl_SON_MEI_POS, ges_chl_SON_MEI_NEG, nan_policy = 'omit') # ***
# SST
ges_sst_SON_MEI_POS = ges_sst_SON[mei_SON > 0.5]
ges_sst_SON_MEI_NEG = ges_sst_SON[mei_SON < -0.5]
print(np.nanmean(ges_sst_SON_MEI_POS))
print(np.nanmean(ges_sst_SON_MEI_NEG))
stats.mannwhitneyu(ges_sst_SON_MEI_POS, ges_sst_SON_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
ges_seaice_SON_MEI_POS = ges_seaice_SON[mei_SON > 0.5]
ges_seaice_SON_MEI_NEG = ges_seaice_SON[mei_SON < -0.5]
print(np.nanmean(ges_seaice_SON_MEI_POS))
print(np.nanmean(ges_seaice_SON_MEI_NEG))
stats.mannwhitneyu(ges_seaice_SON_MEI_POS, ges_seaice_SON_MEI_NEG, nan_policy = 'omit') #
# PAR
ges_par_SON_MEI_POS = ges_par_SON[mei_SON > 0.5]
ges_par_SON_MEI_NEG = ges_par_SON[mei_SON < -0.5]
print(np.nanmean(ges_par_SON_MEI_POS))
print(np.nanmean(ges_par_SON_MEI_NEG))
stats.mannwhitneyu(ges_par_SON_MEI_POS, ges_par_SON_MEI_NEG, nan_policy = 'omit') # **
# Winds
ges_windspeed_SON_MEI_POS = ges_windspeed_SON[mei_SON > 0.5]
ges_windspeed_SON_MEI_NEG = ges_windspeed_SON[mei_SON < -0.5]
print(np.nanmean(ges_windspeed_SON_MEI_POS))
print(np.nanmean(ges_windspeed_SON_MEI_NEG))
stats.mannwhitneyu(ges_windspeed_SON_MEI_POS, ges_windspeed_SON_MEI_NEG, nan_policy = 'omit') # **
#%% # GES
# Summer SAM
for i in np.arange(1998, 2023):
    yeartemp_dec = sam_daily[(time_date_years_sam == i-1) & (time_date_months_sam == 12)]
    yeartemp_jan = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 1)]
    yeartemp_feb = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        sam_DJF = np.nanmean(yeartemp_DJF)
    else:
        sam_DJF = np.hstack((sam_DJF, np.nanmean(yeartemp_DJF)))
# Summer MEI
for i in np.arange(1998, 2023):
    yeartemp_dec = meiv2[(meiv2_years == i-1) & (meiv2_months == 12)]
    yeartemp_jan = meiv2[(meiv2_years == i) & (meiv2_months == 1)]
    yeartemp_feb = meiv2[(meiv2_years == i) & (meiv2_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        mei_DJF = np.nanmean(yeartemp_DJF)
    else:
        mei_DJF = np.hstack((mei_DJF, np.nanmean(yeartemp_DJF)))
# GES Summer Chl-a
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
# GES Summer SST
ges_cluster = sst[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        ges_sst_DJF = np.nanmean(yeartemp_DJF)
    else:
        ges_sst_DJF = np.hstack((ges_sst_DJF, np.nanmean(yeartemp_DJF)))
# GES Summer Sea Ice
ges_cluster = seaice[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        ges_seaice_DJF = np.nanmean(yeartemp_DJF)
    else:
        ges_seaice_DJF = np.hstack((ges_seaice_DJF, np.nanmean(yeartemp_DJF)))
# GES Summer PAR
ges_cluster = par[clusters_par == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = ges_cluster[(time_date_years_par == i-1) & (time_date_months_par == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_par == i) & (time_date_months_par == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_par == i) & (time_date_months_par == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        ges_par_DJF = np.nanmean(yeartemp_DJF)
    else:
        ges_par_DJF = np.hstack((ges_par_DJF, np.nanmean(yeartemp_DJF)))
# GES Summer Winds
ges_cluster = windspeed[clusters_winds == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        ges_windspeed_DJF = np.nanmean(yeartemp_DJF)
    else:
        ges_windspeed_DJF = np.hstack((ges_windspeed_DJF, np.nanmean(yeartemp_DJF)))
#%% Calculate comparisons
## SAM
# Chl-a
ges_chl_DJF_SAM_POS = ges_chl_DJF[sam_DJF > 0.5]
ges_chl_DJF_SAM_NEG = ges_chl_DJF[sam_DJF < -0.5]
print(np.nanmean(ges_chl_DJF_SAM_POS))
print(np.nanmean(ges_chl_DJF_SAM_NEG))
stats.mannwhitneyu(ges_chl_DJF_SAM_POS, ges_chl_DJF_SAM_NEG, nan_policy = 'omit') # ***
# SST
ges_sst_DJF_SAM_POS = ges_sst_DJF[sam_DJF > 0.5]
ges_sst_DJF_SAM_NEG = ges_sst_DJF[sam_DJF < -0.5]
print(np.nanmean(ges_sst_DJF_SAM_POS))
print(np.nanmean(ges_sst_DJF_SAM_NEG))
stats.mannwhitneyu(ges_sst_DJF_SAM_POS, ges_sst_DJF_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
ges_seaice_DJF_SAM_POS = ges_seaice_DJF[sam_DJF > 0.5]
ges_seaice_DJF_SAM_NEG = ges_seaice_DJF[sam_DJF < -0.5]
print(np.nanmean(ges_seaice_DJF_SAM_POS))
print(np.nanmean(ges_seaice_DJF_SAM_NEG))
stats.mannwhitneyu(ges_seaice_DJF_SAM_POS, ges_seaice_DJF_SAM_NEG, nan_policy = 'omit') #
# PAR
ges_par_DJF_SAM_POS = ges_par_DJF[sam_DJF > 0.5]
ges_par_DJF_SAM_NEG = ges_par_DJF[sam_DJF < -0.5]
print(np.nanmean(ges_par_DJF_SAM_POS))
print(np.nanmean(ges_par_DJF_SAM_NEG))
stats.mannwhitneyu(ges_par_DJF_SAM_POS, ges_par_DJF_SAM_NEG, nan_policy = 'omit') # **
# Winds
ges_windspeed_DJF_SAM_POS = ges_windspeed_DJF[sam_DJF > 0.5]
ges_windspeed_DJF_SAM_NEG = ges_windspeed_DJF[sam_DJF < -0.5]
print(np.nanmean(ges_windspeed_DJF_SAM_POS))
print(np.nanmean(ges_windspeed_DJF_SAM_NEG))
stats.mannwhitneyu(ges_windspeed_DJF_SAM_POS, ges_windspeed_DJF_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
ges_chl_DJF_MEI_POS = ges_chl_DJF[mei_DJF > 0.5]
ges_chl_DJF_MEI_NEG = ges_chl_DJF[mei_DJF < -0.5]
print(np.nanmean(ges_chl_DJF_MEI_POS))
print(np.nanmean(ges_chl_DJF_MEI_NEG))
stats.mannwhitneyu(ges_chl_DJF_MEI_POS, ges_chl_DJF_MEI_NEG, nan_policy = 'omit') # ***
# SST
ges_sst_DJF_MEI_POS = ges_sst_DJF[mei_DJF > 0.5]
ges_sst_DJF_MEI_NEG = ges_sst_DJF[mei_DJF < -0.5]
print(np.nanmean(ges_sst_DJF_MEI_POS))
print(np.nanmean(ges_sst_DJF_MEI_NEG))
stats.mannwhitneyu(ges_sst_DJF_MEI_POS, ges_sst_DJF_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
ges_seaice_DJF_MEI_POS = ges_seaice_DJF[mei_DJF > 0.5]
ges_seaice_DJF_MEI_NEG = ges_seaice_DJF[mei_DJF < -0.5]
print(np.nanmean(ges_seaice_DJF_MEI_POS))
print(np.nanmean(ges_seaice_DJF_MEI_NEG))
stats.mannwhitneyu(ges_seaice_DJF_MEI_POS, ges_seaice_DJF_MEI_NEG, nan_policy = 'omit') #
# PAR
ges_par_DJF_MEI_POS = ges_par_DJF[mei_DJF > 0.5]
ges_par_DJF_MEI_NEG = ges_par_DJF[mei_DJF < -0.5]
print(np.nanmean(ges_par_DJF_MEI_POS))
print(np.nanmean(ges_par_DJF_MEI_NEG))
stats.mannwhitneyu(ges_par_DJF_MEI_POS, ges_par_DJF_MEI_NEG, nan_policy = 'omit') # **
# Winds
ges_windspeed_DJF_MEI_POS = ges_windspeed_DJF[mei_DJF > 0.5]
ges_windspeed_DJF_MEI_NEG = ges_windspeed_DJF[mei_DJF < -0.5]
print(np.nanmean(ges_windspeed_DJF_MEI_POS))
print(np.nanmean(ges_windspeed_DJF_MEI_NEG))
stats.mannwhitneyu(ges_windspeed_DJF_MEI_POS, ges_windspeed_DJF_MEI_NEG, nan_policy = 'omit') # **
#%% # GES
# Autumn SAM
for i in np.arange(1998, 2023):
    yeartemp_mar = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 3)]
    yeartemp_apr = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        sam_MA = np.nanmean(yeartemp_MA)
    else:
        sam_MA = np.hstack((sam_MA, np.nanmean(yeartemp_MA)))
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_mar = meiv2[(meiv2_years == i) & (meiv2_months == 3)]
    yeartemp_apr = meiv2[(meiv2_years == i) & (meiv2_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        mei_MA = np.nanmean(yeartemp_MA)
    else:
        mei_MA = np.hstack((mei_MA, np.nanmean(yeartemp_MA)))
# GES Autumn Chl-a
ges_cluster = chl[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
for i in np.arange(1998, 2023):
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        ges_chl_MA = np.nanmean(yeartemp_MA)
    else:
        ges_chl_MA = np.hstack((ges_chl_MA, np.nanmean(yeartemp_MA)))
# GES Autumn SST
ges_cluster = sst[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        ges_sst_MA = np.nanmean(yeartemp_MA)
    else:
        ges_sst_MA = np.hstack((ges_sst_MA, np.nanmean(yeartemp_MA)))
# GES Autumn Sea Ice
ges_cluster = seaice[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        ges_seaice_MA = np.nanmean(yeartemp_MA)
    else:
        ges_seaice_MA = np.hstack((ges_seaice_MA, np.nanmean(yeartemp_MA)))
# GES Autumn PAR
ges_cluster = par[clusters_par == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = ges_cluster[(time_date_years_par == i) & (time_date_months_par == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_par == i) & (time_date_months_par == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        ges_par_MA = np.nanmean(yeartemp_MA)
    else:
        ges_par_MA = np.hstack((ges_par_MA, np.nanmean(yeartemp_MA)))
# GES Autumn Winds
ges_cluster = windspeed[clusters_winds == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        ges_windspeed_MA = np.nanmean(yeartemp_MA)
    else:
        ges_windspeed_MA = np.hstack((ges_windspeed_MA, np.nanmean(yeartemp_MA)))
#%% Calculate comparisons
## SAM
# Chl-a
ges_chl_MA_SAM_POS = ges_chl_MA[sam_MA > 0.5]
ges_chl_MA_SAM_NEG = ges_chl_MA[sam_MA < -0.5]
print(np.nanmean(ges_chl_MA_SAM_POS))
print(np.nanmean(ges_chl_MA_SAM_NEG))
stats.mannwhitneyu(ges_chl_MA_SAM_POS, ges_chl_MA_SAM_NEG, nan_policy = 'omit') # ***
# SST
ges_sst_MA_SAM_POS = ges_sst_MA[sam_MA > 0.5]
ges_sst_MA_SAM_NEG = ges_sst_MA[sam_MA < -0.5]
print(np.nanmean(ges_sst_MA_SAM_POS))
print(np.nanmean(ges_sst_MA_SAM_NEG))
stats.mannwhitneyu(ges_sst_MA_SAM_POS, ges_sst_MA_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
ges_seaice_MA_SAM_POS = ges_seaice_MA[sam_MA > 0.5]
ges_seaice_MA_SAM_NEG = ges_seaice_MA[sam_MA < -0.5]
print(np.nanmean(ges_seaice_MA_SAM_POS))
print(np.nanmean(ges_seaice_MA_SAM_NEG))
stats.mannwhitneyu(ges_seaice_MA_SAM_POS, ges_seaice_MA_SAM_NEG, nan_policy = 'omit') #
# PAR
ges_par_MA_SAM_POS = ges_par_MA[sam_MA > 0.5]
ges_par_MA_SAM_NEG = ges_par_MA[sam_MA < -0.5]
print(np.nanmean(ges_par_MA_SAM_POS))
print(np.nanmean(ges_par_MA_SAM_NEG))
stats.mannwhitneyu(ges_par_MA_SAM_POS, ges_par_MA_SAM_NEG, nan_policy = 'omit') # **
# Winds
ges_windspeed_MA_SAM_POS = ges_windspeed_MA[sam_MA > 0.5]
ges_windspeed_MA_SAM_NEG = ges_windspeed_MA[sam_MA < -0.5]
print(np.nanmean(ges_windspeed_MA_SAM_POS))
print(np.nanmean(ges_windspeed_MA_SAM_NEG))
stats.mannwhitneyu(ges_windspeed_MA_SAM_POS, ges_windspeed_MA_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
ges_chl_MA_MEI_POS = ges_chl_MA[mei_MA > 0.5]
ges_chl_MA_MEI_NEG = ges_chl_MA[mei_MA < -0.5]
print(np.nanmean(ges_chl_MA_MEI_POS))
print(np.nanmean(ges_chl_MA_MEI_NEG))
stats.mannwhitneyu(ges_chl_MA_MEI_POS, ges_chl_MA_MEI_NEG, nan_policy = 'omit') # ***
# SST
ges_sst_MA_MEI_POS = ges_sst_MA[mei_MA > 0.5]
ges_sst_MA_MEI_NEG = ges_sst_MA[mei_MA < -0.5]
print(np.nanmean(ges_sst_MA_MEI_POS))
print(np.nanmean(ges_sst_MA_MEI_NEG))
stats.mannwhitneyu(ges_sst_MA_MEI_POS, ges_sst_MA_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
ges_seaice_MA_MEI_POS = ges_seaice_MA[mei_MA > 0.5]
ges_seaice_MA_MEI_NEG = ges_seaice_MA[mei_MA < -0.5]
print(np.nanmean(ges_seaice_MA_MEI_POS))
print(np.nanmean(ges_seaice_MA_MEI_NEG))
stats.mannwhitneyu(ges_seaice_MA_MEI_POS, ges_seaice_MA_MEI_NEG, nan_policy = 'omit') #
# PAR
ges_par_MA_MEI_POS = ges_par_MA[mei_MA > 0.5]
ges_par_MA_MEI_NEG = ges_par_MA[mei_MA < -0.5]
print(np.nanmean(ges_par_MA_MEI_POS))
print(np.nanmean(ges_par_MA_MEI_NEG))
stats.mannwhitneyu(ges_par_MA_MEI_POS, ges_par_MA_MEI_NEG, nan_policy = 'omit') # **
# Winds
ges_windspeed_MA_MEI_POS = ges_windspeed_MA[mei_MA > 0.5]
ges_windspeed_MA_MEI_NEG = ges_windspeed_MA[mei_MA < -0.5]
print(np.nanmean(ges_windspeed_MA_MEI_POS))
print(np.nanmean(ges_windspeed_MA_MEI_NEG))
stats.mannwhitneyu(ges_windspeed_MA_MEI_POS, ges_windspeed_MA_MEI_NEG, nan_policy = 'omit') # **
#%% September - April!
# Autumn SAM
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
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_sepdec = meiv2[(meiv2_years == i-1) & ((meiv2_months == 9) | (meiv2_months == 10)
                                                                | (meiv2_months == 11) | (meiv2_months == 12))]
    yeartemp_janapr = meiv2[(meiv2_years == i) & ((meiv2_months == 1) | (meiv2_months == 2)
                                                                | (meiv2_months == 3) | (meiv2_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))

    if i == 1998:
        mei_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        mei_SEPAPR = np.hstack((mei_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# GES Autumn Chl-a
ges_cluster = chl[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sepdec = ges_cluster[(time_date_years == i-1) & ((time_date_months == 9) | (time_date_months == 10)
                                                                | (time_date_months == 11) | (time_date_months == 12))]
    yeartemp_janapr = ges_cluster[(time_date_years == i) & ((time_date_months == 1) | (time_date_months == 2)
                                                                | (time_date_months == 3) | (time_date_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        ges_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        ges_chl_SEPAPR = np.hstack((ges_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# GES Autumn SST
ges_cluster = sst[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = ges_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = ges_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        ges_sst_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        ges_sst_SEPAPR = np.hstack((ges_sst_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# GES Autumn Sea Ice
ges_cluster = seaice[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = ges_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = ges_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        ges_seaice_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        ges_seaice_SEPAPR = np.hstack((ges_seaice_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# GES Autumn PAR
ges_cluster = par[clusters_par == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = ges_cluster[(time_date_years_par == i-1) & ((time_date_months_par == 9) | (time_date_months_par == 10)
                                                                | (time_date_months_par == 11) | (time_date_months_par == 12))]
    yeartemp_janapr = ges_cluster[(time_date_years_par == i) & ((time_date_months_par == 1) | (time_date_months_par == 2)
                                                                | (time_date_months_par == 3) | (time_date_months_par == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        ges_par_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        ges_par_SEPAPR = np.hstack((ges_par_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# GES Autumn Winds
ges_cluster = windspeed[clusters_winds == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = ges_cluster[(time_date_years_winds == i-1) & ((time_date_months_winds == 9) | (time_date_months_winds == 10)
                                                                | (time_date_months_winds == 11) | (time_date_months_winds == 12))]
    yeartemp_janapr = ges_cluster[(time_date_years_winds == i) & ((time_date_months_winds == 1) | (time_date_months_winds == 2)
                                                                | (time_date_months_winds == 3) | (time_date_months_winds == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        ges_windspeed_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        ges_windspeed_SEPAPR = np.hstack((ges_windspeed_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate comparisons
## SAM
# Chl-a
ges_chl_SEPAPR_SAM_POS = ges_chl_SEPAPR[sam_SEPAPR > 0.5]
ges_chl_SEPAPR_SAM_NEG = ges_chl_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(ges_chl_SEPAPR_SAM_POS))
print(np.nanmean(ges_chl_SEPAPR_SAM_NEG))
stats.mannwhitneyu(ges_chl_SEPAPR_SAM_POS, ges_chl_SEPAPR_SAM_NEG, nan_policy = 'omit') # ***
# SST
ges_sst_SEPAPR_SAM_POS = ges_sst_SEPAPR[sam_SEPAPR > 0.5]
ges_sst_SEPAPR_SAM_NEG = ges_sst_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(ges_sst_SEPAPR_SAM_POS))
print(np.nanmean(ges_sst_SEPAPR_SAM_NEG))
stats.mannwhitneyu(ges_sst_SEPAPR_SAM_POS, ges_sst_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
ges_seaice_SEPAPR_SAM_POS = ges_seaice_SEPAPR[sam_SEPAPR > 0.5]
ges_seaice_SEPAPR_SAM_NEG = ges_seaice_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(ges_seaice_SEPAPR_SAM_POS))
print(np.nanmean(ges_seaice_SEPAPR_SAM_NEG))
stats.mannwhitneyu(ges_seaice_SEPAPR_SAM_POS, ges_seaice_SEPAPR_SAM_NEG, nan_policy = 'omit') #
# PAR
ges_par_SEPAPR_SAM_POS = ges_par_SEPAPR[sam_SEPAPR > 0.5]
ges_par_SEPAPR_SAM_NEG = ges_par_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(ges_par_SEPAPR_SAM_POS))
print(np.nanmean(ges_par_SEPAPR_SAM_NEG))
stats.mannwhitneyu(ges_par_SEPAPR_SAM_POS, ges_par_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Winds
ges_windspeed_SEPAPR_SAM_POS = ges_windspeed_SEPAPR[sam_SEPAPR > 0.5]
ges_windspeed_SEPAPR_SAM_NEG = ges_windspeed_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(ges_windspeed_SEPAPR_SAM_POS))
print(np.nanmean(ges_windspeed_SEPAPR_SAM_NEG))
stats.mannwhitneyu(ges_windspeed_SEPAPR_SAM_POS, ges_windspeed_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
ges_chl_SEPAPR_MEI_POS = ges_chl_SEPAPR[mei_SEPAPR > 0.5]
ges_chl_SEPAPR_MEI_NEG = ges_chl_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(ges_chl_SEPAPR_MEI_POS))
print(np.nanmean(ges_chl_SEPAPR_MEI_NEG))
stats.mannwhitneyu(ges_chl_SEPAPR_MEI_POS, ges_chl_SEPAPR_MEI_NEG, nan_policy = 'omit') # ***
# SST
ges_sst_SEPAPR_MEI_POS = ges_sst_SEPAPR[mei_SEPAPR > 0.5]
ges_sst_SEPAPR_MEI_NEG = ges_sst_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(ges_sst_SEPAPR_MEI_POS))
print(np.nanmean(ges_sst_SEPAPR_MEI_NEG))
stats.mannwhitneyu(ges_sst_SEPAPR_MEI_POS, ges_sst_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
ges_seaice_SEPAPR_MEI_POS = ges_seaice_SEPAPR[mei_SEPAPR > 0.5]
ges_seaice_SEPAPR_MEI_NEG = ges_seaice_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(ges_seaice_SEPAPR_MEI_POS))
print(np.nanmean(ges_seaice_SEPAPR_MEI_NEG))
stats.mannwhitneyu(ges_seaice_SEPAPR_MEI_POS, ges_seaice_SEPAPR_MEI_NEG, nan_policy = 'omit') #
# PAR
ges_par_SEPAPR_MEI_POS = ges_par_SEPAPR[mei_SEPAPR > 0.5]
ges_par_SEPAPR_MEI_NEG = ges_par_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(ges_par_SEPAPR_MEI_POS))
print(np.nanmean(ges_par_SEPAPR_MEI_NEG))
stats.mannwhitneyu(ges_par_SEPAPR_MEI_POS, ges_par_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Winds
ges_windspeed_SEPAPR_MEI_POS = ges_windspeed_SEPAPR[mei_SEPAPR > 0.5]
ges_windspeed_SEPAPR_MEI_NEG = ges_windspeed_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(ges_windspeed_SEPAPR_MEI_POS))
print(np.nanmean(ges_windspeed_SEPAPR_MEI_NEG))
stats.mannwhitneyu(ges_windspeed_SEPAPR_MEI_POS, ges_windspeed_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
#%% WEDS
# Spring SAM
# WEDS Spring Chl-a
weds_cluster = chl[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        weds_chl_SON = np.nanmean(yeartemp_SON)
    else:
        weds_chl_SON = np.hstack((weds_chl_SON, np.nanmean(yeartemp_SON)))
# WEDS Spring SST
weds_cluster = sst[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        weds_sst_SON = np.nanmean(yeartemp_SON)
    else:
        weds_sst_SON = np.hstack((weds_sst_SON, np.nanmean(yeartemp_SON)))
# WEDS Spring Sea Ice
weds_cluster = seaice[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        weds_seaice_SON = np.nanmean(yeartemp_SON)
    else:
        weds_seaice_SON = np.hstack((weds_seaice_SON, np.nanmean(yeartemp_SON)))
# WEDS Spring PAR
weds_cluster = par[clusters_par == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = weds_cluster[(time_date_years_par == i-1) & (time_date_months_par == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_par == i-1) & (time_date_months_par == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_par == i-1) & (time_date_months_par == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        weds_par_SON = np.nanmean(yeartemp_SON)
    else:
        weds_par_SON = np.hstack((weds_par_SON, np.nanmean(yeartemp_SON)))
# WEDS Spring Winds
weds_cluster = windspeed[clusters_winds == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sep = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        weds_windspeed_SON = np.nanmean(yeartemp_SON)
    else:
        weds_windspeed_SON = np.hstack((weds_windspeed_SON, np.nanmean(yeartemp_SON)))
#%% Calculate comparisons
## SAM
# Chl-a
weds_chl_SON_SAM_POS = weds_chl_SON[sam_SON > 0.5]
weds_chl_SON_SAM_NEG = weds_chl_SON[sam_SON < -0.5]
print(np.nanmean(weds_chl_SON_SAM_POS))
print(np.nanmean(weds_chl_SON_SAM_NEG))
stats.mannwhitneyu(weds_chl_SON_SAM_POS, weds_chl_SON_SAM_NEG, nan_policy = 'omit') # ***
# SST
weds_sst_SON_SAM_POS = weds_sst_SON[sam_SON > 0.5]
weds_sst_SON_SAM_NEG = weds_sst_SON[sam_SON < -0.5]
print(np.nanmean(weds_sst_SON_SAM_POS))
print(np.nanmean(weds_sst_SON_SAM_NEG))
stats.mannwhitneyu(weds_sst_SON_SAM_POS, weds_sst_SON_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
weds_seaice_SON_SAM_POS = weds_seaice_SON[sam_SON > 0.5]
weds_seaice_SON_SAM_NEG = weds_seaice_SON[sam_SON < -0.5]
print(np.nanmean(weds_seaice_SON_SAM_POS))
print(np.nanmean(weds_seaice_SON_SAM_NEG))
stats.mannwhitneyu(weds_seaice_SON_SAM_POS, weds_seaice_SON_SAM_NEG, nan_policy = 'omit') #
# PAR
weds_par_SON_SAM_POS = weds_par_SON[sam_SON > 0.5]
weds_par_SON_SAM_NEG = weds_par_SON[sam_SON < -0.5]
print(np.nanmean(weds_par_SON_SAM_POS))
print(np.nanmean(weds_par_SON_SAM_NEG))
stats.mannwhitneyu(weds_par_SON_SAM_POS, weds_par_SON_SAM_NEG, nan_policy = 'omit') # **
# Winds
weds_windspeed_SON_SAM_POS = weds_windspeed_SON[sam_SON > 0.5]
weds_windspeed_SON_SAM_NEG = weds_windspeed_SON[sam_SON < -0.5]
print(np.nanmean(weds_windspeed_SON_SAM_POS))
print(np.nanmean(weds_windspeed_SON_SAM_NEG))
stats.mannwhitneyu(weds_windspeed_SON_SAM_POS, weds_windspeed_SON_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
weds_chl_SON_MEI_POS = weds_chl_SON[mei_SON > 0.5]
weds_chl_SON_MEI_NEG = weds_chl_SON[mei_SON < -0.5]
print(np.nanmean(weds_chl_SON_MEI_POS))
print(np.nanmean(weds_chl_SON_MEI_NEG))
stats.mannwhitneyu(weds_chl_SON_MEI_POS, weds_chl_SON_MEI_NEG, nan_policy = 'omit') # ***
# SST
weds_sst_SON_MEI_POS = weds_sst_SON[mei_SON > 0.5]
weds_sst_SON_MEI_NEG = weds_sst_SON[mei_SON < -0.5]
print(np.nanmean(weds_sst_SON_MEI_POS))
print(np.nanmean(weds_sst_SON_MEI_NEG))
stats.mannwhitneyu(weds_sst_SON_MEI_POS, weds_sst_SON_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
weds_seaice_SON_MEI_POS = weds_seaice_SON[mei_SON > 0.5]
weds_seaice_SON_MEI_NEG = weds_seaice_SON[mei_SON < -0.5]
print(np.nanmean(weds_seaice_SON_MEI_POS))
print(np.nanmean(weds_seaice_SON_MEI_NEG))
stats.mannwhitneyu(weds_seaice_SON_MEI_POS, weds_seaice_SON_MEI_NEG, nan_policy = 'omit') #
# PAR
weds_par_SON_MEI_POS = weds_par_SON[mei_SON > 0.5]
weds_par_SON_MEI_NEG = weds_par_SON[mei_SON < -0.5]
print(np.nanmean(weds_par_SON_MEI_POS))
print(np.nanmean(weds_par_SON_MEI_NEG))
stats.mannwhitneyu(weds_par_SON_MEI_POS, weds_par_SON_MEI_NEG, nan_policy = 'omit') # **
# Winds
weds_windspeed_SON_MEI_POS = weds_windspeed_SON[mei_SON > 0.5]
weds_windspeed_SON_MEI_NEG = weds_windspeed_SON[mei_SON < -0.5]
print(np.nanmean(weds_windspeed_SON_MEI_POS))
print(np.nanmean(weds_windspeed_SON_MEI_NEG))
stats.mannwhitneyu(weds_windspeed_SON_MEI_POS, weds_windspeed_SON_MEI_NEG, nan_policy = 'omit') # **
#%% # WEDS
# Summer SAM
for i in np.arange(1998, 2023):
    yeartemp_dec = sam_daily[(time_date_years_sam == i-1) & (time_date_months_sam == 12)]
    yeartemp_jan = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 1)]
    yeartemp_feb = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        sam_DJF = np.nanmean(yeartemp_DJF)
    else:
        sam_DJF = np.hstack((sam_DJF, np.nanmean(yeartemp_DJF)))
# Summer MEI
for i in np.arange(1998, 2023):
    yeartemp_dec = meiv2[(meiv2_years == i-1) & (meiv2_months == 12)]
    yeartemp_jan = meiv2[(meiv2_years == i) & (meiv2_months == 1)]
    yeartemp_feb = meiv2[(meiv2_years == i) & (meiv2_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        mei_DJF = np.nanmean(yeartemp_DJF)
    else:
        mei_DJF = np.hstack((mei_DJF, np.nanmean(yeartemp_DJF)))
# WEDS Summer Chl-a
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
# WEDS Summer SST
weds_cluster = sst[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        weds_sst_DJF = np.nanmean(yeartemp_DJF)
    else:
        weds_sst_DJF = np.hstack((weds_sst_DJF, np.nanmean(yeartemp_DJF)))
# WEDS Summer Sea Ice
weds_cluster = seaice[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        weds_seaice_DJF = np.nanmean(yeartemp_DJF)
    else:
        weds_seaice_DJF = np.hstack((weds_seaice_DJF, np.nanmean(yeartemp_DJF)))
# WEDS Summer PAR
weds_cluster = par[clusters_par == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = weds_cluster[(time_date_years_par == i-1) & (time_date_months_par == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_par == i) & (time_date_months_par == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_par == i) & (time_date_months_par == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        weds_par_DJF = np.nanmean(yeartemp_DJF)
    else:
        weds_par_DJF = np.hstack((weds_par_DJF, np.nanmean(yeartemp_DJF)))
# WEDS Summer Winds
weds_cluster = windspeed[clusters_winds == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_dec = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        weds_windspeed_DJF = np.nanmean(yeartemp_DJF)
    else:
        weds_windspeed_DJF = np.hstack((weds_windspeed_DJF, np.nanmean(yeartemp_DJF)))
#%% Calculate comparisons
## SAM
# Chl-a
weds_chl_DJF_SAM_POS = weds_chl_DJF[sam_DJF > 0.5]
weds_chl_DJF_SAM_NEG = weds_chl_DJF[sam_DJF < -0.5]
print(np.nanmean(weds_chl_DJF_SAM_POS))
print(np.nanmean(weds_chl_DJF_SAM_NEG))
stats.mannwhitneyu(weds_chl_DJF_SAM_POS, weds_chl_DJF_SAM_NEG, nan_policy = 'omit') # ***
# SST
weds_sst_DJF_SAM_POS = weds_sst_DJF[sam_DJF > 0.5]
weds_sst_DJF_SAM_NEG = weds_sst_DJF[sam_DJF < -0.5]
print(np.nanmean(weds_sst_DJF_SAM_POS))
print(np.nanmean(weds_sst_DJF_SAM_NEG))
stats.mannwhitneyu(weds_sst_DJF_SAM_POS, weds_sst_DJF_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
weds_seaice_DJF_SAM_POS = weds_seaice_DJF[sam_DJF > 0.5]
weds_seaice_DJF_SAM_NEG = weds_seaice_DJF[sam_DJF < -0.5]
print(np.nanmean(weds_seaice_DJF_SAM_POS))
print(np.nanmean(weds_seaice_DJF_SAM_NEG))
stats.mannwhitneyu(weds_seaice_DJF_SAM_POS, weds_seaice_DJF_SAM_NEG, nan_policy = 'omit') #
# PAR
weds_par_DJF_SAM_POS = weds_par_DJF[sam_DJF > 0.5]
weds_par_DJF_SAM_NEG = weds_par_DJF[sam_DJF < -0.5]
print(np.nanmean(weds_par_DJF_SAM_POS))
print(np.nanmean(weds_par_DJF_SAM_NEG))
stats.mannwhitneyu(weds_par_DJF_SAM_POS, weds_par_DJF_SAM_NEG, nan_policy = 'omit') # **
# Winds
weds_windspeed_DJF_SAM_POS = weds_windspeed_DJF[sam_DJF > 0.5]
weds_windspeed_DJF_SAM_NEG = weds_windspeed_DJF[sam_DJF < -0.5]
print(np.nanmean(weds_windspeed_DJF_SAM_POS))
print(np.nanmean(weds_windspeed_DJF_SAM_NEG))
stats.mannwhitneyu(weds_windspeed_DJF_SAM_POS, weds_windspeed_DJF_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
weds_chl_DJF_MEI_POS = weds_chl_DJF[mei_DJF > 0.5]
weds_chl_DJF_MEI_NEG = weds_chl_DJF[mei_DJF < -0.5]
print(np.nanmean(weds_chl_DJF_MEI_POS))
print(np.nanmean(weds_chl_DJF_MEI_NEG))
stats.mannwhitneyu(weds_chl_DJF_MEI_POS, weds_chl_DJF_MEI_NEG, nan_policy = 'omit') # ***
# SST
weds_sst_DJF_MEI_POS = weds_sst_DJF[mei_DJF > 0.5]
weds_sst_DJF_MEI_NEG = weds_sst_DJF[mei_DJF < -0.5]
print(np.nanmean(weds_sst_DJF_MEI_POS))
print(np.nanmean(weds_sst_DJF_MEI_NEG))
stats.mannwhitneyu(weds_sst_DJF_MEI_POS, weds_sst_DJF_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
weds_seaice_DJF_MEI_POS = weds_seaice_DJF[mei_DJF > 0.5]
weds_seaice_DJF_MEI_NEG = weds_seaice_DJF[mei_DJF < -0.5]
print(np.nanmean(weds_seaice_DJF_MEI_POS))
print(np.nanmean(weds_seaice_DJF_MEI_NEG))
stats.mannwhitneyu(weds_seaice_DJF_MEI_POS, weds_seaice_DJF_MEI_NEG, nan_policy = 'omit') #
# PAR
weds_par_DJF_MEI_POS = weds_par_DJF[mei_DJF > 0.5]
weds_par_DJF_MEI_NEG = weds_par_DJF[mei_DJF < -0.5]
print(np.nanmean(weds_par_DJF_MEI_POS))
print(np.nanmean(weds_par_DJF_MEI_NEG))
stats.mannwhitneyu(weds_par_DJF_MEI_POS, weds_par_DJF_MEI_NEG, nan_policy = 'omit') # **
# Winds
weds_windspeed_DJF_MEI_POS = weds_windspeed_DJF[mei_DJF > 0.5]
weds_windspeed_DJF_MEI_NEG = weds_windspeed_DJF[mei_DJF < -0.5]
print(np.nanmean(weds_windspeed_DJF_MEI_POS))
print(np.nanmean(weds_windspeed_DJF_MEI_NEG))
stats.mannwhitneyu(weds_windspeed_DJF_MEI_POS, weds_windspeed_DJF_MEI_NEG, nan_policy = 'omit') # **
#%% # WEDS
# Autumn SAM
for i in np.arange(1998, 2023):
    yeartemp_mar = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 3)]
    yeartemp_apr = sam_daily[(time_date_years_sam == i) & (time_date_months_sam == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        sam_MA = np.nanmean(yeartemp_MA)
    else:
        sam_MA = np.hstack((sam_MA, np.nanmean(yeartemp_MA)))
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_mar = meiv2[(meiv2_years == i) & (meiv2_months == 3)]
    yeartemp_apr = meiv2[(meiv2_years == i) & (meiv2_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        mei_MA = np.nanmean(yeartemp_MA)
    else:
        mei_MA = np.hstack((mei_MA, np.nanmean(yeartemp_MA)))
# WEDS Autumn Chl-a
weds_cluster = chl[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
for i in np.arange(1998, 2023):
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        weds_chl_MA = np.nanmean(yeartemp_MA)
    else:
        weds_chl_MA = np.hstack((weds_chl_MA, np.nanmean(yeartemp_MA)))
# WEDS Autumn SST
weds_cluster = sst[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        weds_sst_MA = np.nanmean(yeartemp_MA)
    else:
        weds_sst_MA = np.hstack((weds_sst_MA, np.nanmean(yeartemp_MA)))
# WEDS Autumn Sea Ice
weds_cluster = seaice[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        weds_seaice_MA = np.nanmean(yeartemp_MA)
    else:
        weds_seaice_MA = np.hstack((weds_seaice_MA, np.nanmean(yeartemp_MA)))
# WEDS Autumn PAR
weds_cluster = par[clusters_par == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = weds_cluster[(time_date_years_par == i) & (time_date_months_par == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_par == i) & (time_date_months_par == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        weds_par_MA = np.nanmean(yeartemp_MA)
    else:
        weds_par_MA = np.hstack((weds_par_MA, np.nanmean(yeartemp_MA)))
# WEDS Autumn Winds
weds_cluster = windspeed[clusters_winds == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_mar = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    yeartemp_MA = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        weds_windspeed_MA = np.nanmean(yeartemp_MA)
    else:
        weds_windspeed_MA = np.hstack((weds_windspeed_MA, np.nanmean(yeartemp_MA)))
#%% Calculate comparisons
## SAM
# Chl-a
weds_chl_MA_SAM_POS = weds_chl_MA[sam_MA > 0.5]
weds_chl_MA_SAM_NEG = weds_chl_MA[sam_MA < -0.5]
print(np.nanmean(weds_chl_MA_SAM_POS))
print(np.nanmean(weds_chl_MA_SAM_NEG))
stats.mannwhitneyu(weds_chl_MA_SAM_POS, weds_chl_MA_SAM_NEG, nan_policy = 'omit') # ***
# SST
weds_sst_MA_SAM_POS = weds_sst_MA[sam_MA > 0.5]
weds_sst_MA_SAM_NEG = weds_sst_MA[sam_MA < -0.5]
print(np.nanmean(weds_sst_MA_SAM_POS))
print(np.nanmean(weds_sst_MA_SAM_NEG))
stats.mannwhitneyu(weds_sst_MA_SAM_POS, weds_sst_MA_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
weds_seaice_MA_SAM_POS = weds_seaice_MA[sam_MA > 0.5]
weds_seaice_MA_SAM_NEG = weds_seaice_MA[sam_MA < -0.5]
print(np.nanmean(weds_seaice_MA_SAM_POS))
print(np.nanmean(weds_seaice_MA_SAM_NEG))
stats.mannwhitneyu(weds_seaice_MA_SAM_POS, weds_seaice_MA_SAM_NEG, nan_policy = 'omit') #
# PAR
weds_par_MA_SAM_POS = weds_par_MA[sam_MA > 0.5]
weds_par_MA_SAM_NEG = weds_par_MA[sam_MA < -0.5]
print(np.nanmean(weds_par_MA_SAM_POS))
print(np.nanmean(weds_par_MA_SAM_NEG))
stats.mannwhitneyu(weds_par_MA_SAM_POS, weds_par_MA_SAM_NEG, nan_policy = 'omit') # **
# Winds
weds_windspeed_MA_SAM_POS = weds_windspeed_MA[sam_MA > 0.5]
weds_windspeed_MA_SAM_NEG = weds_windspeed_MA[sam_MA < -0.5]
print(np.nanmean(weds_windspeed_MA_SAM_POS))
print(np.nanmean(weds_windspeed_MA_SAM_NEG))
stats.mannwhitneyu(weds_windspeed_MA_SAM_POS, weds_windspeed_MA_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
weds_chl_MA_MEI_POS = weds_chl_MA[mei_MA > 0.5]
weds_chl_MA_MEI_NEG = weds_chl_MA[mei_MA < -0.5]
print(np.nanmean(weds_chl_MA_MEI_POS))
print(np.nanmean(weds_chl_MA_MEI_NEG))
stats.mannwhitneyu(weds_chl_MA_MEI_POS, weds_chl_MA_MEI_NEG, nan_policy = 'omit') # ***
# SST
weds_sst_MA_MEI_POS = weds_sst_MA[mei_MA > 0.5]
weds_sst_MA_MEI_NEG = weds_sst_MA[mei_MA < -0.5]
print(np.nanmean(weds_sst_MA_MEI_POS))
print(np.nanmean(weds_sst_MA_MEI_NEG))
stats.mannwhitneyu(weds_sst_MA_MEI_POS, weds_sst_MA_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
weds_seaice_MA_MEI_POS = weds_seaice_MA[mei_MA > 0.5]
weds_seaice_MA_MEI_NEG = weds_seaice_MA[mei_MA < -0.5]
print(np.nanmean(weds_seaice_MA_MEI_POS))
print(np.nanmean(weds_seaice_MA_MEI_NEG))
stats.mannwhitneyu(weds_seaice_MA_MEI_POS, weds_seaice_MA_MEI_NEG, nan_policy = 'omit') #
# PAR
weds_par_MA_MEI_POS = weds_par_MA[mei_MA > 0.5]
weds_par_MA_MEI_NEG = weds_par_MA[mei_MA < -0.5]
print(np.nanmean(weds_par_MA_MEI_POS))
print(np.nanmean(weds_par_MA_MEI_NEG))
stats.mannwhitneyu(weds_par_MA_MEI_POS, weds_par_MA_MEI_NEG, nan_policy = 'omit') # **
# Winds
weds_windspeed_MA_MEI_POS = weds_windspeed_MA[mei_MA > 0.5]
weds_windspeed_MA_MEI_NEG = weds_windspeed_MA[mei_MA < -0.5]
print(np.nanmean(weds_windspeed_MA_MEI_POS))
print(np.nanmean(weds_windspeed_MA_MEI_NEG))
stats.mannwhitneyu(weds_windspeed_MA_MEI_POS, weds_windspeed_MA_MEI_NEG, nan_policy = 'omit') # **
#%% September - April!
# Autumn SAM
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
# Autumn MEI
for i in np.arange(1998, 2023):
    yeartemp_sepdec = meiv2[(meiv2_years == i-1) & ((meiv2_months == 9) | (meiv2_months == 10)
                                                                | (meiv2_months == 11) | (meiv2_months == 12))]
    yeartemp_janapr = meiv2[(meiv2_years == i) & ((meiv2_months == 1) | (meiv2_months == 2)
                                                                | (meiv2_months == 3) | (meiv2_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))

    if i == 1998:
        mei_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        mei_SEPAPR = np.hstack((mei_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDS Autumn Chl-a
weds_cluster = chl[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
for i in np.arange(1998, 2023):
    yeartemp_sepdec = weds_cluster[(time_date_years == i-1) & ((time_date_months == 9) | (time_date_months == 10)
                                                                | (time_date_months == 11) | (time_date_months == 12))]
    yeartemp_janapr = weds_cluster[(time_date_years == i) & ((time_date_months == 1) | (time_date_months == 2)
                                                                | (time_date_months == 3) | (time_date_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        weds_chl_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        weds_chl_SEPAPR = np.hstack((weds_chl_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDS Autumn SST
weds_cluster = sst[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = weds_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = weds_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        weds_sst_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        weds_sst_SEPAPR = np.hstack((weds_sst_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDS Autumn Sea Ice
weds_cluster = seaice[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = weds_cluster[(time_date_years_sst == i-1) & ((time_date_months_sst == 9) | (time_date_months_sst == 10)
                                                                | (time_date_months_sst == 11) | (time_date_months_sst == 12))]
    yeartemp_janapr = weds_cluster[(time_date_years_sst == i) & ((time_date_months_sst == 1) | (time_date_months_sst == 2)
                                                                | (time_date_months_sst == 3) | (time_date_months_sst == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        weds_seaice_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        weds_seaice_SEPAPR = np.hstack((weds_seaice_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDS Autumn PAR
weds_cluster = par[clusters_par == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = weds_cluster[(time_date_years_par == i-1) & ((time_date_months_par == 9) | (time_date_months_par == 10)
                                                                | (time_date_months_par == 11) | (time_date_months_par == 12))]
    yeartemp_janapr = weds_cluster[(time_date_years_par == i) & ((time_date_months_par == 1) | (time_date_months_par == 2)
                                                                | (time_date_months_par == 3) | (time_date_months_par == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        weds_par_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        weds_par_SEPAPR = np.hstack((weds_par_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
# WEDS Autumn Winds
weds_cluster = windspeed[clusters_winds == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
for i in np.arange(1998, 2023):
    yeartemp_sepdec = weds_cluster[(time_date_years_winds == i-1) & ((time_date_months_winds == 9) | (time_date_months_winds == 10)
                                                                | (time_date_months_winds == 11) | (time_date_months_winds == 12))]
    yeartemp_janapr = weds_cluster[(time_date_years_winds == i) & ((time_date_months_winds == 1) | (time_date_months_winds == 2)
                                                                | (time_date_months_winds == 3) | (time_date_months_winds == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        weds_windspeed_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        weds_windspeed_SEPAPR = np.hstack((weds_windspeed_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate comparisons
## SAM
# Chl-a
weds_chl_SEPAPR_SAM_POS = weds_chl_SEPAPR[sam_SEPAPR > 0.5]
weds_chl_SEPAPR_SAM_NEG = weds_chl_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(weds_chl_SEPAPR_SAM_POS))
print(np.nanmean(weds_chl_SEPAPR_SAM_NEG))
stats.mannwhitneyu(weds_chl_SEPAPR_SAM_POS, weds_chl_SEPAPR_SAM_NEG, nan_policy = 'omit') # ***
# SST
weds_sst_SEPAPR_SAM_POS = weds_sst_SEPAPR[sam_SEPAPR > 0.5]
weds_sst_SEPAPR_SAM_NEG = weds_sst_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(weds_sst_SEPAPR_SAM_POS))
print(np.nanmean(weds_sst_SEPAPR_SAM_NEG))
stats.mannwhitneyu(weds_sst_SEPAPR_SAM_POS, weds_sst_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Sea Ice
weds_seaice_SEPAPR_SAM_POS = weds_seaice_SEPAPR[sam_SEPAPR > 0.5]
weds_seaice_SEPAPR_SAM_NEG = weds_seaice_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(weds_seaice_SEPAPR_SAM_POS))
print(np.nanmean(weds_seaice_SEPAPR_SAM_NEG))
stats.mannwhitneyu(weds_seaice_SEPAPR_SAM_POS, weds_seaice_SEPAPR_SAM_NEG, nan_policy = 'omit') #
# PAR
weds_par_SEPAPR_SAM_POS = weds_par_SEPAPR[sam_SEPAPR > 0.5]
weds_par_SEPAPR_SAM_NEG = weds_par_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(weds_par_SEPAPR_SAM_POS))
print(np.nanmean(weds_par_SEPAPR_SAM_NEG))
stats.mannwhitneyu(weds_par_SEPAPR_SAM_POS, weds_par_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
# Winds
weds_windspeed_SEPAPR_SAM_POS = weds_windspeed_SEPAPR[sam_SEPAPR > 0.5]
weds_windspeed_SEPAPR_SAM_NEG = weds_windspeed_SEPAPR[sam_SEPAPR < -0.5]
print(np.nanmean(weds_windspeed_SEPAPR_SAM_POS))
print(np.nanmean(weds_windspeed_SEPAPR_SAM_NEG))
stats.mannwhitneyu(weds_windspeed_SEPAPR_SAM_POS, weds_windspeed_SEPAPR_SAM_NEG, nan_policy = 'omit') # **
## MEI
# Chl-a
weds_chl_SEPAPR_MEI_POS = weds_chl_SEPAPR[mei_SEPAPR > 0.5]
weds_chl_SEPAPR_MEI_NEG = weds_chl_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(weds_chl_SEPAPR_MEI_POS))
print(np.nanmean(weds_chl_SEPAPR_MEI_NEG))
stats.mannwhitneyu(weds_chl_SEPAPR_MEI_POS, weds_chl_SEPAPR_MEI_NEG, nan_policy = 'omit') # ***
# SST
weds_sst_SEPAPR_MEI_POS = weds_sst_SEPAPR[mei_SEPAPR > 0.5]
weds_sst_SEPAPR_MEI_NEG = weds_sst_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(weds_sst_SEPAPR_MEI_POS))
print(np.nanmean(weds_sst_SEPAPR_MEI_NEG))
stats.mannwhitneyu(weds_sst_SEPAPR_MEI_POS, weds_sst_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Sea Ice
weds_seaice_SEPAPR_MEI_POS = weds_seaice_SEPAPR[mei_SEPAPR > 0.5]
weds_seaice_SEPAPR_MEI_NEG = weds_seaice_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(weds_seaice_SEPAPR_MEI_POS))
print(np.nanmean(weds_seaice_SEPAPR_MEI_NEG))
stats.mannwhitneyu(weds_seaice_SEPAPR_MEI_POS, weds_seaice_SEPAPR_MEI_NEG, nan_policy = 'omit') #
# PAR
weds_par_SEPAPR_MEI_POS = weds_par_SEPAPR[mei_SEPAPR > 0.5]
weds_par_SEPAPR_MEI_NEG = weds_par_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(weds_par_SEPAPR_MEI_POS))
print(np.nanmean(weds_par_SEPAPR_MEI_NEG))
stats.mannwhitneyu(weds_par_SEPAPR_MEI_POS, weds_par_SEPAPR_MEI_NEG, nan_policy = 'omit') # **
# Winds
weds_windspeed_SEPAPR_MEI_POS = weds_windspeed_SEPAPR[mei_SEPAPR > 0.5]
weds_windspeed_SEPAPR_MEI_NEG = weds_windspeed_SEPAPR[mei_SEPAPR < -0.5]
print(np.nanmean(weds_windspeed_SEPAPR_MEI_POS))
print(np.nanmean(weds_windspeed_SEPAPR_MEI_NEG))
stats.mannwhitneyu(weds_windspeed_SEPAPR_MEI_POS, weds_windspeed_SEPAPR_MEI_NEG, nan_policy = 'omit') # **








#%%
