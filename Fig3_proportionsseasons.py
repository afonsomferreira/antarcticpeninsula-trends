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
#%% Separar para o cluster 2 (GES)
ges_cluster = chl[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
#%% Separar para o cluster 3 (DRA)
dra_cluster = chl[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
#%% Separar para o cluster 4 (BRS)
brs_cluster = chl[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
#%% 2001-2010
## Spring
ges_cluster_spring20012010 = ges_cluster[(time_date_years >=2001) & (time_date_years <2011)
            &  ((time_date_months == 9) |  (time_date_months == 10)
            |  (time_date_months == 11))]
brs_cluster_spring20012010 = brs_cluster[(time_date_years >=2001) & (time_date_years <2011)
            &  ((time_date_months == 9) |  (time_date_months == 10)
            |  (time_date_months == 11))]
dra_cluster_spring20012010 = dra_cluster[(time_date_years >=2001) & (time_date_years <2011)
            &  ((time_date_months == 9) |  (time_date_months == 10)
            |  (time_date_months == 11))]
all_cluster_spring20012010 = np.vstack((ges_cluster_spring20012010, brs_cluster_spring20012010, dra_cluster_spring20012010))
all_cluster_spring20012010_mean = np.nanmean(all_cluster_spring20012010, axis=0)
all_cluster_spring20012010_mean = np.nanmean(all_cluster_spring20012010_mean)
## Summer
ges_cluster_summer20012010 = ges_cluster[(time_date_years >=2001) & (time_date_years <2011)
            &  ((time_date_months == 12) |  (time_date_months == 1)
            |  (time_date_months == 2))]
brs_cluster_summer20012010 = brs_cluster[(time_date_years >=2001) & (time_date_years <2011)
            &  ((time_date_months == 12) |  (time_date_months == 1)
            |  (time_date_months == 2))]
dra_cluster_summer20012010 = dra_cluster[(time_date_years >=2001) & (time_date_years <2011)
            &  ((time_date_months == 12) |  (time_date_months == 1)
            |  (time_date_months == 2))]
all_cluster_summer20012010 = np.vstack((ges_cluster_summer20012010, brs_cluster_summer20012010, dra_cluster_summer20012010))
all_cluster_summer20012010_mean = np.nanmean(all_cluster_summer20012010, axis=0)
all_cluster_summer20012010_mean = np.nanmean(all_cluster_summer20012010_mean)
## Autumn
ges_cluster_autumn20012010 = ges_cluster[(time_date_years >=2001) & (time_date_years <2011)
            &  ((time_date_months == 3) |  (time_date_months == 4))]
brs_cluster_autumn20012010 = brs_cluster[(time_date_years >=2001) & (time_date_years <2011)
            &  ((time_date_months == 3) |  (time_date_months == 4))]
dra_cluster_autumn20012010 = dra_cluster[(time_date_years >=2001) & (time_date_years <2011)
            &  ((time_date_months == 3) |  (time_date_months == 4))]
all_cluster_autumn20012010 = np.vstack((ges_cluster_autumn20012010, brs_cluster_autumn20012010, dra_cluster_autumn20012010))
all_cluster_autumn20012010_mean = np.nanmean(all_cluster_autumn20012010, axis=0)
all_cluster_autumn20012010_mean = np.nanmean(all_cluster_autumn20012010_mean)
#%% 2011-2020
## Spring
ges_cluster_spring20112020 = ges_cluster[(time_date_years >=2011) & (time_date_years <2021)
            &  ((time_date_months == 9) |  (time_date_months == 10)
            |  (time_date_months == 11))]
brs_cluster_spring20112020 = brs_cluster[(time_date_years >=2011) & (time_date_years <2021)
            &  ((time_date_months == 9) |  (time_date_months == 10)
            |  (time_date_months == 11))]
dra_cluster_spring20112020 = dra_cluster[(time_date_years >=2011) & (time_date_years <2021)
            &  ((time_date_months == 9) |  (time_date_months == 10)
            |  (time_date_months == 11))]
all_cluster_spring20112020 = np.vstack((ges_cluster_spring20112020, brs_cluster_spring20112020, dra_cluster_spring20112020))
all_cluster_spring20112020_mean = np.nanmean(all_cluster_spring20112020, axis=0)
all_cluster_spring20112020_mean = np.nanmean(all_cluster_spring20112020_mean)
## Summer
ges_cluster_summer20112020 = ges_cluster[(time_date_years >=2011) & (time_date_years <2021)
            &  ((time_date_months == 12) |  (time_date_months == 1)
            |  (time_date_months == 2))]
brs_cluster_summer20112020 = brs_cluster[(time_date_years >=2011) & (time_date_years <2021)
            &  ((time_date_months == 12) |  (time_date_months == 1)
            |  (time_date_months == 2))]
dra_cluster_summer20112020 = dra_cluster[(time_date_years >=2011) & (time_date_years <2021)
            &  ((time_date_months == 12) |  (time_date_months == 1)
            |  (time_date_months == 2))]
all_cluster_summer20112020 = np.vstack((ges_cluster_summer20112020, brs_cluster_summer20112020, dra_cluster_summer20112020))
all_cluster_summer20112020_mean = np.nanmean(all_cluster_summer20112020, axis=0)
all_cluster_summer20112020_mean = np.nanmean(all_cluster_summer20112020_mean)
## Autumn
ges_cluster_autumn20112020 = ges_cluster[(time_date_years >=2011) & (time_date_years <2021)
            &  ((time_date_months == 3) |  (time_date_months == 4))]
brs_cluster_autumn20112020 = brs_cluster[(time_date_years >=2011) & (time_date_years <2021)
            &  ((time_date_months == 3) |  (time_date_months == 4))]
dra_cluster_autumn20112020 = dra_cluster[(time_date_years >=2011) & (time_date_years <2021)
            &  ((time_date_months == 3) |  (time_date_months == 4))]
all_cluster_autumn20112020 = np.vstack((ges_cluster_autumn20112020, brs_cluster_autumn20112020, dra_cluster_autumn20112020))
all_cluster_autumn20112020_mean = np.nanmean(all_cluster_autumn20112020, axis=0)
all_cluster_autumn20112020_mean = np.nanmean(all_cluster_autumn20112020_mean)
#%% Calculate average proportion for each period
# Total
total_20012010 = all_cluster_spring20012010_mean + all_cluster_summer20012010_mean + all_cluster_autumn20012010_mean
total_20112020 = all_cluster_spring20112020_mean + all_cluster_summer20112020_mean + all_cluster_autumn20112020_mean
# 2001-2010
all_cluster_spring20012010_proportion = (all_cluster_spring20012010_mean/total_20012010)*100
all_cluster_summer20012010_proportion = (all_cluster_summer20012010_mean/total_20012010)*100
all_cluster_autumn20012010_proportion = (all_cluster_autumn20012010_mean/total_20012010)*100
# 2011-2020
all_cluster_spring20112020_proportion = (all_cluster_spring20112020_mean/total_20112020)*100
all_cluster_summer20112020_proportion = (all_cluster_summer20112020_mean/total_20112020)*100
all_cluster_autumn20112020_proportion = (all_cluster_autumn20112020_mean/total_20112020)*100
























#%% Separate for each season
# Spring (September, October, November)
for i in np.arange(1998, 2023):
    # separate September
    ges_cluster_sep = ges_cluster[(time_date_years == i) & (time_date_months == 9)]
    dra_cluster_sep = dra_cluster[(time_date_years == i) & (time_date_months == 9)]
    brs_cluster_sep = brs_cluster[(time_date_years == i) & (time_date_months == 9)]
    # separate October
    ges_cluster_oct = ges_cluster[(time_date_years == i) & (time_date_months == 10)]
    dra_cluster_oct = dra_cluster[(time_date_years == i) & (time_date_months == 10)]
    brs_cluster_oct = brs_cluster[(time_date_years == i) & (time_date_months == 10)]    
    # separate November
    ges_cluster_nov = ges_cluster[(time_date_years == i) & (time_date_months == 11)]
    dra_cluster_nov = dra_cluster[(time_date_years == i) & (time_date_months == 11)]
    brs_cluster_nov = brs_cluster[(time_date_years == i) & (time_date_months == 11)]
    # join all
    ges_cluster_spring = np.hstack((ges_cluster_sep, ges_cluster_oct, ges_cluster_nov))
    dra_cluster_spring = np.hstack((dra_cluster_sep, dra_cluster_oct, dra_cluster_nov))
    brs_cluster_spring = np.hstack((brs_cluster_sep, brs_cluster_oct, brs_cluster_nov))
    all_custers_spring = np.vstack((ges_cluster_spring, dra_cluster_spring, brs_cluster_spring))
    # mean
    all_custers_spring_mean = np.nanmean(all_custers_spring, axis=0)
    
    
    
    
    
    # average year for all regions
    allregions_cluster_year = np.nanmean(np.vstack((ges_cluster_year, dra_cluster_year)))
























#%%