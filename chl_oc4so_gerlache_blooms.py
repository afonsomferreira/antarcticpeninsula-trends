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
#%% Separar para o cluster 4 (gerlache)
gerlache_cluster = chl[clusters == 2,:]
#gerlache_cluster = np.where(gerlache_cluster > np.nanmedian(gerlache_cluster)-np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
#gerlache_cluster = np.where(gerlache_cluster < np.nanmedian(gerlache_cluster)+np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)

#gerlache_cluster_1D = gerlache_cluster.ravel()
# Find percentile 90
#gerlache_cluster_p99 = np.nanpercentile(gerlache_cluster,99)
#gerlache_cluster_p99 = 5
#gerlache_cluster_p90 = np.nanpercentile(gerlache_cluster,90)

#gerlache_cluster = np.nanmean(gerlache_cluster,0)
gerlache_cluster_p99 = np.nanpercentile(gerlache_cluster,99)

#np.nanmedian(gerlache_cluster)
#np.nanmax(gerlache_cluster)
#np.nanmin(gerlache_cluster)
#np.nanstd(gerlache_cluster)*3
#gerlache_cluster = np.where(gerlache_cluster > np.nanmedian(gerlache_cluster)-np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
#gerlache_cluster = np.where(gerlache_cluster < np.nanmedian(gerlache_cluster)+np.nanstd(gerlache_cluster)*3, gerlache_cluster, np.nan)
#%%
# 1997-1998
idx_year_chl_end = np.argwhere((time_date_years == 1998) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_19971998_daily = gerlache_cluster[:, :idx_year_chl_end]
time_date19971998 = time_date[:idx_year_chl_end]
gerlache_cluster_19971998_daily = gerlache_cluster_19971998_daily.ravel()
gerlache_19971998_pixelsabovep99 = len(gerlache_cluster_19971998_daily[gerlache_cluster_19971998_daily > gerlache_cluster_p99])
# 1998-1999
idx_year_chl_start = np.argwhere((time_date_years == 1998) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 1999) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_19981999_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date19981999 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_19981999_daily = gerlache_cluster_19981999_daily.ravel()
gerlache_19981999_pixelsabovep99 = len(gerlache_cluster_19981999_daily[gerlache_cluster_19981999_daily > gerlache_cluster_p99])
# 1999-2000
idx_year_chl_start = np.argwhere((time_date_years == 1999) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2000) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_19992000_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date19992000 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_19992000_daily = gerlache_cluster_19992000_daily.ravel()
gerlache_19992000_pixelsabovep99 = len(gerlache_cluster_19992000_daily[gerlache_cluster_19992000_daily > gerlache_cluster_p99])
## 2000-2001
idx_year_chl_start = np.argwhere((time_date_years == 2000) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2001) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20002001_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20002001 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20002001_daily = gerlache_cluster_20002001_daily.ravel()
gerlache_20002001_pixelsabovep99 = len(gerlache_cluster_20002001_daily[gerlache_cluster_20002001_daily > gerlache_cluster_p99])
## 2001-2002
idx_year_chl_start = np.argwhere((time_date_years == 2001) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2002) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20012002_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20012002 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20012002_daily = gerlache_cluster_20012002_daily.ravel()
gerlache_20012002_pixelsabovep99 = len(gerlache_cluster_20012002_daily[gerlache_cluster_20012002_daily > gerlache_cluster_p99])
## 2002-2003
idx_year_chl_start = np.argwhere((time_date_years == 2002) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2003) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20022003_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20022003 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20022003_daily = gerlache_cluster_20022003_daily.ravel()
gerlache_20022003_pixelsabovep99 = len(gerlache_cluster_20022003_daily[gerlache_cluster_20022003_daily > gerlache_cluster_p99])
## 2003-2004
idx_year_chl_start = np.argwhere((time_date_years == 2003) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2004) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20032004_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20032004 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20032004_daily = gerlache_cluster_20032004_daily.ravel()
gerlache_20032004_pixelsabovep99 = len(gerlache_cluster_20032004_daily[gerlache_cluster_20032004_daily > gerlache_cluster_p99])
## 2004-2005
idx_year_chl_start = np.argwhere((time_date_years == 2004) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2005) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20042005_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20042005 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20042005_daily = gerlache_cluster_20042005_daily.ravel()
gerlache_20042005_pixelsabovep99 = len(gerlache_cluster_20042005_daily[gerlache_cluster_20042005_daily > gerlache_cluster_p99])
## 2005-2006
idx_year_chl_start = np.argwhere((time_date_years == 2005) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2006) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20052006_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20052006 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20052006_daily = gerlache_cluster_20052006_daily.ravel()
gerlache_20052006_pixelsabovep99 = len(gerlache_cluster_20052006_daily[gerlache_cluster_20052006_daily > gerlache_cluster_p99])
## 2006-2007
idx_year_chl_start = np.argwhere((time_date_years == 2006) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2007) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20062007_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20062007 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20062007_daily = gerlache_cluster_20062007_daily.ravel()
gerlache_20062007_pixelsabovep99 = len(gerlache_cluster_20062007_daily[gerlache_cluster_20062007_daily > gerlache_cluster_p99])
## 2007-2008
idx_year_chl_start = np.argwhere((time_date_years == 2007) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2008) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20072008_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20072008 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20072008_daily = gerlache_cluster_20072008_daily.ravel()
gerlache_20072008_pixelsabovep99 = len(gerlache_cluster_20072008_daily[gerlache_cluster_20072008_daily > gerlache_cluster_p99])
## 2008-2009
idx_year_chl_start = np.argwhere((time_date_years == 2008) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2009) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20082009_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20082009 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20082009_daily = gerlache_cluster_20082009_daily.ravel()
gerlache_20082009_pixelsabovep99 = len(gerlache_cluster_20082009_daily[gerlache_cluster_20082009_daily > gerlache_cluster_p99])
## 2009-2010
idx_year_chl_start = np.argwhere((time_date_years == 2009) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2010) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20092010_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20092010 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20092010_daily = gerlache_cluster_20092010_daily.ravel()
gerlache_20092010_pixelsabovep99 = len(gerlache_cluster_20092010_daily[gerlache_cluster_20092010_daily > gerlache_cluster_p99])
## 2010-2011
idx_year_chl_start = np.argwhere((time_date_years == 2010) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2011) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20102011_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20102011 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20102011_daily = gerlache_cluster_20102011_daily.ravel()
gerlache_20102011_pixelsabovep99 = len(gerlache_cluster_20102011_daily[gerlache_cluster_20102011_daily > gerlache_cluster_p99])
## 2011-2012
idx_year_chl_start = np.argwhere((time_date_years == 2011) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2012) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20112012_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20112012 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20112012_daily = gerlache_cluster_20112012_daily.ravel()
gerlache_20112012_pixelsabovep99 = len(gerlache_cluster_20112012_daily[gerlache_cluster_20112012_daily > gerlache_cluster_p99])
## 2012-2013
idx_year_chl_start = np.argwhere((time_date_years == 2012) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2013) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20122013_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20122013 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20122013_daily = gerlache_cluster_20122013_daily.ravel()
gerlache_20122013_pixelsabovep99 = len(gerlache_cluster_20122013_daily[gerlache_cluster_20122013_daily > gerlache_cluster_p99])
## 2013-2014
idx_year_chl_start = np.argwhere((time_date_years == 2013) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2014) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20132014_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20132014 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20132014_daily = gerlache_cluster_20132014_daily.ravel()
gerlache_20132014_pixelsabovep99 = len(gerlache_cluster_20132014_daily[gerlache_cluster_20132014_daily > gerlache_cluster_p99])
## 2014-2015
idx_year_chl_start = np.argwhere((time_date_years == 2014) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2015) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20142015_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20142015 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20142015_daily = gerlache_cluster_20142015_daily.ravel()
gerlache_20142015_pixelsabovep99 = len(gerlache_cluster_20142015_daily[gerlache_cluster_20142015_daily > gerlache_cluster_p99])
## 2015-2016
idx_year_chl_start = np.argwhere((time_date_years == 2015) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2016) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20152016_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20152016 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20152016_daily = gerlache_cluster_20152016_daily.ravel()
gerlache_20152016_pixelsabovep99 = len(gerlache_cluster_20152016_daily[gerlache_cluster_20152016_daily > gerlache_cluster_p99])
## 2016-2017
idx_year_chl_start = np.argwhere((time_date_years == 2016) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2017) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20162017_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20162017 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20162017_daily = gerlache_cluster_20162017_daily.ravel()
gerlache_20162017_pixelsabovep99 = len(gerlache_cluster_20162017_daily[gerlache_cluster_20162017_daily > gerlache_cluster_p99])
## 2017-2018
idx_year_chl_start = np.argwhere((time_date_years == 2017) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2018) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20172018_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20172018 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20172018_daily = gerlache_cluster_20172018_daily.ravel()
gerlache_20172018_pixelsabovep99 = len(gerlache_cluster_20172018_daily[gerlache_cluster_20172018_daily > gerlache_cluster_p99])
## 2018-2019
idx_year_chl_start = np.argwhere((time_date_years == 2018) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2019) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20182019_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20182019 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20182019_daily = gerlache_cluster_20182019_daily.ravel()
gerlache_20182019_pixelsabovep99 = len(gerlache_cluster_20182019_daily[gerlache_cluster_20182019_daily > gerlache_cluster_p99])
## 2019-2020
idx_year_chl_start = np.argwhere((time_date_years == 2019) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2020) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20192020_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20192020 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20192020_daily = gerlache_cluster_20192020_daily.ravel()
gerlache_20192020_pixelsabovep99 = len(gerlache_cluster_20192020_daily[gerlache_cluster_20192020_daily > gerlache_cluster_p99])
## 2020-2021
idx_year_chl_start = np.argwhere((time_date_years == 2020) & (time_date_months == 7)).ravel()[0]
idx_year_chl_end = np.argwhere((time_date_years == 2021) & (time_date_months == 6)).ravel()[-1]+1
gerlache_cluster_20202021_daily = gerlache_cluster[:, idx_year_chl_start:idx_year_chl_end]
time_date20202021 = time_date[idx_year_chl_start:idx_year_chl_end]
gerlache_cluster_20202021_daily = gerlache_cluster_20202021_daily.ravel()
gerlache_20202021_pixelsabovep99 = len(gerlache_cluster_20202021_daily[gerlache_cluster_20202021_daily > gerlache_cluster_p99])
# Join all
gerlache_cluster_pixelsabovep99 = np.hstack((gerlache_19971998_pixelsabovep99,
                                              gerlache_19981999_pixelsabovep99,
                                              gerlache_19992000_pixelsabovep99,
                                              gerlache_20002001_pixelsabovep99,
                                              gerlache_20012002_pixelsabovep99,
                                              gerlache_20022003_pixelsabovep99,
                                              gerlache_20032004_pixelsabovep99,
                                              gerlache_20042005_pixelsabovep99,
                                              gerlache_20052006_pixelsabovep99,
                                              gerlache_20062007_pixelsabovep99,
                                              gerlache_20072008_pixelsabovep99,
                                              gerlache_20082009_pixelsabovep99,
                                              gerlache_20092010_pixelsabovep99,
                                              gerlache_20102011_pixelsabovep99,
                                              gerlache_20112012_pixelsabovep99,
                                              gerlache_20122013_pixelsabovep99,
                                              gerlache_20132014_pixelsabovep99,
                                              gerlache_20142015_pixelsabovep99,
                                              gerlache_20152016_pixelsabovep99,
                                              gerlache_20162017_pixelsabovep99,
                                              gerlache_20172018_pixelsabovep99,
                                              gerlache_20182019_pixelsabovep99,
                                              gerlache_20192020_pixelsabovep99,
                                              gerlache_20202021_pixelsabovep99,
                                              ))
#%%
plt.scatter(np.arange(1998, 2022), gerlache_cluster_pixelsabovep99)
#%%
for i in np.arange(1998,2022):
    for j in np.arange(1, 13):
        gerlache_temp = gerlache_cluster[:,(time_date_years == i) & (time_date_months == j)]
        gerlache_blooms_temp = len(gerlache_temp[gerlache_temp > 5])
        if j == 1:
            gerlache_blooms_year = gerlache_blooms_temp
        else:
            gerlache_blooms_year = np.hstack((gerlache_blooms_year, gerlache_blooms_temp))
    if i == 1998:
        gerlache_blooms_year_all = gerlache_blooms_year
    else:
        gerlache_blooms_year_all = np.vstack((gerlache_blooms_year_all, gerlache_blooms_year))
  

#%%
plt.scatter(np.arange(1998, 2022),gerlache_blooms_year_all[:,0], label='JAN')
plt.scatter(np.arange(1998, 2022),gerlache_blooms_year_all[:,1], label='FEB')
plt.scatter(np.arange(1998, 2022),gerlache_blooms_year_all[:,2], label='MAR')
plt.scatter(np.arange(1998, 2022),gerlache_blooms_year_all[:,10], label='NOV')
plt.scatter(np.arange(1998, 2022),gerlache_blooms_year_all[:,11], label='DEC')
plt.legend()

#%%
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
plt.plot(xnew, f_cubic(xnew), color = '#9800cb', linewidth = 5, label='1997-2021')

#plt.plot(np.arange(1,13),gerlache_cluster_mean19972021, color = [241/256, 180/256, 47/256, 1], linewidth = 4, label='1997-2021')
plt.plot(np.arange(1,13),gerlache_cluster_19982005, color = 'k', linewidth = 2, linestyle='--', label='1997-2005')
plt.plot(np.arange(1,13),gerlache_cluster_20062014, color = 'k', linewidth = 2, linestyle=':', label='2005-2014')
plt.plot(np.arange(1,13),gerlache_cluster_20152021, color = 'k', linewidth = 2, linestyle='-.', label='2015-2021')
plt.errorbar(np.arange(1,13),gerlache_cluster_mean19972021, gerlache_yearlycicles_std, linestyle='None', marker='None',
             color = '#9800cb', alpha=0.5, capsize=10, elinewidth=1, markeredgewidth=1)
#plt.fill_between(np.arange(1,13), gerlache_cluster_mean19972021, gerlache_cluster_mean19972021+gerlache_yearlycicles_std, color ='#9800cb', alpha=.2, edgecolor = None)
#plt.fill_between(np.arange(1,13), gerlache_cluster_mean19972021, gerlache_cluster_mean19972021-gerlache_yearlycicles_std, color ='#9800cb', alpha=.2, edgecolor = None)
plt.xticks([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
           ['Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr','May', 'Jun'], fontsize=12)
plt.xlim(3,10)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
plt.legend(fontsize=12, loc=2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\gerlache_meancycle_errorbar.png'
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
## gerlache
# Series for each month
gerlache_cluster_sep = gerlache_cluster_allcicles[:,2] #P-VAL <0.1
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_sep)], gerlache_cluster_sep[~np.isnan(gerlache_cluster_sep)])
gerlache_cluster_oct = gerlache_cluster_allcicles[:,3]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_oct)], gerlache_cluster_oct[~np.isnan(gerlache_cluster_oct)])
gerlache_cluster_nov = gerlache_cluster_allcicles[:,4]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_nov)], gerlache_cluster_nov[~np.isnan(gerlache_cluster_nov)])
gerlache_cluster_dec = gerlache_cluster_allcicles[:,5] #LIMIAR 0.57 
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_dec)], gerlache_cluster_dec[~np.isnan(gerlache_cluster_dec)])
gerlache_cluster_jan = gerlache_cluster_allcicles[:,6]
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_jan)], gerlache_cluster_jan[~np.isnan(gerlache_cluster_jan)])
gerlache_cluster_feb = gerlache_cluster_allcicles[:,7] # P-VAL <0.1
slope, _, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_feb)], gerlache_cluster_feb[~np.isnan(gerlache_cluster_feb)])
gerlache_cluster_mar = gerlache_cluster_allcicles[:,8] # SIGNIFICATIVO
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_mar)], gerlache_cluster_mar[~np.isnan(gerlache_cluster_mar)])
gerlache_cluster_apr = gerlache_cluster_allcicles[:,9] #SIGNIFICATIVO
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(gerlache_cluster_apr)], gerlache_cluster_apr[~np.isnan(gerlache_cluster_apr)])
#plt.scatter(np.arange(1998, 2022), gerlache_cluster_summermean_allyears)
# Series for each Spring-Summer # SIGNIFICATIVO
gerlache_cluster_mean_allyears=np.nanmean(gerlache_cluster_allcicles,1)
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), gerlache_cluster_mean_allyears)
# Series for just Nov-Mar # SIGNIFICATIVO
gerlache_cluster_summermean_allyears=np.nanmean(gerlache_cluster_allcicles[:,4:9],1)
slope, intercept, rvalue, pvalue, _ = stats.linregress(np.arange(1998, 2022), gerlache_cluster_summermean_allyears)
plt.scatter(np.arange(1998, 2022), gerlache_cluster_summermean_allyears)
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

#%% gerlache Plot comparison between 1997-2005, 2006-2014, 2015-2021
# Sep
gerlache_cluster_sep_19982005 = gerlache_cluster_allcicles[:8,2]
gerlache_cluster_sep_20062014 = gerlache_cluster_allcicles[8:16,2]
gerlache_cluster_sep_20152021 = gerlache_cluster_allcicles[16:,2]
plt.boxplot([gerlache_cluster_sep_19982005, gerlache_cluster_sep_20062014,
             gerlache_cluster_sep_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_sep_19982005, gerlache_cluster_sep_20062014, gerlache_cluster_sep_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_sep_19982005, gerlache_cluster_sep_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_sep_19982005, gerlache_cluster_sep_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_sep_20062014, gerlache_cluster_sep_20152021, nan_policy='omit')
# Oct
gerlache_cluster_oct_19982005 = gerlache_cluster_allcicles[:8,3]
gerlache_cluster_oct_20062014 = gerlache_cluster_allcicles[8:16,3]
gerlache_cluster_oct_20152021 = gerlache_cluster_allcicles[16:,3]
plt.boxplot([gerlache_cluster_oct_19982005, gerlache_cluster_oct_20062014,
             gerlache_cluster_oct_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_oct_19982005, gerlache_cluster_oct_20062014, gerlache_cluster_oct_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_oct_19982005, gerlache_cluster_oct_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_oct_19982005, gerlache_cluster_oct_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_oct_20062014, gerlache_cluster_oct_20152021, nan_policy='omit')
# Nov
gerlache_cluster_nov_19982005 = gerlache_cluster_allcicles[:8,4]
gerlache_cluster_nov_20062014 = gerlache_cluster_allcicles[8:16,4]
gerlache_cluster_nov_20152021 = gerlache_cluster_allcicles[16:,4]
plt.boxplot([gerlache_cluster_nov_19982005, gerlache_cluster_nov_20062014,
             gerlache_cluster_nov_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_nov_19982005, gerlache_cluster_nov_20062014, gerlache_cluster_nov_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_nov_19982005, gerlache_cluster_nov_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_nov_19982005, gerlache_cluster_nov_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_nov_20062014, gerlache_cluster_nov_20152021, nan_policy='omit')
# Dec
gerlache_cluster_dec_19982005 = gerlache_cluster_allcicles[:8,5]
gerlache_cluster_dec_20062014 = gerlache_cluster_allcicles[8:16,5]
gerlache_cluster_dec_20152021 = gerlache_cluster_allcicles[16:,5]
plt.boxplot([gerlache_cluster_dec_19982005, gerlache_cluster_dec_20062014,
             gerlache_cluster_dec_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_dec_19982005, gerlache_cluster_dec_20062014, gerlache_cluster_dec_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_dec_19982005, gerlache_cluster_dec_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_dec_19982005, gerlache_cluster_dec_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_dec_20062014, gerlache_cluster_dec_20152021, nan_policy='omit')
# Jan
gerlache_cluster_jan_19982005 = gerlache_cluster_allcicles[:8,6]
gerlache_cluster_jan_20062014 = gerlache_cluster_allcicles[8:16,6]
gerlache_cluster_jan_20152021 = gerlache_cluster_allcicles[16:,6]
plt.boxplot([gerlache_cluster_jan_19982005, gerlache_cluster_jan_20062014,
             gerlache_cluster_jan_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_jan_19982005, gerlache_cluster_jan_20062014, gerlache_cluster_jan_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_jan_19982005, gerlache_cluster_jan_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_jan_19982005, gerlache_cluster_jan_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_jan_20062014, gerlache_cluster_jan_20152021, nan_policy='omit')
# Feb
gerlache_cluster_feb_19982005 = gerlache_cluster_allcicles[:8,7]
gerlache_cluster_feb_20062014 = gerlache_cluster_allcicles[8:16,7]
gerlache_cluster_feb_20062014 = gerlache_cluster_feb_20062014[~np.isnan(gerlache_cluster_feb_20062014)]
gerlache_cluster_feb_20152021 = gerlache_cluster_allcicles[16:,7]
plt.boxplot([gerlache_cluster_feb_19982005, gerlache_cluster_feb_20062014,
             gerlache_cluster_feb_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_feb_19982005, gerlache_cluster_feb_20062014, gerlache_cluster_feb_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_feb_19982005, gerlache_cluster_feb_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_feb_19982005, gerlache_cluster_feb_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_feb_20062014, gerlache_cluster_feb_20152021, nan_policy='omit')
# Mar
gerlache_cluster_mar_19982005 = gerlache_cluster_allcicles[:8,8]
gerlache_cluster_mar_20062014 = gerlache_cluster_allcicles[8:16,8]
gerlache_cluster_mar_20062014 = gerlache_cluster_mar_20062014[~np.isnan(gerlache_cluster_mar_20062014)]
gerlache_cluster_mar_20152021 = gerlache_cluster_allcicles[16:,8]
plt.boxplot([gerlache_cluster_mar_19982005, gerlache_cluster_mar_20062014,
             gerlache_cluster_mar_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_mar_19982005, gerlache_cluster_mar_20062014, gerlache_cluster_mar_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_mar_19982005, gerlache_cluster_mar_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_mar_19982005, gerlache_cluster_mar_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_mar_20062014, gerlache_cluster_mar_20152021, nan_policy='omit')
# Apr
gerlache_cluster_apr_19982005 = gerlache_cluster_allcicles[:8,9]
gerlache_cluster_apr_20062014 = gerlache_cluster_allcicles[8:16,9]
#gerlache_cluster_apr_20062014 = gerlache_cluster_apr_20062014[~np.isnan(gerlache_cluster_apr_20062014)]
gerlache_cluster_apr_20152021 = gerlache_cluster_allcicles[16:,9]
plt.boxplot([gerlache_cluster_apr_19982005, gerlache_cluster_apr_20062014,
             gerlache_cluster_apr_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_apr_19982005, gerlache_cluster_apr_20062014, gerlache_cluster_apr_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_apr_19982005, gerlache_cluster_apr_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_apr_19982005, gerlache_cluster_apr_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_apr_20062014, gerlache_cluster_apr_20152021, nan_policy='omit')
# Spring Summer
gerlache_cluster_springsummer_19982005 = np.nanmean(gerlache_cluster_allcicles[:8,2:10],1)
gerlache_cluster_springsummer_20062014 = np.nanmean(gerlache_cluster_allcicles[8:16,2:10],1)
#gerlache_cluster_apr_20062014 = gerlache_cluster_apr_20062014[~np.isnan(gerlache_cluster_apr_20062014)]
gerlache_cluster_springsummer_20152021 = np.nanmean(gerlache_cluster_allcicles[16:,2:10],1)
plt.boxplot([gerlache_cluster_springsummer_19982005, gerlache_cluster_springsummer_20062014,
             gerlache_cluster_springsummer_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_springsummer_19982005, gerlache_cluster_springsummer_20062014, gerlache_cluster_springsummer_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_springsummer_19982005, gerlache_cluster_springsummer_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_springsummer_19982005, gerlache_cluster_springsummer_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_springsummer_20062014, gerlache_cluster_springsummer_20152021, nan_policy='omit')
# Nov-Mar
gerlache_cluster_novmar_19982005 = np.nanmean(gerlache_cluster_allcicles[:8,4:9],1)
gerlache_cluster_novmar_20062014 = np.nanmean(gerlache_cluster_allcicles[8:16,4:9],1)
#gerlache_cluster_apr_20062014 = gerlache_cluster_apr_20062014[~np.isnan(gerlache_cluster_apr_20062014)]
gerlache_cluster_novmar_20152021 = np.nanmean(gerlache_cluster_allcicles[16:,4:9],1)
plt.boxplot([gerlache_cluster_novmar_19982005, gerlache_cluster_novmar_20062014,
             gerlache_cluster_novmar_20152021], labels=['1998-2005', '2006-2014', '2015-2021'])
stats.kruskal(gerlache_cluster_novmar_19982005, gerlache_cluster_novmar_20062014, gerlache_cluster_novmar_20152021,
              nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_novmar_19982005, gerlache_cluster_novmar_20062014, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_novmar_19982005, gerlache_cluster_novmar_20152021, nan_policy='omit')
stats.mannwhitneyu(gerlache_cluster_novmar_20062014, gerlache_cluster_novmar_20152021, nan_policy='omit')
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
#%% gerlache Cycle comparison between 1997-2005, 2006-2014, 2015-2021
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