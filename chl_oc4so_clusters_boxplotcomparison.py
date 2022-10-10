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
import matplotlib.patches as mpatches
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

# Test outliers for weddell_cluster
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

#weddell_cluster_19982005 = np.nanmean(weddel_cluster_allcicles[:8,:], axis=0)
#weddell_cluster_20062014 = np.nanmean(weddel_cluster_allcicles[8:16,:], axis=0)
#weddell_cluster_20152021 = np.nanmean(weddel_cluster_allcicles[16:,:], axis=0)
# Separate by 8 years
weddell_cluster_19982005_data = weddel_cluster_allcicles[:8,2:10]
weddell_cluster_19982005_data = weddell_cluster_19982005_data[~np.isnan(weddell_cluster_19982005_data)]
weddell_cluster_19982005_means = np.nanmean(weddel_cluster_allcicles[:8,2:10], axis=0)
weddell_cluster_20062014_data = weddel_cluster_allcicles[8:16,2:10]
weddell_cluster_20062014_data = weddell_cluster_20062014_data[~np.isnan(weddell_cluster_20062014_data)]
weddell_cluster_20062014_means = np.nanmean(weddel_cluster_allcicles[8:16,2:10], axis=0)
weddell_cluster_20152021_data = weddel_cluster_allcicles[16:,2:10]
weddell_cluster_20152021_data = weddell_cluster_20152021_data[~np.isnan(weddell_cluster_20152021_data)]
weddell_cluster_20152021_means = np.nanmean(weddel_cluster_allcicles[16:,2:10], axis=0)

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

gerlache_cluster_19982005_data = gerlache_cluster_allcicles[:8,2:10]
gerlache_cluster_19982005_data = gerlache_cluster_19982005_data[~np.isnan(gerlache_cluster_19982005_data)]
gerlache_cluster_19982005_means = np.nanmean(gerlache_cluster_allcicles[:8,2:10], axis=0)
gerlache_cluster_20062014_data = gerlache_cluster_allcicles[8:16,2:10]
gerlache_cluster_20062014_data = gerlache_cluster_20062014_data[~np.isnan(gerlache_cluster_20062014_data)]
gerlache_cluster_20062014_means = np.nanmean(gerlache_cluster_allcicles[8:16,2:10], axis=0)
gerlache_cluster_20152021_data = gerlache_cluster_allcicles[16:,2:10]
gerlache_cluster_20152021_data = gerlache_cluster_20152021_data[~np.isnan(gerlache_cluster_20152021_data)]
gerlache_cluster_20152021_means = np.nanmean(gerlache_cluster_allcicles[16:,2:10], axis=0)
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

oceanic_cluster_19982005_data = oceanic_cluster_allcicles[:8,2:10]
oceanic_cluster_19982005_data = oceanic_cluster_19982005_data[~np.isnan(oceanic_cluster_19982005_data)]
oceanic_cluster_19982005_means = np.nanmean(oceanic_cluster_allcicles[:8,2:10], axis=0)
oceanic_cluster_20062014_data = oceanic_cluster_allcicles[8:16,2:10]
oceanic_cluster_20062014_data = oceanic_cluster_20062014_data[~np.isnan(oceanic_cluster_20062014_data)]
oceanic_cluster_20062014_means = np.nanmean(oceanic_cluster_allcicles[8:16,2:10], axis=0)
oceanic_cluster_20152021_data = oceanic_cluster_allcicles[16:,2:10]
oceanic_cluster_20152021_data = oceanic_cluster_20152021_data[~np.isnan(oceanic_cluster_20152021_data)]
oceanic_cluster_20152021_means = np.nanmean(oceanic_cluster_allcicles[16:,2:10], axis=0)
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

bransfield_cluster_19982005_data = bransfield_cluster_allcicles[:8,2:10]
bransfield_cluster_19982005_data = bransfield_cluster_19982005_data[~np.isnan(bransfield_cluster_19982005_data)]
bransfield_cluster_19982005_means = np.nanmean(bransfield_cluster_allcicles[:8,2:10], axis=0)
bransfield_cluster_20062014_data = bransfield_cluster_allcicles[8:16,2:10]
bransfield_cluster_20062014_data = bransfield_cluster_20062014_data[~np.isnan(bransfield_cluster_20062014_data)]
bransfield_cluster_20062014_means = np.nanmean(bransfield_cluster_allcicles[8:16,2:10], axis=0)
bransfield_cluster_20152021_data = bransfield_cluster_allcicles[16:,2:10]
bransfield_cluster_20152021_data = bransfield_cluster_20152021_data[~np.isnan(bransfield_cluster_20152021_data)]
bransfield_cluster_20152021_means = np.nanmean(bransfield_cluster_allcicles[16:,2:10], axis=0)
#%% Boxplot comparison
class AnyObject(object):
    pass

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
                                   edgecolor='black', hatch='xx', lw=3,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch
c_weddell = [43/256, 131/256, 186/256, 1]
c_gerlache = [215/256, 25/256, 28/256, 1]
c_bransfield = '#9800cb'
c_oceanic = '#d09c26'
fig = plt.figure()
ax = plt.axes()
#plt.hold(True)
# first boxplot pair
bp = plt.boxplot([weddell_cluster_19982005_means, weddell_cluster_20062014_means,
             weddell_cluster_20152021_means], positions = [1, 2, 3], widths = 0.7, 
                 patch_artist=True, boxprops=dict(facecolor=c_weddell, color='k'),
                 capprops=dict(color='k'), whiskerprops=dict(color='k'),
                 flierprops=dict(color=c_weddell, markeredgecolor=c_weddell),
                 medianprops=dict(color='k', linewidth=2))
bp['boxes'][0].set(hatch = '///')
#bp['boxes'][1].set(hatch = '---')
bp['boxes'][2].set(hatch = '...')
bp = plt.boxplot([gerlache_cluster_19982005_means, gerlache_cluster_20062014_means,
             gerlache_cluster_20152021_means], positions = [5, 6, 7], widths = 0.7, 
                 patch_artist=True, boxprops=dict(facecolor=c_gerlache, color='k'),
                 capprops=dict(color='k'), whiskerprops=dict(color='k'),
                 flierprops=dict(color=c_gerlache, markeredgecolor=c_gerlache),
                 medianprops=dict(color='k'))
bp['boxes'][0].set(hatch = '///')
#bp['boxes'][1].set(hatch = '---')
bp['boxes'][2].set(hatch = '...')
bp = plt.boxplot([bransfield_cluster_19982005_means, bransfield_cluster_20062014_means,
             bransfield_cluster_20152021_means], positions = [9, 10, 11], widths = 0.7, 
                 patch_artist=True, boxprops=dict(facecolor=c_bransfield, color='k'),
                 capprops=dict(color='k'), whiskerprops=dict(color='k'),
                 flierprops=dict(color=c_bransfield, markeredgecolor=c_bransfield),
                 medianprops=dict(color='k'))
bp['boxes'][0].set(hatch = '///')
#bp['boxes'][1].set(hatch = '---')
bp['boxes'][2].set(hatch = '...')
bp = plt.boxplot([oceanic_cluster_19982005_means, oceanic_cluster_20062014_means,
             oceanic_cluster_20152021_means], positions = [13, 14, 15], widths = 0.7, 
                 patch_artist=True, boxprops=dict(facecolor=c_oceanic, color='k'),
                 capprops=dict(color='k'), whiskerprops=dict(color='k'),
                 flierprops=dict(color=c_oceanic, markeredgecolor=c_oceanic),
                 medianprops=dict(color='k'))
bp['boxes'][0].set(hatch = '///')
#bp['boxes'][1].set(hatch = '---')
bp['boxes'][2].set(hatch = '...')
plt.xlim(0,16)
plt.ylim(0,9)
ax.set_xticks([2, 6, 10, 14])
ax.set_xticklabels(['Weddell', 'Gerlache', 'Bransfield', 'Oceanic'], fontsize=14)
plt.yticks(fontsize=12)
plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
legend1 = mpatches.Patch(facecolor='w',alpha=1,hatch=r'///',label='1998-2005', edgecolor='k')
legend2= mpatches.Patch(facecolor='w',alpha=1,label='2006-2014', edgecolor='k')
legend3 = mpatches.Patch(facecolor='w',alpha=1,hatch='...',label='2015-2021', edgecolor='k')
ax.legend(handles = [legend1,legend2,legend3],loc=1, fontsize=14, handlelength=3)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\areas_comparison.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()