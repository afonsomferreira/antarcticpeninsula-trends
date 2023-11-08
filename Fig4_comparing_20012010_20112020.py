# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:33:06 2020

@author: Afonso
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.path import Path
from tqdm import tqdm
from scipy import integrate
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.patches as mpatches
import shap
import sklearn
from matplotlib import patches
import math
from netCDF4 import Dataset
import seaborn as sns
import cmocean
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
def serial_date_to_string_mld(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1950, 1, 1, 0, 0) + datetime.timedelta(hours=srl_no)
    return new_date
def serial_date_to_string_ssh(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1950, 1, 1, 0, 0) + datetime.timedelta(days=srl_no)
    return new_date
def serial_date_to_string_ugos(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1950, 1, 1, 0, 0) + datetime.timedelta(hours=srl_no)
    return new_date
def serial_date_to_string_winds(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1900, 1, 1, 0, 0) + datetime.timedelta(hours=srl_no)
    return new_date
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy
def dropcol_importances(rf, X_train, y_train):
    rf_ = clone(rf)
    rf_.random_state = 23
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 23
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I
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
#%% Load data 1998-2022 PAR
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\par\\')
### Load data 1998-2020
fh = np.load('par_19972021_new.npz', allow_pickle=True)
par = fh['par']
par = par.astype(np.float64)
#par = par * 100
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Load upscaled 4km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_par.npz',allow_pickle = True)
clusters = fh['clusters']
#%% Load SST
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
### Load data 1998-2020
fh = np.load('sst-seaice_19972021.npz', allow_pickle=True)
sst = fh['sst']
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
#%% Load Sea Ice
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
### Load data 1998-2020
fh = np.load('sst-seaice_19972021.npz', allow_pickle=True)
seaice = fh['seaice']
seaice = seaice * 100
time_date_seaice = fh['time_date']
time_date_years_seaice = np.empty_like(time_date_seaice)
time_date_months_seaice = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years_seaice[i] = time_date_seaice[i].year
    time_date_months_seaice[i] = time_date_seaice[i].month
#%% Load Winds
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\winds\\')
### Load data 1998-2020
fh = np.load('winds_19972022_era5.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
wind_u = fh['wind_u']
wind_v = fh['wind_v']
windspeed = np.sqrt(wind_u**2 + wind_u**2)
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
#%% PAR Separar para cada cluster
#WEDS
weds_cluster = par[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        weds_spring_par20012010 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        weds_summer_par20012010 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        weds_autumn_par20012010 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        weds_sepapr_par20012010 = np.nanmean(np.hstack((weds_spring_par20012010, weds_summer_par20012010, weds_autumn_par20012010)))
    else:
        weds_spring_par20012010 = np.hstack((weds_spring_par20012010, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        weds_summer_par20012010 = np.hstack((weds_summer_par20012010, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        weds_autumn_par20012010 = np.hstack((weds_autumn_par20012010, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        weds_sepapr_par20012010 = np.hstack((weds_sepapr_par20012010, np.nanmean(np.hstack((weds_spring_par20012010, weds_summer_par20012010, weds_autumn_par20012010)))))

# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        weds_spring_par20112020 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        weds_summer_par20112020 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        weds_autumn_par20112020 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        weds_sepapr_par20112020 = np.nanmean(np.hstack((weds_spring_par20112020, weds_summer_par20112020, weds_autumn_par20112020)))
    else:
        weds_spring_par20112020 = np.hstack((weds_spring_par20112020, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        weds_summer_par20112020 = np.hstack((weds_summer_par20112020, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        weds_autumn_par20112020 = np.hstack((weds_autumn_par20112020, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        weds_sepapr_par20112020 = np.hstack((weds_sepapr_par20112020, np.nanmean(np.hstack((weds_spring_par20112020, weds_summer_par20112020, weds_autumn_par20112020)))))
#GES
weds_cluster = par[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        weds_spring_par20012010 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        weds_summer_par20012010 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        weds_autumn_par20012010 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        weds_sepapr_par20012010 = np.nanmean(np.hstack((weds_spring_par20012010, weds_summer_par20012010, weds_autumn_par20012010)))
    else:
        weds_spring_par20012010 = np.hstack((weds_spring_par20012010, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        weds_summer_par20012010 = np.hstack((weds_summer_par20012010, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        weds_autumn_par20012010 = np.hstack((weds_autumn_par20012010, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        weds_sepapr_par20012010 = np.hstack((weds_sepapr_par20012010, np.nanmean(np.hstack((weds_spring_par20012010, weds_summer_par20012010, weds_autumn_par20012010)))))

# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        weds_spring_par20112020 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        weds_summer_par20112020 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        weds_autumn_par20112020 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        weds_sepapr_par20112020 = np.nanmean(np.hstack((weds_spring_par20112020, weds_summer_par20112020, weds_autumn_par20112020)))
    else:
        weds_spring_par20112020 = np.hstack((weds_spring_par20112020, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        weds_summer_par20112020 = np.hstack((weds_summer_par20112020, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        weds_autumn_par20112020 = np.hstack((weds_autumn_par20112020, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        weds_sepapr_par20112020 = np.hstack((weds_sepapr_par20112020, np.nanmean(np.hstack((weds_spring_par20112020, weds_summer_par20112020, weds_autumn_par20112020)))))


































































#%% Separar para o cluster 2 (GES)
ges_cluster = par[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
np.nanmedian(ges_cluster)
np.nanmax(ges_cluster)
np.nanmin(ges_cluster)
np.nanstd(ges_cluster)*3
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        ges_spring_par20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_par20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_par20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        ges_sepapr_par20012010 = np.hstack((ges_spring_par20012010, ges_summer_par20012010, ges_autumn_par20012010))
    else:
        ges_spring_par20012010 = np.hstack((ges_spring_par20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_par20012010 = np.hstack((ges_summer_par20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_par20012010 = np.hstack((ges_autumn_par20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        ges_sepapr_par20012010 = np.hstack((ges_sepapr_par20012010, np.hstack((ges_spring_par20012010, ges_summer_par20012010, ges_autumn_par20012010))))

# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        ges_spring_par20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_par20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_par20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        ges_sepapr_par20112020 = np.hstack((ges_spring_par20112020, ges_summer_par20112020, ges_autumn_par20112020))
    else:
        ges_spring_par20112020 = np.hstack((ges_spring_par20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_par20112020 = np.hstack((ges_summer_par20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_par20112020 = np.hstack((ges_autumn_par20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        ges_sepapr_par20112020 = np.hstack((ges_sepapr_par20112020, np.hstack((ges_spring_par20112020, ges_summer_par20112020, ges_autumn_par20112020))))
#%% Separar para o cluster 3 (DRA)
dra_cluster = par[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
np.nanmedian(dra_cluster)
np.nanmax(dra_cluster)
np.nanmin(dra_cluster)
np.nanstd(dra_cluster)*3
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        dra_spring_par20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_par20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_par20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        dra_sepapr_par20012010 = np.hstack((dra_spring_par20012010, dra_summer_par20012010, dra_autumn_par20012010))
    else:
        dra_spring_par20012010 = np.hstack((dra_spring_par20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_par20012010 = np.hstack((dra_summer_par20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_par20012010 = np.hstack((dra_autumn_par20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        dra_sepapr_par20012010 = np.hstack((dra_sepapr_par20012010, np.hstack((dra_spring_par20012010, dra_summer_par20012010, dra_autumn_par20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        dra_spring_par20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_par20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_par20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        dra_sepapr_par20112020 = np.hstack((dra_spring_par20112020, dra_summer_par20112020, dra_autumn_par20112020))
    else:
        dra_spring_par20112020 = np.hstack((dra_spring_par20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_par20112020 = np.hstack((dra_summer_par20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_par20112020 = np.hstack((dra_autumn_par20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        dra_sepapr_par20112020 = np.hstack((dra_sepapr_par20112020, np.hstack((dra_spring_par20112020, dra_summer_par20112020, dra_autumn_par20112020))))
#%% Separar para o cluster 4 (BRS)
brs_cluster = par[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
np.nanmedian(brs_cluster)
np.nanmax(brs_cluster)
np.nanmin(brs_cluster)
np.nanstd(brs_cluster)*3
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        brs_spring_par20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_par20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_par20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        brs_sepapr_par20012010 = np.hstack((brs_spring_par20012010, brs_summer_par20012010, brs_autumn_par20012010))
    else:
        brs_spring_par20012010 = np.hstack((brs_spring_par20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_par20012010 = np.hstack((brs_summer_par20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_par20012010 = np.hstack((brs_autumn_par20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        brs_sepapr_par20012010 = np.hstack((brs_sepapr_par20012010, np.hstack((brs_spring_par20012010, brs_summer_par20012010, brs_autumn_par20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        brs_spring_par20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_par20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_par20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        brs_sepapr_par20112020 = np.hstack((brs_spring_par20112020, brs_summer_par20112020, brs_autumn_par20112020))
    else:
        brs_spring_par20112020 = np.hstack((brs_spring_par20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_par20112020 = np.hstack((brs_summer_par20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_par20112020 = np.hstack((brs_autumn_par20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        brs_sepapr_par20112020 = np.hstack((brs_sepapr_par20112020, np.hstack((brs_spring_par20112020, brs_summer_par20112020, brs_autumn_par20112020))))
#%% Separar para o cluster 5 (WEDN)
wedn_cluster = par[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
np.nanmedian(wedn_cluster)
np.nanmax(wedn_cluster)
np.nanmin(wedn_cluster)
np.nanstd(wedn_cluster)*3
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        wedn_spring_par20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_par20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_par20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        wedn_sepapr_par20012010 = np.hstack((wedn_spring_par20012010, wedn_summer_par20012010, wedn_autumn_par20012010))
    else:
        wedn_spring_par20012010 = np.hstack((wedn_spring_par20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_par20012010 = np.hstack((wedn_summer_par20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_par20012010 = np.hstack((wedn_autumn_par20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        wedn_sepapr_par20012010 = np.hstack((wedn_sepapr_par20012010, np.hstack((wedn_spring_par20012010, wedn_summer_par20012010, wedn_autumn_par20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        wedn_spring_par20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_par20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_par20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        wedn_sepapr_par20112020 = np.hstack((wedn_spring_par20112020, wedn_summer_par20112020, wedn_autumn_par20112020))
    else:
        wedn_spring_par20112020 = np.hstack((wedn_spring_par20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_par20112020 = np.hstack((wedn_summer_par20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_par20112020 = np.hstack((wedn_autumn_par20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        wedn_sepapr_par20112020 = np.hstack((wedn_sepapr_par20112020, np.hstack((wedn_spring_par20112020, wedn_summer_par20112020, wedn_autumn_par20112020))))
#%% SST Separar para o cluster 1 (WEDS)
weds_cluster = sst[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
np.nanmedian(weds_cluster)
np.nanmax(weds_cluster)
np.nanmin(weds_cluster)
np.nanstd(weds_cluster)*3
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2001:
        weds_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        weds_sepapr_sst20012010 = np.hstack((weds_spring_sst20012010, weds_summer_sst20012010, weds_autumn_sst20012010))
    else:
        weds_spring_sst20012010 = np.hstack((weds_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_sst20012010 = np.hstack((weds_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_sst20012010 = np.hstack((weds_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        weds_sepapr_sst20012010 = np.hstack((weds_sepapr_sst20012010, np.hstack((weds_spring_sst20012010, weds_summer_sst20012010, weds_autumn_sst20012010))))

# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = weds_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2011:
        weds_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        weds_sepapr_sst20112020 = np.hstack((weds_spring_sst20112020, weds_summer_sst20112020, weds_autumn_sst20112020))
    else:
        weds_spring_sst20112020 = np.hstack((weds_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_sst20112020 = np.hstack((weds_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_sst20112020 = np.hstack((weds_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        weds_sepapr_sst20112020 = np.hstack((weds_sepapr_sst20112020, np.hstack((weds_spring_sst20112020, weds_summer_sst20112020, weds_autumn_sst20112020))))
#%% Sesstar ssta o cluster 2 (GES)
ges_cluster = sst[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
np.nanmedian(ges_cluster)
np.nanmax(ges_cluster)
np.nanmin(ges_cluster)
np.nanstd(ges_cluster)*3
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2001:
        ges_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        ges_sepapr_sst20012010 = np.hstack((ges_spring_sst20012010, ges_summer_sst20012010, ges_autumn_sst20012010))
    else:
        ges_spring_sst20012010 = np.hstack((ges_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_sst20012010 = np.hstack((ges_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_sst20012010 = np.hstack((ges_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        ges_sepapr_sst20012010 = np.hstack((ges_sepapr_sst20012010, np.hstack((ges_spring_sst20012010, ges_summer_sst20012010, ges_autumn_sst20012010))))

# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = ges_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2011:
        ges_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        ges_sepapr_sst20112020 = np.hstack((ges_spring_sst20112020, ges_summer_sst20112020, ges_autumn_sst20112020))
    else:
        ges_spring_sst20112020 = np.hstack((ges_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_sst20112020 = np.hstack((ges_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_sst20112020 = np.hstack((ges_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        ges_sepapr_sst20112020 = np.hstack((ges_sepapr_sst20112020, np.hstack((ges_spring_sst20112020, ges_summer_sst20112020, ges_autumn_sst20112020))))
#%% Sesstar ssta o cluster 3 (DRA)
dra_cluster = sst[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
np.nanmedian(dra_cluster)
np.nanmax(dra_cluster)
np.nanmin(dra_cluster)
np.nanstd(dra_cluster)*3
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2001:
        dra_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        dra_sepapr_sst20012010 = np.hstack((dra_spring_sst20012010, dra_summer_sst20012010, dra_autumn_sst20012010))
    else:
        dra_spring_sst20012010 = np.hstack((dra_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_sst20012010 = np.hstack((dra_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_sst20012010 = np.hstack((dra_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        dra_sepapr_sst20012010 = np.hstack((dra_sepapr_sst20012010, np.hstack((dra_spring_sst20012010, dra_summer_sst20012010, dra_autumn_sst20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = dra_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2011:
        dra_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        dra_sepapr_sst20112020 = np.hstack((dra_spring_sst20112020, dra_summer_sst20112020, dra_autumn_sst20112020))
    else:
        dra_spring_sst20112020 = np.hstack((dra_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_sst20112020 = np.hstack((dra_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_sst20112020 = np.hstack((dra_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        dra_sepapr_sst20112020 = np.hstack((dra_sepapr_sst20112020, np.hstack((dra_spring_sst20112020, dra_summer_sst20112020, dra_autumn_sst20112020))))
#%% Separar para o cluster 4 (BRS)
brs_cluster = sst[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
np.nanmedian(brs_cluster)
np.nanmax(brs_cluster)
np.nanmin(brs_cluster)
np.nanstd(brs_cluster)*3
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2001:
        brs_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        brs_sepapr_sst20012010 = np.hstack((brs_spring_sst20012010, brs_summer_sst20012010, brs_autumn_sst20012010))
    else:
        brs_spring_sst20012010 = np.hstack((brs_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_sst20012010 = np.hstack((brs_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_sst20012010 = np.hstack((brs_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        brs_sepapr_sst20012010 = np.hstack((brs_sepapr_sst20012010, np.hstack((brs_spring_sst20012010, brs_summer_sst20012010, brs_autumn_sst20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = brs_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2011:
        brs_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        brs_sepapr_sst20112020 = np.hstack((brs_spring_sst20112020, brs_summer_sst20112020, brs_autumn_sst20112020))
    else:
        brs_spring_sst20112020 = np.hstack((brs_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_sst20112020 = np.hstack((brs_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_sst20112020 = np.hstack((brs_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        brs_sepapr_sst20112020 = np.hstack((brs_sepapr_sst20112020, np.hstack((brs_spring_sst20112020, brs_summer_sst20112020, brs_autumn_sst20112020))))
#%% Sesstar ssta o cluster 5 (WEDN)
wedn_cluster = sst[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
np.nanmedian(wedn_cluster)
np.nanmax(wedn_cluster)
np.nanmin(wedn_cluster)
np.nanstd(wedn_cluster)*3
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2001:
        wedn_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        wedn_sepapr_sst20012010 = np.hstack((wedn_spring_sst20012010, wedn_summer_sst20012010, wedn_autumn_sst20012010))
    else:
        wedn_spring_sst20012010 = np.hstack((wedn_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_sst20012010 = np.hstack((wedn_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_sst20012010 = np.hstack((wedn_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        wedn_sepapr_sst20012010 = np.hstack((wedn_sepapr_sst20012010, np.hstack((wedn_spring_sst20012010, wedn_summer_sst20012010, wedn_autumn_sst20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years_sst == i-1) & (time_date_months_sst == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_sst == i) & (time_date_months_sst == 4)]
    if i == 2011:
        wedn_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        wedn_sepapr_sst20112020 = np.hstack((wedn_spring_sst20112020, wedn_summer_sst20112020, wedn_autumn_sst20112020))
    else:
        wedn_spring_sst20112020 = np.hstack((wedn_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_sst20112020 = np.hstack((wedn_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_sst20112020 = np.hstack((wedn_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        wedn_sepapr_sst20112020 = np.hstack((wedn_sepapr_sst20112020, np.hstack((wedn_spring_sst20112020, wedn_summer_sst20112020, wedn_autumn_sst20112020))))
#%% Sea Ice Separar para o cluster 1 (WEDS)
weds_cluster = seaice[clusters_sst == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
np.nanmedian(weds_cluster)
np.nanmax(weds_cluster)
np.nanmin(weds_cluster)
np.nanstd(weds_cluster)*3
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = weds_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = weds_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = weds_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2001:
        weds_spring_seaice20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_seaice20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_seaice20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        weds_sepapr_seaice20012010 = np.hstack((weds_spring_seaice20012010, weds_summer_seaice20012010, weds_autumn_seaice20012010))
    else:
        weds_spring_seaice20012010 = np.hstack((weds_spring_seaice20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_seaice20012010 = np.hstack((weds_summer_seaice20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_seaice20012010 = np.hstack((weds_autumn_seaice20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        weds_sepapr_seaice20012010 = np.hstack((weds_sepapr_seaice20012010, np.hstack((weds_spring_seaice20012010, weds_summer_seaice20012010, weds_autumn_seaice20012010))))

# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = weds_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = weds_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = weds_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2011:
        weds_spring_seaice20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_seaice20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_seaice20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        weds_sepapr_seaice20112020 = np.hstack((weds_spring_seaice20112020, weds_summer_seaice20112020, weds_autumn_seaice20112020))
    else:
        weds_spring_seaice20112020 = np.hstack((weds_spring_seaice20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_seaice20112020 = np.hstack((weds_summer_seaice20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_seaice20112020 = np.hstack((weds_autumn_seaice20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        weds_sepapr_seaice20112020 = np.hstack((weds_sepapr_seaice20112020, np.hstack((weds_spring_seaice20112020, weds_summer_seaice20112020, weds_autumn_seaice20112020))))
#%% Seseaicear seaicea o cluster 2 (GES)
ges_cluster = seaice[clusters_sst == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
np.nanmedian(ges_cluster)
np.nanmax(ges_cluster)
np.nanmin(ges_cluster)
np.nanstd(ges_cluster)*3
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = ges_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = ges_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = ges_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2001:
        ges_spring_seaice20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_seaice20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_seaice20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        ges_sepapr_seaice20012010 = np.hstack((ges_spring_seaice20012010, ges_summer_seaice20012010, ges_autumn_seaice20012010))
    else:
        ges_spring_seaice20012010 = np.hstack((ges_spring_seaice20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_seaice20012010 = np.hstack((ges_summer_seaice20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_seaice20012010 = np.hstack((ges_autumn_seaice20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        ges_sepapr_seaice20012010 = np.hstack((ges_sepapr_seaice20012010, np.hstack((ges_spring_seaice20012010, ges_summer_seaice20012010, ges_autumn_seaice20012010))))

# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = ges_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = ges_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = ges_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2011:
        ges_spring_seaice20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_seaice20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_seaice20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        ges_sepapr_seaice20112020 = np.hstack((ges_spring_seaice20112020, ges_summer_seaice20112020, ges_autumn_seaice20112020))
    else:
        ges_spring_seaice20112020 = np.hstack((ges_spring_seaice20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_seaice20112020 = np.hstack((ges_summer_seaice20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_seaice20112020 = np.hstack((ges_autumn_seaice20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        ges_sepapr_seaice20112020 = np.hstack((ges_sepapr_seaice20112020, np.hstack((ges_spring_seaice20112020, ges_summer_seaice20112020, ges_autumn_seaice20112020))))
#%% Sesstar ssta o cluster 3 (DRA)
dra_cluster = seaice[clusters_sst == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
np.nanmedian(dra_cluster)
np.nanmax(dra_cluster)
np.nanmin(dra_cluster)
np.nanstd(dra_cluster)*3
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = dra_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = dra_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = dra_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2001:
        dra_spring_seaice20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_seaice20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_seaice20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        dra_sepapr_seaice20012010 = np.hstack((dra_spring_seaice20012010, dra_summer_seaice20012010, dra_autumn_seaice20012010))
    else:
        dra_spring_seaice20012010 = np.hstack((dra_spring_seaice20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_seaice20012010 = np.hstack((dra_summer_seaice20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_seaice20012010 = np.hstack((dra_autumn_seaice20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        dra_sepapr_seaice20012010 = np.hstack((dra_sepapr_seaice20012010, np.hstack((dra_spring_seaice20012010, dra_summer_seaice20012010, dra_autumn_seaice20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = dra_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = dra_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = dra_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2011:
        dra_spring_seaice20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_seaice20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_seaice20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        dra_sepapr_seaice20112020 = np.hstack((dra_spring_seaice20112020, dra_summer_seaice20112020, dra_autumn_seaice20112020))
    else:
        dra_spring_seaice20112020 = np.hstack((dra_spring_seaice20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_seaice20112020 = np.hstack((dra_summer_seaice20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_seaice20112020 = np.hstack((dra_autumn_seaice20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        dra_sepapr_seaice20112020 = np.hstack((dra_sepapr_seaice20112020, np.hstack((dra_spring_seaice20112020, dra_summer_seaice20112020, dra_autumn_seaice20112020))))
#%% Separar para o cluster 4 (BRS)
brs_cluster = seaice[clusters_sst == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
np.nanmedian(brs_cluster)
np.nanmax(brs_cluster)
np.nanmin(brs_cluster)
np.nanstd(brs_cluster)*3
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = brs_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = brs_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = brs_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2001:
        brs_spring_seaice20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_seaice20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_seaice20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        brs_sepapr_seaice20012010 = np.hstack((brs_spring_seaice20012010, brs_summer_seaice20012010, brs_autumn_seaice20012010))
    else:
        brs_spring_seaice20012010 = np.hstack((brs_spring_seaice20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_seaice20012010 = np.hstack((brs_summer_seaice20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_seaice20012010 = np.hstack((brs_autumn_seaice20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        brs_sepapr_seaice20012010 = np.hstack((brs_sepapr_seaice20012010, np.hstack((brs_spring_seaice20012010, brs_summer_seaice20012010, brs_autumn_seaice20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = brs_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = brs_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = brs_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2011:
        brs_spring_seaice20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_seaice20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_seaice20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        brs_sepapr_seaice20112020 = np.hstack((brs_spring_seaice20112020, brs_summer_seaice20112020, brs_autumn_seaice20112020))
    else:
        brs_spring_seaice20112020 = np.hstack((brs_spring_seaice20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_seaice20112020 = np.hstack((brs_summer_seaice20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_seaice20112020 = np.hstack((brs_autumn_seaice20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        brs_sepapr_seaice20112020 = np.hstack((brs_sepapr_seaice20112020, np.hstack((brs_spring_seaice20112020, brs_summer_seaice20112020, brs_autumn_seaice20112020))))
#%% Sesstar ssta o cluster 5 (WEDN)
wedn_cluster = seaice[clusters_sst == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
np.nanmedian(wedn_cluster)
np.nanmax(wedn_cluster)
np.nanmin(wedn_cluster)
np.nanstd(wedn_cluster)*3
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = wedn_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2001:
        wedn_spring_seaice20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_seaice20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_seaice20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        wedn_sepapr_seaice20012010 = np.hstack((wedn_spring_seaice20012010, wedn_summer_seaice20012010, wedn_autumn_seaice20012010))
    else:
        wedn_spring_seaice20012010 = np.hstack((wedn_spring_seaice20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_seaice20012010 = np.hstack((wedn_summer_seaice20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_seaice20012010 = np.hstack((wedn_autumn_seaice20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        wedn_sepapr_seaice20012010 = np.hstack((wedn_sepapr_seaice20012010, np.hstack((wedn_spring_seaice20012010, wedn_summer_seaice20012010, wedn_autumn_seaice20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = wedn_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years_seaice == i-1) & (time_date_months_seaice == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_seaice == i) & (time_date_months_seaice == 4)]
    if i == 2011:
        wedn_spring_seaice20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_seaice20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_seaice20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        wedn_sepapr_seaice20112020 = np.hstack((wedn_spring_seaice20112020, wedn_summer_seaice20112020, wedn_autumn_seaice20112020))
    else:
        wedn_spring_seaice20112020 = np.hstack((wedn_spring_seaice20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_seaice20112020 = np.hstack((wedn_summer_seaice20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_seaice20112020 = np.hstack((wedn_autumn_seaice20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        wedn_sepapr_seaice20112020 = np.hstack((wedn_sepapr_seaice20112020, np.hstack((wedn_spring_seaice20112020, wedn_summer_seaice20112020, wedn_autumn_seaice20112020))))
#%% Winds Separar para o cluster 1 (WEDS)
weds_cluster = windspeed[clusters_winds == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
np.nanmedian(weds_cluster)
np.nanmax(weds_cluster)
np.nanmin(weds_cluster)
np.nanstd(weds_cluster)*3
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2001:
        weds_spring_winds20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_winds20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_winds20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        weds_sepapr_winds20012010 = np.hstack((weds_spring_winds20012010, weds_summer_winds20012010, weds_autumn_winds20012010))
    else:
        weds_spring_winds20012010 = np.hstack((weds_spring_winds20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_winds20012010 = np.hstack((weds_summer_winds20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_winds20012010 = np.hstack((weds_autumn_winds20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        weds_sepapr_winds20012010 = np.hstack((weds_sepapr_winds20012010, np.hstack((weds_spring_winds20012010, weds_summer_winds20012010, weds_autumn_winds20012010))))

# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = weds_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = weds_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2011:
        weds_spring_winds20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_winds20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_winds20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        weds_sepapr_winds20112020 = np.hstack((weds_spring_winds20112020, weds_summer_winds20112020, weds_autumn_winds20112020))
    else:
        weds_spring_winds20112020 = np.hstack((weds_spring_winds20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_winds20112020 = np.hstack((weds_summer_winds20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_winds20112020 = np.hstack((weds_autumn_winds20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        weds_sepapr_winds20112020 = np.hstack((weds_sepapr_winds20112020, np.hstack((weds_spring_winds20112020, weds_summer_winds20112020, weds_autumn_winds20112020))))
#%% Seseaicear seaicea o cluster 2 (GES)
ges_cluster = windspeed[clusters_winds == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
np.nanmedian(ges_cluster)
np.nanmax(ges_cluster)
np.nanmin(ges_cluster)
np.nanstd(ges_cluster)*3
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2001:
        ges_spring_winds20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_winds20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_winds20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        ges_sepapr_winds20012010 = np.hstack((ges_spring_winds20012010, ges_summer_winds20012010, ges_autumn_winds20012010))
    else:
        ges_spring_winds20012010 = np.hstack((ges_spring_winds20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_winds20012010 = np.hstack((ges_summer_winds20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_winds20012010 = np.hstack((ges_autumn_winds20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        ges_sepapr_winds20012010 = np.hstack((ges_sepapr_winds20012010, np.hstack((ges_spring_winds20012010, ges_summer_winds20012010, ges_autumn_winds20012010))))

# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = ges_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = ges_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2011:
        ges_spring_winds20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_winds20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_winds20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        ges_sepapr_winds20112020 = np.hstack((ges_spring_winds20112020, ges_summer_winds20112020, ges_autumn_winds20112020))
    else:
        ges_spring_winds20112020 = np.hstack((ges_spring_winds20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_winds20112020 = np.hstack((ges_summer_winds20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_winds20112020 = np.hstack((ges_autumn_winds20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        ges_sepapr_winds20112020 = np.hstack((ges_sepapr_winds20112020, np.hstack((ges_spring_winds20112020, ges_summer_winds20112020, ges_autumn_winds20112020))))
#%% Sesstar ssta o cluster 3 (DRA)
dra_cluster = windspeed[clusters_winds == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
np.nanmedian(dra_cluster)
np.nanmax(dra_cluster)
np.nanmin(dra_cluster)
np.nanstd(dra_cluster)*3
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2001:
        dra_spring_winds20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_winds20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_winds20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        dra_sepapr_winds20012010 = np.hstack((dra_spring_winds20012010, dra_summer_winds20012010, dra_autumn_winds20012010))
    else:
        dra_spring_winds20012010 = np.hstack((dra_spring_winds20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_winds20012010 = np.hstack((dra_summer_winds20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_winds20012010 = np.hstack((dra_autumn_winds20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        dra_sepapr_winds20012010 = np.hstack((dra_sepapr_winds20012010, np.hstack((dra_spring_winds20012010, dra_summer_winds20012010, dra_autumn_winds20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = dra_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = dra_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2011:
        dra_spring_winds20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_winds20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_winds20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        dra_sepapr_winds20112020 = np.hstack((dra_spring_winds20112020, dra_summer_winds20112020, dra_autumn_winds20112020))
    else:
        dra_spring_winds20112020 = np.hstack((dra_spring_winds20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_winds20112020 = np.hstack((dra_summer_winds20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_winds20112020 = np.hstack((dra_autumn_winds20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        dra_sepapr_winds20112020 = np.hstack((dra_sepapr_winds20112020, np.hstack((dra_spring_winds20112020, dra_summer_winds20112020, dra_autumn_winds20112020))))
#%% Separar para o cluster 4 (BRS)
brs_cluster = windspeed[clusters_winds == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
np.nanmedian(brs_cluster)
np.nanmax(brs_cluster)
np.nanmin(brs_cluster)
np.nanstd(brs_cluster)*3
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2001:
        brs_spring_winds20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_winds20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_winds20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        brs_sepapr_winds20012010 = np.hstack((brs_spring_winds20012010, brs_summer_winds20012010, brs_autumn_winds20012010))
    else:
        brs_spring_winds20012010 = np.hstack((brs_spring_winds20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_winds20012010 = np.hstack((brs_summer_winds20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_winds20012010 = np.hstack((brs_autumn_winds20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        brs_sepapr_winds20012010 = np.hstack((brs_sepapr_winds20012010, np.hstack((brs_spring_winds20012010, brs_summer_winds20012010, brs_autumn_winds20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = brs_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = brs_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2011:
        brs_spring_winds20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_winds20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_winds20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        brs_sepapr_winds20112020 = np.hstack((brs_spring_winds20112020, brs_summer_winds20112020, brs_autumn_winds20112020))
    else:
        brs_spring_winds20112020 = np.hstack((brs_spring_winds20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_winds20112020 = np.hstack((brs_summer_winds20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_winds20112020 = np.hstack((brs_autumn_winds20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        brs_sepapr_winds20112020 = np.hstack((brs_sepapr_winds20112020, np.hstack((brs_spring_winds20112020, brs_summer_winds20112020, brs_autumn_winds20112020))))
#%% Sesstar ssta o cluster 5 (WEDN)
wedn_cluster = windspeed[clusters_winds == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
np.nanmedian(wedn_cluster)
np.nanmax(wedn_cluster)
np.nanmin(wedn_cluster)
np.nanstd(wedn_cluster)*3
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2001:
        wedn_spring_winds20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_winds20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_winds20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
        wedn_sepapr_winds20012010 = np.hstack((wedn_spring_winds20012010, wedn_summer_winds20012010, wedn_autumn_winds20012010))
    else:
        wedn_spring_winds20012010 = np.hstack((wedn_spring_winds20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_winds20012010 = np.hstack((wedn_summer_winds20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_winds20012010 = np.hstack((wedn_autumn_winds20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
        wedn_sepapr_winds20012010 = np.hstack((wedn_sepapr_winds20012010, np.hstack((wedn_spring_winds20012010, wedn_summer_winds20012010, wedn_autumn_winds20012010))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years_winds == i-1) & (time_date_months_winds == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years_winds == i) & (time_date_months_winds == 4)]
    if i == 2011:
        wedn_spring_winds20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_winds20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_winds20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
        wedn_sepapr_winds20112020 = np.hstack((wedn_spring_winds20112020, wedn_summer_winds20112020, wedn_autumn_winds20112020))
    else:
        wedn_spring_winds20112020 = np.hstack((wedn_spring_winds20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_winds20112020 = np.hstack((wedn_summer_winds20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_winds20112020 = np.hstack((wedn_autumn_winds20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
        wedn_sepapr_winds20112020 = np.hstack((wedn_sepapr_winds20112020, np.hstack((wedn_spring_winds20112020, wedn_summer_winds20112020, wedn_autumn_winds20112020))))
#%% PAR boxplots
# WEDS
spring_par_boxplot_weds = [weds_spring_par20012010[~np.isnan(weds_spring_par20012010)],  weds_spring_par20112020[~np.isnan(weds_spring_par20112020)]]
summer_par_boxplot_weds = [weds_summer_par20012010[~np.isnan(weds_summer_par20012010)],  weds_summer_par20112020[~np.isnan(weds_summer_par20112020)]]
autumn_par_boxplot_weds = [weds_autumn_par20012010[~np.isnan(weds_autumn_par20012010)],  weds_autumn_par20112020[~np.isnan(weds_autumn_par20112020)]]
sepapr_par_boxplot_weds = [weds_autumn_par20012010[~np.isnan(weds_autumn_par20012010)],  weds_autumn_par20112020[~np.isnan(weds_autumn_par20112020)]]
# GES
spring_par_boxplot_ges = [ges_spring_par20012010[~np.isnan(ges_spring_par20012010)],  ges_spring_par20112020[~np.isnan(ges_spring_par20112020)]]
summer_par_boxplot_ges = [ges_summer_par20012010[~np.isnan(ges_summer_par20012010)],  ges_summer_par20112020[~np.isnan(ges_summer_par20112020)]]
autumn_par_boxplot_ges = [ges_autumn_par20012010[~np.isnan(ges_autumn_par20012010)],  ges_autumn_par20112020[~np.isnan(ges_autumn_par20112020)]]
sepapr_par_boxplot_ges = [ges_autumn_par20012010[~np.isnan(ges_autumn_par20012010)],  ges_autumn_par20112020[~np.isnan(ges_autumn_par20112020)]]
# DRA
spring_par_boxplot_dra = [dra_spring_par20012010[~np.isnan(dra_spring_par20012010)],  dra_spring_par20112020[~np.isnan(dra_spring_par20112020)]]
summer_par_boxplot_dra = [dra_summer_par20012010[~np.isnan(dra_summer_par20012010)],  dra_summer_par20112020[~np.isnan(dra_summer_par20112020)]]
autumn_par_boxplot_dra = [dra_autumn_par20012010[~np.isnan(dra_autumn_par20012010)],  dra_autumn_par20112020[~np.isnan(dra_autumn_par20112020)]]
sepapr_par_boxplot_dra = [dra_autumn_par20012010[~np.isnan(dra_autumn_par20012010)],  dra_autumn_par20112020[~np.isnan(dra_autumn_par20112020)]]
# BRS
spring_par_boxplot_brs = [brs_spring_par20012010[~np.isnan(brs_spring_par20012010)],  brs_spring_par20112020[~np.isnan(brs_spring_par20112020)]]
summer_par_boxplot_brs = [brs_summer_par20012010[~np.isnan(brs_summer_par20012010)],  brs_summer_par20112020[~np.isnan(brs_summer_par20112020)]]
autumn_par_boxplot_brs = [brs_autumn_par20012010[~np.isnan(brs_autumn_par20012010)],  brs_autumn_par20112020[~np.isnan(brs_autumn_par20112020)]]
sepapr_par_boxplot_brs = [brs_autumn_par20012010[~np.isnan(brs_autumn_par20012010)],  brs_autumn_par20112020[~np.isnan(brs_autumn_par20112020)]]
# WEDN
spring_par_boxplot_wedn = [wedn_spring_par20012010[~np.isnan(wedn_spring_par20012010)],  wedn_spring_par20112020[~np.isnan(wedn_spring_par20112020)]]
summer_par_boxplot_wedn = [wedn_summer_par20012010[~np.isnan(wedn_summer_par20012010)],  wedn_summer_par20112020[~np.isnan(wedn_summer_par20112020)]]
autumn_par_boxplot_wedn = [wedn_autumn_par20012010[~np.isnan(wedn_autumn_par20012010)],  wedn_autumn_par20112020[~np.isnan(wedn_autumn_par20112020)]]
sepapr_par_boxplot_wedn = [wedn_autumn_par20012010[~np.isnan(wedn_autumn_par20012010)],  wedn_autumn_par20112020[~np.isnan(wedn_autumn_par20112020)]]
#%% SST boxplots
# WEDS
spring_sst_boxplot_weds = [weds_spring_sst20012010[~np.isnan(weds_spring_sst20012010)],  weds_spring_sst20112020[~np.isnan(weds_spring_sst20112020)]]
summer_sst_boxplot_weds = [weds_summer_sst20012010[~np.isnan(weds_summer_sst20012010)],  weds_summer_sst20112020[~np.isnan(weds_summer_sst20112020)]]
autumn_sst_boxplot_weds = [weds_autumn_sst20012010[~np.isnan(weds_autumn_sst20012010)],  weds_autumn_sst20112020[~np.isnan(weds_autumn_sst20112020)]]
sepapr_sst_boxplot_weds = [weds_autumn_sst20012010[~np.isnan(weds_autumn_sst20012010)],  weds_autumn_sst20112020[~np.isnan(weds_autumn_sst20112020)]]
# GES
spring_sst_boxplot_ges = [ges_spring_sst20012010[~np.isnan(ges_spring_sst20012010)],  ges_spring_sst20112020[~np.isnan(ges_spring_sst20112020)]]
summer_sst_boxplot_ges = [ges_summer_sst20012010[~np.isnan(ges_summer_sst20012010)],  ges_summer_sst20112020[~np.isnan(ges_summer_sst20112020)]]
autumn_sst_boxplot_ges = [ges_autumn_sst20012010[~np.isnan(ges_autumn_sst20012010)],  ges_autumn_sst20112020[~np.isnan(ges_autumn_sst20112020)]]
sepapr_sst_boxplot_ges = [ges_autumn_sst20012010[~np.isnan(ges_autumn_sst20012010)],  ges_autumn_sst20112020[~np.isnan(ges_autumn_sst20112020)]]
# DRA
spring_sst_boxplot_dra = [dra_spring_sst20012010[~np.isnan(dra_spring_sst20012010)],  dra_spring_sst20112020[~np.isnan(dra_spring_sst20112020)]]
summer_sst_boxplot_dra = [dra_summer_sst20012010[~np.isnan(dra_summer_sst20012010)],  dra_summer_sst20112020[~np.isnan(dra_summer_sst20112020)]]
autumn_sst_boxplot_dra = [dra_autumn_sst20012010[~np.isnan(dra_autumn_sst20012010)],  dra_autumn_sst20112020[~np.isnan(dra_autumn_sst20112020)]]
sepapr_sst_boxplot_dra = [dra_autumn_sst20012010[~np.isnan(dra_autumn_sst20012010)],  dra_autumn_sst20112020[~np.isnan(dra_autumn_sst20112020)]]
# BRS
spring_sst_boxplot_brs = [brs_spring_sst20012010[~np.isnan(brs_spring_sst20012010)],  brs_spring_sst20112020[~np.isnan(brs_spring_sst20112020)]]
summer_sst_boxplot_brs = [brs_summer_sst20012010[~np.isnan(brs_summer_sst20012010)],  brs_summer_sst20112020[~np.isnan(brs_summer_sst20112020)]]
autumn_sst_boxplot_brs = [brs_autumn_sst20012010[~np.isnan(brs_autumn_sst20012010)],  brs_autumn_sst20112020[~np.isnan(brs_autumn_sst20112020)]]
sepapr_sst_boxplot_brs = [brs_autumn_sst20012010[~np.isnan(brs_autumn_sst20012010)],  brs_autumn_sst20112020[~np.isnan(brs_autumn_sst20112020)]]
# WEDN
spring_sst_boxplot_wedn = [wedn_spring_sst20012010[~np.isnan(wedn_spring_sst20012010)],  wedn_spring_sst20112020[~np.isnan(wedn_spring_sst20112020)]]
summer_sst_boxplot_wedn = [wedn_summer_sst20012010[~np.isnan(wedn_summer_sst20012010)],  wedn_summer_sst20112020[~np.isnan(wedn_summer_sst20112020)]]
autumn_sst_boxplot_wedn = [wedn_autumn_sst20012010[~np.isnan(wedn_autumn_sst20012010)],  wedn_autumn_sst20112020[~np.isnan(wedn_autumn_sst20112020)]]
sepapr_sst_boxplot_wedn = [wedn_autumn_sst20012010[~np.isnan(wedn_autumn_sst20012010)],  wedn_autumn_sst20112020[~np.isnan(wedn_autumn_sst20112020)]]
#%% Sea Ice boxplots
# WEDS
spring_seaice_boxplot_weds = [weds_spring_seaice20012010[~np.isnan(weds_spring_seaice20012010)],  weds_spring_seaice20112020[~np.isnan(weds_spring_seaice20112020)]]
summer_seaice_boxplot_weds = [weds_summer_seaice20012010[~np.isnan(weds_summer_seaice20012010)],  weds_summer_seaice20112020[~np.isnan(weds_summer_seaice20112020)]]
autumn_seaice_boxplot_weds = [weds_autumn_seaice20012010[~np.isnan(weds_autumn_seaice20012010)],  weds_autumn_seaice20112020[~np.isnan(weds_autumn_seaice20112020)]]
sepapr_seaice_boxplot_weds = [weds_autumn_seaice20012010[~np.isnan(weds_autumn_seaice20012010)],  weds_autumn_seaice20112020[~np.isnan(weds_autumn_seaice20112020)]]
# GES
spring_seaice_boxplot_ges = [ges_spring_seaice20012010[~np.isnan(ges_spring_seaice20012010)],  ges_spring_seaice20112020[~np.isnan(ges_spring_seaice20112020)]]
summer_seaice_boxplot_ges = [ges_summer_seaice20012010[~np.isnan(ges_summer_seaice20012010)],  ges_summer_seaice20112020[~np.isnan(ges_summer_seaice20112020)]]
autumn_seaice_boxplot_ges = [ges_autumn_seaice20012010[~np.isnan(ges_autumn_seaice20012010)],  ges_autumn_seaice20112020[~np.isnan(ges_autumn_seaice20112020)]]
sepapr_seaice_boxplot_ges = [ges_autumn_seaice20012010[~np.isnan(ges_autumn_seaice20012010)],  ges_autumn_seaice20112020[~np.isnan(ges_autumn_seaice20112020)]]
# DRA
spring_seaice_boxplot_dra = [dra_spring_seaice20012010[~np.isnan(dra_spring_seaice20012010)],  dra_spring_seaice20112020[~np.isnan(dra_spring_seaice20112020)]]
summer_seaice_boxplot_dra = [dra_summer_seaice20012010[~np.isnan(dra_summer_seaice20012010)],  dra_summer_seaice20112020[~np.isnan(dra_summer_seaice20112020)]]
autumn_seaice_boxplot_dra = [dra_autumn_seaice20012010[~np.isnan(dra_autumn_seaice20012010)],  dra_autumn_seaice20112020[~np.isnan(dra_autumn_seaice20112020)]]
sepapr_seaice_boxplot_dra = [dra_autumn_seaice20012010[~np.isnan(dra_autumn_seaice20012010)],  dra_autumn_seaice20112020[~np.isnan(dra_autumn_seaice20112020)]]
# BRS
spring_seaice_boxplot_brs = [brs_spring_seaice20012010[~np.isnan(brs_spring_seaice20012010)],  brs_spring_seaice20112020[~np.isnan(brs_spring_seaice20112020)]]
summer_seaice_boxplot_brs = [brs_summer_seaice20012010[~np.isnan(brs_summer_seaice20012010)],  brs_summer_seaice20112020[~np.isnan(brs_summer_seaice20112020)]]
autumn_seaice_boxplot_brs = [brs_autumn_seaice20012010[~np.isnan(brs_autumn_seaice20012010)],  brs_autumn_seaice20112020[~np.isnan(brs_autumn_seaice20112020)]]
sepapr_seaice_boxplot_brs = [brs_autumn_seaice20012010[~np.isnan(brs_autumn_seaice20012010)],  brs_autumn_seaice20112020[~np.isnan(brs_autumn_seaice20112020)]]
# WEDN
spring_seaice_boxplot_wedn = [wedn_spring_seaice20012010[~np.isnan(wedn_spring_seaice20012010)],  wedn_spring_seaice20112020[~np.isnan(wedn_spring_seaice20112020)]]
summer_seaice_boxplot_wedn = [wedn_summer_seaice20012010[~np.isnan(wedn_summer_seaice20012010)],  wedn_summer_seaice20112020[~np.isnan(wedn_summer_seaice20112020)]]
autumn_seaice_boxplot_wedn = [wedn_autumn_seaice20012010[~np.isnan(wedn_autumn_seaice20012010)],  wedn_autumn_seaice20112020[~np.isnan(wedn_autumn_seaice20112020)]]
sepapr_seaice_boxplot_wedn = [wedn_autumn_seaice20012010[~np.isnan(wedn_autumn_seaice20012010)],  wedn_autumn_seaice20112020[~np.isnan(wedn_autumn_seaice20112020)]]
#%% Winds boxplots
# WEDS
spring_winds_boxplot_weds = [weds_spring_winds20012010[~np.isnan(weds_spring_winds20012010)],  weds_spring_winds20112020[~np.isnan(weds_spring_winds20112020)]]
summer_winds_boxplot_weds = [weds_summer_winds20012010[~np.isnan(weds_summer_winds20012010)],  weds_summer_winds20112020[~np.isnan(weds_summer_winds20112020)]]
autumn_winds_boxplot_weds = [weds_autumn_winds20012010[~np.isnan(weds_autumn_winds20012010)],  weds_autumn_winds20112020[~np.isnan(weds_autumn_winds20112020)]]
sepapr_winds_boxplot_weds = [weds_autumn_winds20012010[~np.isnan(weds_autumn_winds20012010)],  weds_autumn_winds20112020[~np.isnan(weds_autumn_winds20112020)]]
# GES
spring_winds_boxplot_ges = [ges_spring_winds20012010[~np.isnan(ges_spring_winds20012010)],  ges_spring_winds20112020[~np.isnan(ges_spring_winds20112020)]]
summer_winds_boxplot_ges = [ges_summer_winds20012010[~np.isnan(ges_summer_winds20012010)],  ges_summer_winds20112020[~np.isnan(ges_summer_winds20112020)]]
autumn_winds_boxplot_ges = [ges_autumn_winds20012010[~np.isnan(ges_autumn_winds20012010)],  ges_autumn_winds20112020[~np.isnan(ges_autumn_winds20112020)]]
sepapr_winds_boxplot_ges = [ges_autumn_winds20012010[~np.isnan(ges_autumn_winds20012010)],  ges_autumn_winds20112020[~np.isnan(ges_autumn_winds20112020)]]
# DRA
spring_winds_boxplot_dra = [dra_spring_winds20012010[~np.isnan(dra_spring_winds20012010)],  dra_spring_winds20112020[~np.isnan(dra_spring_winds20112020)]]
summer_winds_boxplot_dra = [dra_summer_winds20012010[~np.isnan(dra_summer_winds20012010)],  dra_summer_winds20112020[~np.isnan(dra_summer_winds20112020)]]
autumn_winds_boxplot_dra = [dra_autumn_winds20012010[~np.isnan(dra_autumn_winds20012010)],  dra_autumn_winds20112020[~np.isnan(dra_autumn_winds20112020)]]
sepapr_winds_boxplot_dra = [dra_autumn_winds20012010[~np.isnan(dra_autumn_winds20012010)],  dra_autumn_winds20112020[~np.isnan(dra_autumn_winds20112020)]]
# BRS
spring_winds_boxplot_brs = [brs_spring_winds20012010[~np.isnan(brs_spring_winds20012010)],  brs_spring_winds20112020[~np.isnan(brs_spring_winds20112020)]]
summer_winds_boxplot_brs = [brs_summer_winds20012010[~np.isnan(brs_summer_winds20012010)],  brs_summer_winds20112020[~np.isnan(brs_summer_winds20112020)]]
autumn_winds_boxplot_brs = [brs_autumn_winds20012010[~np.isnan(brs_autumn_winds20012010)],  brs_autumn_winds20112020[~np.isnan(brs_autumn_winds20112020)]]
sepapr_winds_boxplot_brs = [brs_autumn_winds20012010[~np.isnan(brs_autumn_winds20012010)],  brs_autumn_winds20112020[~np.isnan(brs_autumn_winds20112020)]]
# WEDN
spring_winds_boxplot_wedn = [wedn_spring_winds20012010[~np.isnan(wedn_spring_winds20012010)],  wedn_spring_winds20112020[~np.isnan(wedn_spring_winds20112020)]]
summer_winds_boxplot_wedn = [wedn_summer_winds20012010[~np.isnan(wedn_summer_winds20012010)],  wedn_summer_winds20112020[~np.isnan(wedn_summer_winds20112020)]]
autumn_winds_boxplot_wedn = [wedn_autumn_winds20012010[~np.isnan(wedn_autumn_winds20012010)],  wedn_autumn_winds20112020[~np.isnan(wedn_autumn_winds20112020)]]
sepapr_winds_boxplot_wedn = [wedn_autumn_winds20012010[~np.isnan(wedn_autumn_winds20012010)],  wedn_autumn_winds20112020[~np.isnan(wedn_autumn_winds20112020)]]
#%% Prepare plot
def setBoxColors_DRA(bp):
    plt.setp(bp['boxes'][0], facecolor='#f2a612')
    plt.setp(bp['caps'][0], color='#f2a612')
    plt.setp(bp['caps'][1], color='#f2a612')
    plt.setp(bp['whiskers'][0], color='#f2a612')
    plt.setp(bp['whiskers'][1], color='#f2a612')
    plt.setp(bp['fliers'][0], color='#f2a612')
    plt.setp(bp['fliers'][1], color='#f2a612')
    plt.setp(bp['medians'][0], color='k', linewidth=2)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='#f2a612')
    plt.setp(bp['boxes'][1], hatch = '////')
    plt.setp(bp['caps'][2], color='#f2a612')
    plt.setp(bp['caps'][3], color='#f2a612')
    plt.setp(bp['whiskers'][2], color='#f2a612')
    plt.setp(bp['whiskers'][3], color='#f2a612')
    plt.setp(bp['medians'][1], color='k', linewidth=2)
def setBoxColors_WEDN(bp):
    plt.setp(bp['boxes'][0], facecolor='#534d41')
    plt.setp(bp['caps'][0], color='#534d41')
    plt.setp(bp['caps'][1], color='#534d41')
    plt.setp(bp['whiskers'][0], color='#534d41')
    plt.setp(bp['whiskers'][1], color='#534d41')
    plt.setp(bp['fliers'][0], color='#534d41')
    plt.setp(bp['fliers'][1], color='#534d41')
    plt.setp(bp['medians'][0], color='k', linewidth=2)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='#534d41')
    plt.setp(bp['boxes'][1], hatch = '////')
    plt.setp(bp['caps'][2], color='#534d41')
    plt.setp(bp['caps'][3], color='#534d41')
    plt.setp(bp['whiskers'][2], color='#534d41')
    plt.setp(bp['whiskers'][3], color='#534d41')
    plt.setp(bp['medians'][1], color='k', linewidth=2)
def setBoxColors_BRS(bp):
    plt.setp(bp['boxes'][0], facecolor='#6a984e')
    plt.setp(bp['caps'][0], color='#6a984e')
    plt.setp(bp['caps'][1], color='#6a984e')
    plt.setp(bp['whiskers'][0], color='#6a984e')
    plt.setp(bp['whiskers'][1], color='#6a984e')
    plt.setp(bp['fliers'][0], color='#6a984e')
    plt.setp(bp['fliers'][1], color='#6a984e')
    plt.setp(bp['medians'][0], color='k', linewidth=2)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='#6a984e')
    plt.setp(bp['boxes'][1], hatch = '////')
    plt.setp(bp['caps'][2], color='#6a984e')
    plt.setp(bp['caps'][3], color='#6a984e')
    plt.setp(bp['whiskers'][2], color='#6a984e')
    plt.setp(bp['whiskers'][3], color='#6a984e')
    plt.setp(bp['medians'][1], color='k', linewidth=2)
def setBoxColors_GES(bp):
    plt.setp(bp['boxes'][0], facecolor='#2c4ea3')
    plt.setp(bp['caps'][0], color='#2c4ea3')
    plt.setp(bp['caps'][1], color='#2c4ea3')
    plt.setp(bp['whiskers'][0], color='#2c4ea3')
    plt.setp(bp['whiskers'][1], color='#2c4ea3')
    plt.setp(bp['fliers'][0], color='#2c4ea3')
    plt.setp(bp['fliers'][1], color='#2c4ea3')
    plt.setp(bp['medians'][0], color='k', linewidth=2)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='#2c4ea3')
    plt.setp(bp['boxes'][1], hatch = '////')
    plt.setp(bp['caps'][2], color='#2c4ea3')
    plt.setp(bp['caps'][3], color='#2c4ea3')
    plt.setp(bp['whiskers'][2], color='#2c4ea3')
    plt.setp(bp['whiskers'][3], color='#2c4ea3')
    plt.setp(bp['medians'][1], color='k', linewidth=2)
def setBoxColors_WEDS(bp):
    plt.setp(bp['boxes'][0], facecolor='#da2b39')
    plt.setp(bp['caps'][0], color='#da2b39')
    plt.setp(bp['caps'][1], color='#da2b39')
    plt.setp(bp['whiskers'][0], color='#da2b39')
    plt.setp(bp['whiskers'][1], color='#da2b39')
    plt.setp(bp['fliers'][0], color='#da2b39')
    plt.setp(bp['fliers'][1], color='#da2b39')
    plt.setp(bp['medians'][0], color='k', linewidth=2)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 2.5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='#da2b39')
    plt.setp(bp['boxes'][1], hatch = '////')
    plt.setp(bp['caps'][2], color='#da2b39')
    plt.setp(bp['caps'][3], color='#da2b39')
    plt.setp(bp['whiskers'][2], color='#da2b39')
    plt.setp(bp['whiskers'][3], color='#da2b39')
    plt.setp(bp['medians'][1], color='k', linewidth=2)    
# SST
fig = plt.figure(figsize=(6,6))
ax = plt.axes()
# DRA
bp = plt.boxplot(sepapr_sst_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = plt.boxplot(sepapr_sst_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = plt.boxplot(sepapr_sst_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = plt.boxplot(sepapr_sst_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = plt.boxplot(sepapr_sst_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
plt.xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=12)
plt.ylabel('SST (°C)', fontsize=14)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
legend_elements = [patches.Patch(facecolor='w', edgecolor='k', label='2001-2010'),
                   patches.Patch(facecolor='w', edgecolor='k', label='2011-2020', hatch='////')]
plt.legend(handles=legend_elements, loc=0, fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\SST_allregions.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Sea Ice
fig = plt.figure(figsize=(6,6))
ax = plt.axes()
# DRA
bp = plt.boxplot(sepapr_seaice_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = plt.boxplot(sepapr_seaice_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = plt.boxplot(sepapr_seaice_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = plt.boxplot(sepapr_seaice_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = plt.boxplot(sepapr_seaice_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
plt.xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=12)
plt.ylabel('Sea Ice (%)', fontsize=14)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
legend_elements = [patches.Patch(facecolor='w', edgecolor='k', label='2001-2010'),
                   patches.Patch(facecolor='w', edgecolor='k', label='2011-2020', hatch='////')]
plt.legend(handles=legend_elements, loc=0, fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\Seaice_allregions.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# PAR
fig = plt.figure(figsize=(6,6))
ax = plt.axes()
# DRA
bp = plt.boxplot(sepapr_par_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = plt.boxplot(sepapr_par_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = plt.boxplot(sepapr_par_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = plt.boxplot(sepapr_par_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = plt.boxplot(sepapr_par_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
plt.xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=12)
plt.ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=14)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
legend_elements = [patches.Patch(facecolor='w', edgecolor='k', label='2001-2010'),
                   patches.Patch(facecolor='w', edgecolor='k', label='2011-2020', hatch='////')]
plt.legend(handles=legend_elements, loc=0, fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\par_allregions.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# WINDS
fig = plt.figure(figsize=(6,6))
ax = plt.axes()
# DRA
bp = plt.boxplot(sepapr_winds_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = plt.boxplot(sepapr_winds_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = plt.boxplot(sepapr_winds_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = plt.boxplot(sepapr_winds_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = plt.boxplot(sepapr_winds_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
plt.xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=12)
plt.ylabel('Wind Speed (m$^{-1}$)', fontsize=14)
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
legend_elements = [patches.Patch(facecolor='w', edgecolor='k', label='2001-2010'),
                   patches.Patch(facecolor='w', edgecolor='k', label='2011-2020', hatch='////')]
plt.legend(handles=legend_elements, loc=0, fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\winds_allregions.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Test t for differences between 2001-2010 and 2011-2020!
# SST
stats.ttest_ind(sepapr_sst_boxplot_dra[0], sepapr_sst_boxplot_dra[1]) # ***
stats.ttest_ind(sepapr_sst_boxplot_brs[0], sepapr_sst_boxplot_brs[1]) # **
stats.ttest_ind(sepapr_sst_boxplot_wedn[0], sepapr_sst_boxplot_wedn[1]) # ***
stats.ttest_ind(sepapr_sst_boxplot_ges[0], sepapr_sst_boxplot_ges[1]) # ***
stats.ttest_ind(sepapr_sst_boxplot_weds[0], sepapr_sst_boxplot_weds[1]) # ***
# sea Ice
stats.ttest_ind(sepapr_seaice_boxplot_dra[0], sepapr_seaice_boxplot_dra[1]) # ***
stats.ttest_ind(sepapr_seaice_boxplot_brs[0], sepapr_seaice_boxplot_brs[1])
stats.ttest_ind(sepapr_seaice_boxplot_wedn[0], sepapr_seaice_boxplot_wedn[1]) # *
stats.ttest_ind(sepapr_seaice_boxplot_ges[0], sepapr_seaice_boxplot_ges[1]) #
stats.ttest_ind(sepapr_seaice_boxplot_weds[0], sepapr_seaice_boxplot_weds[1]) # ***
# PAR
stats.ttest_ind(sepapr_par_boxplot_dra[0], sepapr_par_boxplot_dra[1]) #
stats.ttest_ind(sepapr_par_boxplot_brs[0], sepapr_par_boxplot_brs[1]) #
stats.ttest_ind(sepapr_par_boxplot_wedn[0], sepapr_par_boxplot_wedn[1]) #
stats.ttest_ind(sepapr_par_boxplot_ges[0], sepapr_par_boxplot_ges[1]) #
stats.ttest_ind(sepapr_par_boxplot_weds[0], sepapr_par_boxplot_weds[1]) #
# Winds
stats.ttest_ind(sepapr_winds_boxplot_dra[0], sepapr_winds_boxplot_dra[1]) # ***
stats.ttest_ind(sepapr_winds_boxplot_brs[0], sepapr_winds_boxplot_brs[1]) # **
stats.ttest_ind(sepapr_winds_boxplot_wedn[0], sepapr_winds_boxplot_wedn[1]) # ***
stats.ttest_ind(sepapr_winds_boxplot_ges[0], sepapr_winds_boxplot_ges[1]) # ***
stats.ttest_ind(sepapr_winds_boxplot_weds[0], sepapr_winds_boxplot_weds[1]) # ***
np.nanmean(sepapr_par_boxplot_wedn[0])
np.nanmean(sepapr_par_boxplot_wedn[1])
np.nanmean(sepapr_par_boxplot_wedn[0]) - np.nanmean(sepapr_par_boxplot_wedn[1])
#%% Plot 4 by 3 figure
fig, ax = plt.subplots(4, 3, figsize=(9,9))
# SST SPRING
# DRA
bp = ax[0,0].boxplot(spring_sst_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[0,0].boxplot(spring_sst_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[0,0].boxplot(spring_sst_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[0,0].boxplot(spring_sst_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[0,0].boxplot(spring_sst_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[0,0].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
ax[0,0].set_ylabel('SST (°C)', fontsize=12)
#ax[0,0].set_ylim(-2,4)
#ax[0,0].tick_params(axis='x', labelsize=12)
#ax[0,0].tick_params(axis='y', labelsize=12)
# SST SUMMER
# DRA
bp = ax[0,1].boxplot(summer_sst_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[0,1].boxplot(summer_sst_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[0,1].boxplot(summer_sst_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[0,1].boxplot(summer_sst_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[0,1].boxplot(summer_sst_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[0,1].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
#ax[0,1].set_ylabel('SST (°C)', fontsize=12)
#ax[0,1].tick_params(axis='x', labelsize=12)
#ax[0,1.tick_params(axis='y', labelsize=12)
# SST AUTUMN
# DRA
bp = ax[0,2].boxplot(autumn_sst_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[0,2].boxplot(autumn_sst_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[0,2].boxplot(autumn_sst_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[0,2].boxplot(autumn_sst_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[0,2].boxplot(autumn_sst_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[0,2].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
#ax[0,1].set_ylabel('SST (°C)', fontsize=12)
#ax[0,1].tick_params(axis='x', labelsize=12)
#ax[0,1.tick_params(axis='y', labelsize=12)
## Sea Ice SPRING
# DRA
bp = ax[1,0].boxplot(spring_seaice_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[1,0].boxplot(spring_seaice_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[1,0].boxplot(spring_seaice_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[1,0].boxplot(spring_seaice_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[1,0].boxplot(spring_seaice_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[1,0].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
ax[1,0].set_ylabel('Sea Ice (%)', fontsize=12)
ax[1,0].set_ylim(0,100)
#ax[0,0].tick_params(axis='x', labelsize=12)
#ax[0,0].tick_params(axis='y', labelsize=12)
# Sea Ice  SUMMER
# DRA
bp = ax[1,1].boxplot(summer_seaice_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[1,1].boxplot(summer_seaice_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[1,1].boxplot(summer_seaice_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[1,1].boxplot(summer_seaice_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[1,1].boxplot(summer_seaice_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[1,1].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
ax[1,1].set_ylim(0,100)
#ax[0,1].set_ylabel('seaice (°C)', fontsize=12)
#ax[0,1].tick_params(axis='x', labelsize=12)
#ax[0,1.tick_params(axis='y', labelsize=12)
# Sea Ice  AUTUMN
# DRA
bp = ax[1,2].boxplot(autumn_seaice_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[1,2].boxplot(autumn_seaice_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[1,2].boxplot(autumn_seaice_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[1,2].boxplot(autumn_seaice_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[1,2].boxplot(autumn_seaice_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[1,2].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
ax[1,2].set_ylim(0,100)
#ax[0,1].set_ylabel('SST (°C)', fontsize=12)
#ax[0,1].tick_params(axis='x', labelsize=12)
#ax[0,1.tick_params(axis='y', labelsize=12)
## PAR SPRING
# DRA
bp = ax[2,0].boxplot(spring_par_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[2,0].boxplot(spring_par_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[2,0].boxplot(spring_par_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[2,0].boxplot(spring_par_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[2,0].boxplot(spring_par_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[2,0].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
ax[2,0].set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax[2,0].set_ylim(0,100)
#ax[0,0].tick_params(axis='x', labelsize=12)
#ax[0,0].tick_params(axis='y', labelsize=12)
# PAR  SUMMER
# DRA
bp = ax[2,1].boxplot(summer_par_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[2,1].boxplot(summer_par_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[2,1].boxplot(summer_par_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[2,1].boxplot(summer_par_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[2,1].boxplot(summer_par_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[2,1].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
#ax[2,1].set_ylim(0,100)
#ax[0,1].set_ylabel('par (°C)', fontsize=12)
#ax[0,1].tick_params(axis='x', labelsize=12)
#ax[0,1.tick_params(axis='y', labelsize=12)
# PAR  AUTUMN
# DRA
bp = ax[2,2].boxplot(autumn_par_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[2,2].boxplot(autumn_par_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[2,2].boxplot(autumn_par_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[2,2].boxplot(autumn_par_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[2,2].boxplot(autumn_par_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[2,2].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
#ax[2,2].set_ylim(0,100)
#ax[0,1].set_ylabel('SST (°C)', fontsize=12)
#ax[0,1].tick_params(axis='x', labelsize=12)
#ax[0,1.tick_params(axis='y', labelsize=12)
## Winds SPRING
# DRA
bp = ax[3,0].boxplot(spring_winds_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[3,0].boxplot(spring_winds_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[3,0].boxplot(spring_winds_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[3,0].boxplot(spring_winds_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[3,0].boxplot(spring_winds_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[3,0].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
ax[3,0].set_ylabel('Wind Speed (m$^{-1}$)', fontsize=12)
#ax[2,0].set_ylim(0,100)
#ax[0,0].tick_windsams(axis='x', labelsize=12)
#ax[0,0].tick_windsams(axis='y', labelsize=12)
# Winds  SUMMER
# DRA
bp = ax[3,1].boxplot(summer_winds_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[3,1].boxplot(summer_winds_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[3,1].boxplot(summer_winds_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[3,1].boxplot(summer_winds_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[3,1].boxplot(summer_winds_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[3,1].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
#ax[2,1].set_ylim(0,100)
#ax[0,1].set_ylabel('winds (°C)', fontsize=12)
#ax[0,1].tick_windsams(axis='x', labelsize=12)
#ax[0,1.tick_windsams(axis='y', labelsize=12)
# Winds  AUTUMN
# DRA
bp = ax[3,2].boxplot(autumn_winds_boxplot_dra, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors_DRA(bp)
# BRS
bp = ax[3,2].boxplot(autumn_winds_boxplot_brs, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors_BRS(bp)
# WEDN
bp = ax[3,2].boxplot(autumn_winds_boxplot_wedn, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors_WEDN(bp)
# GES
bp = ax[3,2].boxplot(autumn_winds_boxplot_ges, positions = [10, 11], widths = 0.6, patch_artist=True)
setBoxColors_GES(bp)
# WEDS
bp = ax[3,2].boxplot(autumn_winds_boxplot_weds, positions = [13, 14], widths = 0.6, patch_artist=True)
setBoxColors_WEDS(bp)
ax[3,2].set_xticks(ticks=[1.5, 4.5, 7.5, 10.5, 13.5], labels=['DRA', 'BRS', 'WEDN', 'GES', 'WEDS'],
           fontsize=10)
#ax[2,2].set_ylim(0,100)
#ax[0,1].set_ylabel('SST (°C)', fontsize=12)
#ax[0,1].tick_windsams(axis='x', labelsize=12)
#ax[0,1.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\Fig3.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
# SST SPRING
stats.ttest_ind(spring_sst_boxplot_dra[0], spring_sst_boxplot_dra[1]) #
stats.ttest_ind(spring_sst_boxplot_brs[0], spring_sst_boxplot_brs[1]) #
stats.ttest_ind(spring_sst_boxplot_wedn[0], spring_sst_boxplot_wedn[1]) # ***
stats.ttest_ind(spring_sst_boxplot_ges[0], spring_sst_boxplot_ges[1]) # ***
stats.ttest_ind(spring_sst_boxplot_weds[0], spring_sst_boxplot_weds[1]) # ***
(np.nanmean(spring_sst_boxplot_weds[1]) - np.nanmean(spring_sst_boxplot_weds[0]))/abs(np.nanmean(spring_sst_boxplot_weds[0]))*100
# SST SUMMER
stats.ttest_ind(summer_sst_boxplot_dra[0], summer_sst_boxplot_dra[1]) # ***
stats.ttest_ind(summer_sst_boxplot_brs[0], summer_sst_boxplot_brs[1]) #
stats.ttest_ind(summer_sst_boxplot_wedn[0], summer_sst_boxplot_wedn[1]) # ***
stats.ttest_ind(summer_sst_boxplot_ges[0], summer_sst_boxplot_ges[1]) # **
stats.ttest_ind(summer_sst_boxplot_weds[0], summer_sst_boxplot_weds[1]) #
(np.nanmean(summer_sst_boxplot_brs[1]) - np.nanmean(summer_sst_boxplot_brs[0]))/abs(np.nanmean(summer_sst_boxplot_brs[0]))*100
# SST AUTUMN
stats.ttest_ind(autumn_sst_boxplot_dra[0], autumn_sst_boxplot_dra[1]) # ***
stats.ttest_ind(autumn_sst_boxplot_brs[0], autumn_sst_boxplot_brs[1]) # **
stats.ttest_ind(autumn_sst_boxplot_wedn[0], autumn_sst_boxplot_wedn[1]) # ***
stats.ttest_ind(autumn_sst_boxplot_ges[0], autumn_sst_boxplot_ges[1]) # ***
stats.ttest_ind(autumn_sst_boxplot_weds[0], autumn_sst_boxplot_weds[1]) # ***
(np.nanmean(autumn_sst_boxplot_weds[1]) - np.nanmean(autumn_sst_boxplot_weds[0]))/abs(np.nanmean(autumn_sst_boxplot_weds[0]))*100
# SEA ICE SPRING
stats.ttest_ind(spring_seaice_boxplot_dra[0], spring_seaice_boxplot_dra[1]) #
stats.ttest_ind(spring_seaice_boxplot_brs[0], spring_seaice_boxplot_brs[1]) # **
stats.ttest_ind(spring_seaice_boxplot_wedn[0], spring_seaice_boxplot_wedn[1]) # ***
stats.ttest_ind(spring_seaice_boxplot_ges[0], spring_seaice_boxplot_ges[1]) # ***
stats.ttest_ind(spring_seaice_boxplot_weds[0], spring_seaice_boxplot_weds[1]) # ***
(np.nanmean(spring_seaice_boxplot_weds[1]) - np.nanmean(spring_seaice_boxplot_weds[0]))/abs(np.nanmean(spring_seaice_boxplot_weds[0]))*100
# SEA ICE SUMMER
stats.ttest_ind(summer_seaice_boxplot_dra[0], summer_seaice_boxplot_dra[1]) # ***
stats.ttest_ind(summer_seaice_boxplot_brs[0], summer_seaice_boxplot_brs[1]) # ***
stats.ttest_ind(summer_seaice_boxplot_wedn[0], summer_seaice_boxplot_wedn[1]) # ***
stats.ttest_ind(summer_seaice_boxplot_ges[0], summer_seaice_boxplot_ges[1]) # ***
stats.ttest_ind(summer_seaice_boxplot_weds[0], summer_seaice_boxplot_weds[1]) # ***
(np.nanmean(summer_seaice_boxplot_weds[1]) - np.nanmean(summer_seaice_boxplot_weds[0]))/abs(np.nanmean(summer_seaice_boxplot_weds[0]))*100
# SEA ICE AUTUMN
stats.ttest_ind(autumn_seaice_boxplot_dra[0], autumn_seaice_boxplot_dra[1]) # ***
stats.ttest_ind(autumn_seaice_boxplot_brs[0], autumn_seaice_boxplot_brs[1])
stats.ttest_ind(autumn_seaice_boxplot_wedn[0], autumn_seaice_boxplot_wedn[1])
stats.ttest_ind(autumn_seaice_boxplot_ges[0], autumn_seaice_boxplot_ges[1]) #
stats.ttest_ind(autumn_seaice_boxplot_weds[0], autumn_seaice_boxplot_weds[1]) # ***
(np.nanmean(autumn_seaice_boxplot_weds[1]) - np.nanmean(autumn_seaice_boxplot_weds[0]))/abs(np.nanmean(autumn_seaice_boxplot_weds[0]))*100
# PAR SPRING
stats.ttest_ind(spring_par_boxplot_dra[0], spring_par_boxplot_dra[1]) #
stats.ttest_ind(spring_par_boxplot_brs[0], spring_par_boxplot_brs[1]) #
stats.ttest_ind(spring_par_boxplot_wedn[0], spring_par_boxplot_wedn[1]) # ***
stats.ttest_ind(spring_par_boxplot_ges[0], spring_par_boxplot_ges[1]) # ***
stats.ttest_ind(spring_par_boxplot_weds[0], spring_par_boxplot_weds[1]) # ***
(np.nanmean(spring_par_boxplot_weds[1]) - np.nanmean(spring_par_boxplot_weds[0]))/abs(np.nanmean(spring_par_boxplot_weds[0]))*100
# PAR SUMMER
stats.ttest_ind(summer_par_boxplot_dra[0], summer_par_boxplot_dra[1]) # **
stats.ttest_ind(summer_par_boxplot_brs[0], summer_par_boxplot_brs[1]) #
stats.ttest_ind(summer_par_boxplot_wedn[0], summer_par_boxplot_wedn[1]) #
stats.ttest_ind(summer_par_boxplot_ges[0], summer_par_boxplot_ges[1]) # ***
stats.ttest_ind(summer_par_boxplot_weds[0], summer_par_boxplot_weds[1]) # ***
(np.nanmean(summer_par_boxplot_weds[1]) - np.nanmean(summer_par_boxplot_weds[0]))/abs(np.nanmean(summer_par_boxplot_weds[0]))*100
# PAR AUTUMN
stats.ttest_ind(autumn_par_boxplot_dra[0], autumn_par_boxplot_dra[1]) #
stats.ttest_ind(autumn_par_boxplot_brs[0], autumn_par_boxplot_brs[1])
stats.ttest_ind(autumn_par_boxplot_wedn[0], autumn_par_boxplot_wedn[1])
stats.ttest_ind(autumn_par_boxplot_ges[0], autumn_par_boxplot_ges[1]) #
stats.ttest_ind(autumn_par_boxplot_weds[0], autumn_par_boxplot_weds[1]) # ***
(np.nanmean(autumn_par_boxplot_weds[1]) - np.nanmean(autumn_par_boxplot_weds[0]))/abs(np.nanmean(autumn_par_boxplot_weds[0]))*100
# WINDS SPRING
stats.ttest_ind(spring_winds_boxplot_dra[0], spring_winds_boxplot_dra[1]) # ***
stats.ttest_ind(spring_winds_boxplot_brs[0], spring_winds_boxplot_brs[1]) # ***
stats.ttest_ind(spring_winds_boxplot_wedn[0], spring_winds_boxplot_wedn[1]) # ***
stats.ttest_ind(spring_winds_boxplot_ges[0], spring_winds_boxplot_ges[1]) # ***
stats.ttest_ind(spring_winds_boxplot_weds[0], spring_winds_boxplot_weds[1]) # ***
(np.nanmean(spring_winds_boxplot_ges[1]) - np.nanmean(spring_winds_boxplot_ges[0]))/abs(np.nanmean(spring_winds_boxplot_ges[0]))*100
# WINDS SUMMER
stats.ttest_ind(summer_winds_boxplot_dra[0], summer_winds_boxplot_dra[1]) # ***
stats.ttest_ind(summer_winds_boxplot_brs[0], summer_winds_boxplot_brs[1]) # ***
stats.ttest_ind(summer_winds_boxplot_wedn[0], summer_winds_boxplot_wedn[1]) # ***
stats.ttest_ind(summer_winds_boxplot_ges[0], summer_winds_boxplot_ges[1]) 
stats.ttest_ind(summer_winds_boxplot_weds[0], summer_winds_boxplot_weds[1]) # **
(np.nanmean(summer_winds_boxplot_weds[1]) - np.nanmean(summer_winds_boxplot_weds[0]))/abs(np.nanmean(summer_winds_boxplot_weds[0]))*100
# WINDS AUTUMN
stats.ttest_ind(autumn_winds_boxplot_dra[0], autumn_winds_boxplot_dra[1]) # **
stats.ttest_ind(autumn_winds_boxplot_brs[0], autumn_winds_boxplot_brs[1]) # ***
stats.ttest_ind(autumn_winds_boxplot_wedn[0], autumn_winds_boxplot_wedn[1]) # ***
stats.ttest_ind(autumn_winds_boxplot_ges[0], autumn_winds_boxplot_ges[1]) # ***
stats.ttest_ind(autumn_winds_boxplot_weds[0], autumn_winds_boxplot_weds[1])
(np.nanmean(autumn_winds_boxplot_weds[1]) - np.nanmean(autumn_winds_boxplot_weds[0]))/abs(np.nanmean(autumn_winds_boxplot_weds[0]))*100

































#%%
plt.xticks(ticks=[1.5, 4.5, 7.5], labels=['SPRING', 'SUMMER', 'AUTUMN'],
           fontsize=12)
plt.ylabel('par', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\ges_par.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

























































#%% par - Separate betweeen 2001-2010 and 2011-2020
def setBoxColors(bp):
    plt.setp(bp['boxes'][0], facecolor='blue')
    plt.setp(bp['caps'][0], color='blue')
    plt.setp(bp['caps'][1], color='blue')
    plt.setp(bp['whiskers'][0], color='blue')
    plt.setp(bp['whiskers'][1], color='blue')
    plt.setp(bp['fliers'][0], color='blue')
    plt.setp(bp['fliers'][1], color='blue')
    plt.setp(bp['medians'][0], color='k', linewidth=1)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='red')
    plt.setp(bp['caps'][2], color='red')
    plt.setp(bp['caps'][3], color='red')
    plt.setp(bp['whiskers'][2], color='red')
    plt.setp(bp['whiskers'][3], color='red')
    plt.setp(bp['medians'][1], color='k', linewidth=1)

spring_boxplot = [ges_spring_par20012010[~np.isnan(ges_spring_par20012010)],  ges_spring_par20112020[~np.isnan(ges_spring_par20112020)]]
summer_boxplot = [ges_summer_par20012010[~np.isnan(ges_summer_par20012010)],  ges_summer_par20112020[~np.isnan(ges_summer_par20112020)]]
autumn_boxplot = [ges_autumn_par20012010[~np.isnan(ges_autumn_par20012010)],  ges_autumn_par20112020[~np.isnan(ges_autumn_par20112020)]]

fig = plt.figure()
ax = plt.axes()
# first boxplot pair
bp = plt.boxplot(spring_boxplot, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# second boxplot pair
bp = plt.boxplot(summer_boxplot, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# thrid boxplot pair
bp = plt.boxplot(autumn_boxplot, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors(bp)
plt.xticks(ticks=[1.5, 4.5, 7.5], labels=['SPRING', 'SUMMER', 'AUTUMN'],
           fontsize=12)
plt.ylabel('par', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\ges_par.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(ges_spring_par20012010, ges_spring_par20112020, nan_policy = 'omit') # *** -46.04%
stats.mannwhitneyu(ges_summer_par20012010, ges_summer_par20112020, nan_policy = 'omit') # *** -12.93%
stats.mannwhitneyu(ges_autumn_par20012010, ges_autumn_par20112020, nan_policy = 'omit') # - 5.38%
#%% Separar para o cluster 1 (WEDsouth)
weds_cluster = par[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
np.nanmedian(weds_cluster)
np.nanmax(weds_cluster)
np.nanmin(weds_cluster)
np.nanstd(weds_cluster)*3
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        weds_spring_par20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_par20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_par20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        weds_spring_par20012010 = np.hstack((weds_spring_par20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_par20012010 = np.hstack((weds_summer_par20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_par20012010 = np.hstack((weds_autumn_par20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        weds_spring_par20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_par20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_par20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        weds_spring_par20112020 = np.hstack((weds_spring_par20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_par20112020 = np.hstack((weds_summer_par20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_par20112020 = np.hstack((weds_autumn_par20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
#%% par - Separate betweeen 2001-2010 and 2011-2020
def setBoxColors(bp):
    plt.setp(bp['boxes'][0], facecolor='blue')
    plt.setp(bp['caps'][0], color='blue')
    plt.setp(bp['caps'][1], color='blue')
    plt.setp(bp['whiskers'][0], color='blue')
    plt.setp(bp['whiskers'][1], color='blue')
    plt.setp(bp['fliers'][0], color='blue')
    plt.setp(bp['fliers'][1], color='blue')
    plt.setp(bp['medians'][0], color='k', linewidth=1)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='red')
    plt.setp(bp['caps'][2], color='red')
    plt.setp(bp['caps'][3], color='red')
    plt.setp(bp['whiskers'][2], color='red')
    plt.setp(bp['whiskers'][3], color='red')
    plt.setp(bp['medians'][1], color='k', linewidth=1)

spring_boxplot = [weds_spring_par20012010[~np.isnan(weds_spring_par20012010)],  weds_spring_par20112020[~np.isnan(weds_spring_par20112020)]]
summer_boxplot = [weds_summer_par20012010[~np.isnan(weds_summer_par20012010)],  weds_summer_par20112020[~np.isnan(weds_summer_par20112020)]]
autumn_boxplot = [weds_autumn_par20012010[~np.isnan(weds_autumn_par20012010)],  weds_autumn_par20112020[~np.isnan(weds_autumn_par20112020)]]

fig = plt.figure()
ax = plt.axes()
# first boxplot pair
bp = plt.boxplot(spring_boxplot, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# second boxplot pair
bp = plt.boxplot(summer_boxplot, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# thrid boxplot pair
bp = plt.boxplot(autumn_boxplot, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors(bp)
plt.xticks(ticks=[1.5, 4.5, 7.5], labels=['SPRING', 'SUMMER', 'AUTUMN'],
           fontsize=12)
plt.ylabel('par', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\weds_par.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(weds_spring_par20012010, weds_spring_par20112020, nan_policy = 'omit') # *** -34.50
stats.mannwhitneyu(weds_summer_par20012010, weds_summer_par20112020, nan_policy = 'omit') # *** -35.76
stats.mannwhitneyu(weds_autumn_par20012010, weds_autumn_par20112020, nan_policy = 'omit') # ** -37.30
#%% Separar para o cluster 4 (BRS)
brs_cluster = par[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
np.nanmedian(brs_cluster)
np.nanmax(brs_cluster)
np.nanmin(brs_cluster)
np.nanstd(brs_cluster)*3
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        brs_spring_par20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_par20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_par20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        brs_spring_par20012010 = np.hstack((brs_spring_par20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_par20012010 = np.hstack((brs_summer_par20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_par20012010 = np.hstack((brs_autumn_par20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        brs_spring_par20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_par20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_par20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        brs_spring_par20112020 = np.hstack((brs_spring_par20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_par20112020 = np.hstack((brs_summer_par20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_par20112020 = np.hstack((brs_autumn_par20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
#%% par - Separate betweeen 2001-2010 and 2011-2020
def setBoxColors(bp):
    plt.setp(bp['boxes'][0], facecolor='blue')
    plt.setp(bp['caps'][0], color='blue')
    plt.setp(bp['caps'][1], color='blue')
    plt.setp(bp['whiskers'][0], color='blue')
    plt.setp(bp['whiskers'][1], color='blue')
    plt.setp(bp['fliers'][0], color='blue')
    plt.setp(bp['fliers'][1], color='blue')
    plt.setp(bp['medians'][0], color='k', linewidth=1)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='red')
    plt.setp(bp['caps'][2], color='red')
    plt.setp(bp['caps'][3], color='red')
    plt.setp(bp['whiskers'][2], color='red')
    plt.setp(bp['whiskers'][3], color='red')
    plt.setp(bp['medians'][1], color='k', linewidth=1)

spring_boxplot = [brs_spring_par20012010[~np.isnan(brs_spring_par20012010)],  brs_spring_par20112020[~np.isnan(brs_spring_par20112020)]]
summer_boxplot = [brs_summer_par20012010[~np.isnan(brs_summer_par20012010)],  brs_summer_par20112020[~np.isnan(brs_summer_par20112020)]]
autumn_boxplot = [brs_autumn_par20012010[~np.isnan(brs_autumn_par20012010)],  brs_autumn_par20112020[~np.isnan(brs_autumn_par20112020)]]

fig = plt.figure()
ax = plt.axes()
# first boxplot pair
bp = plt.boxplot(spring_boxplot, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# second boxplot pair
bp = plt.boxplot(summer_boxplot, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# thrid boxplot pair
bp = plt.boxplot(autumn_boxplot, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors(bp)
plt.xticks(ticks=[1.5, 4.5, 7.5], labels=['SPRING', 'SUMMER', 'AUTUMN'],
           fontsize=12)
plt.ylabel('par', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\brs_par.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(brs_spring_par20012010, brs_spring_par20112020, nan_policy = 'omit') # *
stats.mannwhitneyu(brs_summer_par20012010, brs_summer_par20112020, nan_policy = 'omit') #
stats.mannwhitneyu(brs_autumn_par20012010, brs_autumn_par20112020, nan_policy = 'omit') #
#%% Separar para o cluster 5 (WEDn)
wedn_cluster = par[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
np.nanmedian(wedn_cluster)
np.nanmax(wedn_cluster)
np.nanmin(wedn_cluster)
np.nanstd(wedn_cluster)*3
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        wedn_spring_par20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_par20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_par20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        wedn_spring_par20012010 = np.hstack((wedn_spring_par20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_par20012010 = np.hstack((wedn_summer_par20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_par20012010 = np.hstack((wedn_autumn_par20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        wedn_spring_par20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_par20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_par20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        wedn_spring_par20112020 = np.hstack((wedn_spring_par20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_par20112020 = np.hstack((wedn_summer_par20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_par20112020 = np.hstack((wedn_autumn_par20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
#%% par - Separate betweeen 2001-2010 and 2011-2020
def setBoxColors(bp):
    plt.setp(bp['boxes'][0], facecolor='blue')
    plt.setp(bp['caps'][0], color='blue')
    plt.setp(bp['caps'][1], color='blue')
    plt.setp(bp['whiskers'][0], color='blue')
    plt.setp(bp['whiskers'][1], color='blue')
    plt.setp(bp['fliers'][0], color='blue')
    plt.setp(bp['fliers'][1], color='blue')
    plt.setp(bp['medians'][0], color='k', linewidth=1)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='red')
    plt.setp(bp['caps'][2], color='red')
    plt.setp(bp['caps'][3], color='red')
    plt.setp(bp['whiskers'][2], color='red')
    plt.setp(bp['whiskers'][3], color='red')
    plt.setp(bp['medians'][1], color='k', linewidth=1)

spring_boxplot = [wedn_spring_par20012010[~np.isnan(wedn_spring_par20012010)],  wedn_spring_par20112020[~np.isnan(wedn_spring_par20112020)]]
summer_boxplot = [wedn_summer_par20012010[~np.isnan(wedn_summer_par20012010)],  wedn_summer_par20112020[~np.isnan(wedn_summer_par20112020)]]
autumn_boxplot = [wedn_autumn_par20012010[~np.isnan(wedn_autumn_par20012010)],  wedn_autumn_par20112020[~np.isnan(wedn_autumn_par20112020)]]

fig = plt.figure()
ax = plt.axes()
# first boxplot pair
bp = plt.boxplot(spring_boxplot, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# second boxplot pair
bp = plt.boxplot(summer_boxplot, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# thrid boxplot pair
bp = plt.boxplot(autumn_boxplot, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors(bp)
plt.xticks(ticks=[1.5, 4.5, 7.5], labels=['SPRING', 'SUMMER', 'AUTUMN'],
           fontsize=12)
plt.ylabel('par', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\wedn_par.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(wedn_spring_par20012010, wedn_spring_par20112020, nan_policy = 'omit') # ***  
stats.mannwhitneyu(wedn_summer_par20012010, wedn_summer_par20112020, nan_policy = 'omit') #
stats.mannwhitneyu(wedn_autumn_par20012010, wedn_autumn_par20112020, nan_policy = 'omit') # 
#%% Separar para o cluster 3 (DRA)
dra_cluster = par[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
np.nanmedian(dra_cluster)
np.nanmax(dra_cluster)
np.nanmin(dra_cluster)
np.nanstd(dra_cluster)*3
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
# 2001-2010
for i in np.arange(2001, 2011):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2001:
        dra_spring_par20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_par20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_par20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        dra_spring_par20012010 = np.hstack((dra_spring_par20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_par20012010 = np.hstack((dra_summer_par20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_par20012010 = np.hstack((dra_autumn_par20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
# 2011-2020
for i in np.arange(2011, 2021):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    if i == 2011:
        dra_spring_par20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_par20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_par20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        dra_spring_par20112020 = np.hstack((dra_spring_par20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_par20112020 = np.hstack((dra_summer_par20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_par20112020 = np.hstack((dra_autumn_par20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
#%% par - Separate betweeen 2001-2010 and 2011-2020
def setBoxColors(bp):
    plt.setp(bp['boxes'][0], facecolor='blue')
    plt.setp(bp['caps'][0], color='blue')
    plt.setp(bp['caps'][1], color='blue')
    plt.setp(bp['whiskers'][0], color='blue')
    plt.setp(bp['whiskers'][1], color='blue')
    plt.setp(bp['fliers'][0], color='blue')
    plt.setp(bp['fliers'][1], color='blue')
    plt.setp(bp['medians'][0], color='k', linewidth=1)
    plt.setp(bp['fliers'][0], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)
    plt.setp(bp['fliers'][1], marker='x', markerfacecolor='k', markersize = 5, alpha=0.3)    
    plt.setp(bp['boxes'][1], facecolor='red')
    plt.setp(bp['caps'][2], color='red')
    plt.setp(bp['caps'][3], color='red')
    plt.setp(bp['whiskers'][2], color='red')
    plt.setp(bp['whiskers'][3], color='red')
    plt.setp(bp['medians'][1], color='k', linewidth=1)

spring_boxplot = [dra_spring_par20012010[~np.isnan(dra_spring_par20012010)],  dra_spring_par20112020[~np.isnan(dra_spring_par20112020)]]
summer_boxplot = [dra_summer_par20012010[~np.isnan(dra_summer_par20012010)],  dra_summer_par20112020[~np.isnan(dra_summer_par20112020)]]
autumn_boxplot = [dra_autumn_par20012010[~np.isnan(dra_autumn_par20012010)],  dra_autumn_par20112020[~np.isnan(dra_autumn_par20112020)]]

fig = plt.figure()
ax = plt.axes()
# first boxplot pair
bp = plt.boxplot(spring_boxplot, positions = [1, 2], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# second boxplot pair
bp = plt.boxplot(summer_boxplot, positions = [4, 5], widths = 0.6, patch_artist=True)
setBoxColors(bp)
# thrid boxplot pair
bp = plt.boxplot(autumn_boxplot, positions = [7, 8], widths = 0.6, patch_artist=True)
setBoxColors(bp)
plt.xticks(ticks=[1.5, 4.5, 7.5], labels=['SPRING', 'SUMMER', 'AUTUMN'],
           fontsize=12)
plt.ylabel('par', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\dra_par.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(dra_spring_par20012010, dra_spring_par20112020, nan_policy = 'omit') # -2.41
stats.mannwhitneyu(dra_summer_par20012010, dra_summer_par20112020, nan_policy = 'omit') # ** -3.56
stats.mannwhitneyu(dra_autumn_par20012010, dra_autumn_par20112020, nan_policy = 'omit') # -3.37
    





