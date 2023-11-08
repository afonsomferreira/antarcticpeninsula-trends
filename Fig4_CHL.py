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
#%% Load data 1998-2022
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
#%% Chl Separar para cada cluster
#WEDS
weds_cluster = chl[clusters == 1,:]
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
        weds_spring_chl20012010 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        weds_summer_chl20012010 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        weds_autumn_chl20012010 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        weds_sepapr_chl20012010 = np.nanmean(np.hstack((weds_spring_chl20012010, weds_summer_chl20012010, weds_autumn_chl20012010)))
    else:
        weds_spring_chl20012010 = np.hstack((weds_spring_chl20012010, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        weds_summer_chl20012010 = np.hstack((weds_summer_chl20012010, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        weds_autumn_chl20012010 = np.hstack((weds_autumn_chl20012010, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        weds_sepapr_chl20012010 = np.hstack((weds_sepapr_chl20012010, np.nanmean(np.hstack((weds_spring_chl20012010, weds_summer_chl20012010, weds_autumn_chl20012010)))))

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
        weds_spring_chl20112020 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        weds_summer_chl20112020 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        weds_autumn_chl20112020 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        weds_sepapr_chl20112020 = np.nanmean(np.hstack((weds_spring_chl20112020, weds_summer_chl20112020, weds_autumn_chl20112020)))
    else:
        weds_spring_chl20112020 = np.hstack((weds_spring_chl20112020, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        weds_summer_chl20112020 = np.hstack((weds_summer_chl20112020, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        weds_autumn_chl20112020 = np.hstack((weds_autumn_chl20112020, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        weds_sepapr_chl20112020 = np.hstack((weds_sepapr_chl20112020, np.nanmean(np.hstack((weds_spring_chl20112020, weds_summer_chl20112020, weds_autumn_chl20112020)))))
#GES
ges_cluster = chl[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
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
        ges_spring_chl20012010 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        ges_summer_chl20012010 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        ges_autumn_chl20012010 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        ges_sepapr_chl20012010 = np.nanmean(np.hstack((ges_spring_chl20012010, ges_summer_chl20012010, ges_autumn_chl20012010)))
    else:
        ges_spring_chl20012010 = np.hstack((ges_spring_chl20012010, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        ges_summer_chl20012010 = np.hstack((ges_summer_chl20012010, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        ges_autumn_chl20012010 = np.hstack((ges_autumn_chl20012010, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        ges_sepapr_chl20012010 = np.hstack((ges_sepapr_chl20012010, np.nanmean(np.hstack((ges_spring_chl20012010, ges_summer_chl20012010, ges_autumn_chl20012010)))))
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
        ges_spring_chl20112020 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        ges_summer_chl20112020 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        ges_autumn_chl20112020 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        ges_sepapr_chl20112020 = np.nanmean(np.hstack((ges_spring_chl20112020, ges_summer_chl20112020, ges_autumn_chl20112020)))
    else:
        ges_spring_chl20112020 = np.hstack((ges_spring_chl20112020, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        ges_summer_chl20112020 = np.hstack((ges_summer_chl20112020, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        ges_autumn_chl20112020 = np.hstack((ges_autumn_chl20112020, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        ges_sepapr_chl20112020 = np.hstack((ges_sepapr_chl20112020, np.nanmean(np.hstack((ges_spring_chl20112020, ges_summer_chl20112020, ges_autumn_chl20112020)))))
#DRA
dra_cluster = chl[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
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
        dra_spring_chl20012010 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        dra_summer_chl20012010 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        dra_autumn_chl20012010 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        dra_sepapr_chl20012010 = np.nanmean(np.hstack((dra_spring_chl20012010, dra_summer_chl20012010, dra_autumn_chl20012010)))
    else:
        dra_spring_chl20012010 = np.hstack((dra_spring_chl20012010, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        dra_summer_chl20012010 = np.hstack((dra_summer_chl20012010, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        dra_autumn_chl20012010 = np.hstack((dra_autumn_chl20012010, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        dra_sepapr_chl20012010 = np.hstack((dra_sepapr_chl20012010, np.nanmean(np.hstack((dra_spring_chl20012010, dra_summer_chl20012010, dra_autumn_chl20012010)))))
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
        dra_spring_chl20112020 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        dra_summer_chl20112020 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        dra_autumn_chl20112020 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        dra_sepapr_chl20112020 = np.nanmean(np.hstack((dra_spring_chl20112020, dra_summer_chl20112020, dra_autumn_chl20112020)))
    else:
        dra_spring_chl20112020 = np.hstack((dra_spring_chl20112020, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        dra_summer_chl20112020 = np.hstack((dra_summer_chl20112020, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        dra_autumn_chl20112020 = np.hstack((dra_autumn_chl20112020, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        dra_sepapr_chl20112020 = np.hstack((dra_sepapr_chl20112020, np.nanmean(np.hstack((dra_spring_chl20112020, dra_summer_chl20112020, dra_autumn_chl20112020)))))
#BRS
brs_cluster = chl[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
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
        brs_spring_chl20012010 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        brs_summer_chl20012010 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        brs_autumn_chl20012010 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        brs_sepapr_chl20012010 = np.nanmean(np.hstack((brs_spring_chl20012010, brs_summer_chl20012010, brs_autumn_chl20012010)))
    else:
        brs_spring_chl20012010 = np.hstack((brs_spring_chl20012010, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        brs_summer_chl20012010 = np.hstack((brs_summer_chl20012010, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        brs_autumn_chl20012010 = np.hstack((brs_autumn_chl20012010, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        brs_sepapr_chl20012010 = np.hstack((brs_sepapr_chl20012010, np.nanmean(np.hstack((brs_spring_chl20012010, brs_summer_chl20012010, brs_autumn_chl20012010)))))
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
        brs_spring_chl20112020 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        brs_summer_chl20112020 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        brs_autumn_chl20112020 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        brs_sepapr_chl20112020 = np.nanmean(np.hstack((brs_spring_chl20112020, brs_summer_chl20112020, brs_autumn_chl20112020)))
    else:
        brs_spring_chl20112020 = np.hstack((brs_spring_chl20112020, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        brs_summer_chl20112020 = np.hstack((brs_summer_chl20112020, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        brs_autumn_chl20112020 = np.hstack((brs_autumn_chl20112020, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        brs_sepapr_chl20112020 = np.hstack((brs_sepapr_chl20112020, np.nanmean(np.hstack((brs_spring_chl20112020, brs_summer_chl20112020, brs_autumn_chl20112020)))))
#WEDN
wedn_cluster = chl[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
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
        wedn_spring_chl20012010 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        wedn_summer_chl20012010 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        wedn_autumn_chl20012010 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        wedn_sepapr_chl20012010 = np.nanmean(np.hstack((wedn_spring_chl20012010, wedn_summer_chl20012010, wedn_autumn_chl20012010)))
    else:
        wedn_spring_chl20012010 = np.hstack((wedn_spring_chl20012010, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        wedn_summer_chl20012010 = np.hstack((wedn_summer_chl20012010, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        wedn_autumn_chl20012010 = np.hstack((wedn_autumn_chl20012010, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        wedn_sepapr_chl20012010 = np.hstack((wedn_sepapr_chl20012010, np.nanmean(np.hstack((wedn_spring_chl20012010, wedn_summer_chl20012010, wedn_autumn_chl20012010)))))
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
        wedn_spring_chl20112020 = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))
        wedn_summer_chl20112020 = np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))
        wedn_autumn_chl20112020 = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
        wedn_sepapr_chl20112020 = np.nanmean(np.hstack((wedn_spring_chl20112020, wedn_summer_chl20112020, wedn_autumn_chl20112020)))
    else:
        wedn_spring_chl20112020 = np.hstack((wedn_spring_chl20112020, np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov)))))
        wedn_summer_chl20112020 = np.hstack((wedn_summer_chl20112020, np.nanmean(np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)))))
        wedn_autumn_chl20112020 = np.hstack((wedn_autumn_chl20112020, np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))))
        wedn_sepapr_chl20112020 = np.hstack((wedn_sepapr_chl20112020, np.nanmean(np.hstack((wedn_spring_chl20112020, wedn_summer_chl20112020, wedn_autumn_chl20112020)))))
#%% Test differences
# Chl - WEDS
stats.mannwhitneyu(weds_sepapr_chl20012010, weds_sepapr_chl20112020) #
# Chl - GES
stats.mannwhitneyu(ges_sepapr_chl20012010, ges_sepapr_chl20112020) # *
# Chl - DRA
stats.mannwhitneyu(dra_sepapr_chl20012010, dra_sepapr_chl20112020) # *
# Chl - BRS
stats.mannwhitneyu(brs_sepapr_chl20012010, brs_sepapr_chl20112020) # *
# Chl - WEDN
stats.mannwhitneyu(wedn_sepapr_chl20012010, wedn_sepapr_chl20112020) #
#%% Plot diffs
plt.boxplot([weds_sepapr_chl20012010, weds_sepapr_chl20112020])
plt.boxplot([ges_sepapr_chl20012010, ges_sepapr_chl20112020])
plt.boxplot([dra_sepapr_chl20012010, dra_sepapr_chl20112020])
plt.boxplot([brs_sepapr_chl20012010, brs_sepapr_chl20112020])
plt.boxplot([wedn_sepapr_chl20012010, wedn_sepapr_chl20112020])
#%% Calculate means
np.nanmean(weds_sepapr_chl20012010)
np.nanmean(weds_sepapr_chl20112020)


