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
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
### Load data 1998-2020
fh = np.load('sst-seaice_19972021.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
sst = fh['sst']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Load upscaled 4km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_sst.npz',allow_pickle = True)
clusters = fh['clusters']
#%% Separar para o cluster 2 (GES)
ges_cluster = sst[clusters == 2,:]
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
        ges_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        ges_spring_sst20012010 = np.hstack((ges_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_sst20012010 = np.hstack((ges_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_sst20012010 = np.hstack((ges_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
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
        ges_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        ges_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        ges_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        ges_spring_sst20112020 = np.hstack((ges_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        ges_summer_sst20112020 = np.hstack((ges_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        ges_autumn_sst20112020 = np.hstack((ges_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
#%% sst - Separate betweeen 2001-2010 and 2011-2020
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

spring_boxplot = [ges_spring_sst20012010[~np.isnan(ges_spring_sst20012010)],  ges_spring_sst20112020[~np.isnan(ges_spring_sst20112020)]]
summer_boxplot = [ges_summer_sst20012010[~np.isnan(ges_summer_sst20012010)],  ges_summer_sst20112020[~np.isnan(ges_summer_sst20112020)]]
autumn_boxplot = [ges_autumn_sst20012010[~np.isnan(ges_autumn_sst20012010)],  ges_autumn_sst20112020[~np.isnan(ges_autumn_sst20112020)]]

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
plt.ylabel('SST', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\ges_sst.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(ges_spring_sst20012010, ges_spring_sst20112020, nan_policy = 'omit')
stats.mannwhitneyu(ges_summer_sst20012010, ges_summer_sst20112020, nan_policy = 'omit') # ***
stats.mannwhitneyu(ges_autumn_sst20012010, ges_autumn_sst20112020, nan_policy = 'omit') # ***
#%% Separar para o cluster 1 (WEDsouth)
weds_cluster = sst[clusters == 1,:]
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
        weds_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        weds_spring_sst20012010 = np.hstack((weds_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_sst20012010 = np.hstack((weds_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_sst20012010 = np.hstack((weds_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
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
        weds_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        weds_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        weds_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        weds_spring_sst20112020 = np.hstack((weds_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        weds_summer_sst20112020 = np.hstack((weds_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        weds_autumn_sst20112020 = np.hstack((weds_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
#%% sst - Separate betweeen 2001-2010 and 2011-2020
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

spring_boxplot = [weds_spring_sst20012010[~np.isnan(weds_spring_sst20012010)],  weds_spring_sst20112020[~np.isnan(weds_spring_sst20112020)]]
summer_boxplot = [weds_summer_sst20012010[~np.isnan(weds_summer_sst20012010)],  weds_summer_sst20112020[~np.isnan(weds_summer_sst20112020)]]
autumn_boxplot = [weds_autumn_sst20012010[~np.isnan(weds_autumn_sst20012010)],  weds_autumn_sst20112020[~np.isnan(weds_autumn_sst20112020)]]

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
plt.ylabel('SST', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\weds_sst.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(weds_spring_sst20012010, weds_spring_sst20112020, nan_policy = 'omit') # ***
stats.mannwhitneyu(weds_summer_sst20012010, weds_summer_sst20112020, nan_policy = 'omit') #
stats.mannwhitneyu(weds_autumn_sst20012010, weds_autumn_sst20112020, nan_policy = 'omit') # ***
#%% Separar para o cluster 4 (BRS)
brs_cluster = sst[clusters == 4,:]
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
        brs_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        brs_spring_sst20012010 = np.hstack((brs_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_sst20012010 = np.hstack((brs_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_sst20012010 = np.hstack((brs_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
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
        brs_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        brs_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        brs_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        brs_spring_sst20112020 = np.hstack((brs_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        brs_summer_sst20112020 = np.hstack((brs_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        brs_autumn_sst20112020 = np.hstack((brs_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
#%% sst - Separate betweeen 2001-2010 and 2011-2020
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

spring_boxplot = [brs_spring_sst20012010[~np.isnan(brs_spring_sst20012010)],  brs_spring_sst20112020[~np.isnan(brs_spring_sst20112020)]]
summer_boxplot = [brs_summer_sst20012010[~np.isnan(brs_summer_sst20012010)],  brs_summer_sst20112020[~np.isnan(brs_summer_sst20112020)]]
autumn_boxplot = [brs_autumn_sst20012010[~np.isnan(brs_autumn_sst20012010)],  brs_autumn_sst20112020[~np.isnan(brs_autumn_sst20112020)]]

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
plt.ylabel('SST', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\brs_sst.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(brs_spring_sst20012010, brs_spring_sst20112020, nan_policy = 'omit') #
stats.mannwhitneyu(brs_summer_sst20012010, brs_summer_sst20112020, nan_policy = 'omit') # **
stats.mannwhitneyu(brs_autumn_sst20012010, brs_autumn_sst20112020, nan_policy = 'omit') # *
#%% Separar para o cluster 5 (WEDn)
wedn_cluster = sst[clusters == 5,:]
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
        wedn_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        wedn_spring_sst20012010 = np.hstack((wedn_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_sst20012010 = np.hstack((wedn_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_sst20012010 = np.hstack((wedn_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
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
        wedn_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        wedn_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        wedn_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        wedn_spring_sst20112020 = np.hstack((wedn_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        wedn_summer_sst20112020 = np.hstack((wedn_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        wedn_autumn_sst20112020 = np.hstack((wedn_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
#%% sst - Separate betweeen 2001-2010 and 2011-2020
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

spring_boxplot = [wedn_spring_sst20012010[~np.isnan(wedn_spring_sst20012010)],  wedn_spring_sst20112020[~np.isnan(wedn_spring_sst20112020)]]
summer_boxplot = [wedn_summer_sst20012010[~np.isnan(wedn_summer_sst20012010)],  wedn_summer_sst20112020[~np.isnan(wedn_summer_sst20112020)]]
autumn_boxplot = [wedn_autumn_sst20012010[~np.isnan(wedn_autumn_sst20012010)],  wedn_autumn_sst20112020[~np.isnan(wedn_autumn_sst20112020)]]

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
plt.ylabel('SST', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\wedn_sst.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(wedn_spring_sst20012010, wedn_spring_sst20112020, nan_policy = 'omit') # ***
stats.mannwhitneyu(wedn_summer_sst20012010, wedn_summer_sst20112020, nan_policy = 'omit') # ***
stats.mannwhitneyu(wedn_autumn_sst20012010, wedn_autumn_sst20112020, nan_policy = 'omit') # ***
#%% Separar para o cluster 3 (DRA)
dra_cluster = sst[clusters == 3,:]
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
        dra_spring_sst20012010 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_sst20012010 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_sst20012010 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        dra_spring_sst20012010 = np.hstack((dra_spring_sst20012010, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_sst20012010 = np.hstack((dra_summer_sst20012010, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_sst20012010 = np.hstack((dra_autumn_sst20012010, np.hstack((yeartemp_mar, yeartemp_apr))))
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
        dra_spring_sst20112020 = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
        dra_summer_sst20112020 = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
        dra_autumn_sst20112020 = np.hstack((yeartemp_mar, yeartemp_apr))
    else:
        dra_spring_sst20112020 = np.hstack((dra_spring_sst20112020, np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))))
        dra_summer_sst20112020 = np.hstack((dra_summer_sst20112020, np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))))
        dra_autumn_sst20112020 = np.hstack((dra_autumn_sst20112020, np.hstack((yeartemp_mar, yeartemp_apr))))
#%% sst - Separate betweeen 2001-2010 and 2011-2020
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

spring_boxplot = [dra_spring_sst20012010[~np.isnan(dra_spring_sst20012010)],  dra_spring_sst20112020[~np.isnan(dra_spring_sst20112020)]]
summer_boxplot = [dra_summer_sst20012010[~np.isnan(dra_summer_sst20012010)],  dra_summer_sst20112020[~np.isnan(dra_summer_sst20112020)]]
autumn_boxplot = [dra_autumn_sst20012010[~np.isnan(dra_autumn_sst20012010)],  dra_autumn_sst20112020[~np.isnan(dra_autumn_sst20112020)]]

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
plt.ylabel('SST', fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\20012010-20112020\\dra_sst.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Calculate statistically significant differences
stats.mannwhitneyu(dra_spring_sst20012010, dra_spring_sst20112020, nan_policy = 'omit') # 
stats.mannwhitneyu(dra_summer_sst20012010, dra_summer_sst20112020, nan_policy = 'omit') # ***
stats.mannwhitneyu(dra_autumn_sst20012010, dra_autumn_sst20112020, nan_policy = 'omit') # ***






