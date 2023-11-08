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
#%% Load Variables
# Load clusters
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('antarcticpeninsula_newclusters_seaicebelow15.npz',allow_pickle = True)
clusters = fh['clusters']
# CHL
fh = np.load('chloc4so_19972021_10km.npz', allow_pickle=True)
lat_chl = fh['lat'][100:]
lon_chl = fh['lon'][30:250]
chl = fh['chl'][100:, 30:250, :]
time_date_chl = fh['time_date']
# Correct values
chl[chl > 50] = 50
# Phenology DRA
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('phenology_DRA_10km.npz', allow_pickle=True)
dra_binit = fh['b_init']
dra_bterm = fh['b_term']
dra_bpeak = fh['b_peak']
#dra_chlmax = fh['chl_max']
#dra_barea = fh['b_area']
#dra_bdur = fh['b_dur']
dra_timeyears = fh['time_years']
fh = np.load('phenology_BRS_10km.npz', allow_pickle=True)
brs_binit = fh['b_init']
brs_bterm = fh['b_term']
brs_bpeak = fh['b_peak']
#brs_chlmax = fh['chl_max']
#brs_barea = fh['b_area']
#brs_bdur = fh['b_dur']
#brs_timeyears = fh['time_years']
fh = np.load('phenology_WEDI_10km.npz', allow_pickle=True)
wedi_binit = fh['b_init']
#wedi_bterm = fh['b_term']
wedi_bpeak = fh['b_peak']
#wedi_chlmax = fh['chl_max']
#wedi_barea = fh['b_area']
#wedi_bdur = fh['b_dur']
#wedi_timeyears = fh['time_years']
fh = np.load('phenology_WEDO_10km.npz', allow_pickle=True)
wedo_binit = fh['b_init']
wedo_bterm = fh['b_term']
wedo_bpeak = fh['b_peak']
#wedo_chlmax = fh['chl_max']
#wedo_barea = fh['b_area']
#wedo_bdur = fh['b_dur']
#wedo_timeyears = fh['time_years']
fh = np.load('phenology_GES_10km.npz', allow_pickle=True)
ges_binit = fh['b_init']
ges_bterm = fh['b_term']
ges_bpeak = fh['b_peak']
#ges_chlmax = fh['chl_max']
#ges_barea = fh['b_area']
#ges_bdur = fh['b_dur']
#ges_timeyears = fh['time_years']
# SST
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia')
### Load data 1998-2020
fh = np.load('sst_19972021_10km.npz', allow_pickle=True)
lat_sst = fh['lat'][100:]
lon_sst = fh['lon'][30:250]
sst = fh['sst'][100:, 30:250, :]
time_date_sst = fh['time_date']
### Load 1981-1996
fh = np.load('sst_19811996_10km.npz', allow_pickle=True)
#lat_ = fh['lat']
#lon = fh['lon']
sst_19811996 = fh['sst'][100:, 30:250, :]
time_date_19811996 = fh['time_date']
# SEA ICE
### Load data 1998-2020
fh = np.load('seaice_19972021_10km.npz', allow_pickle=True)
lat_seaice = fh['lat'][100:]
lon_seaice = fh['lon'][30:250]
seaice = fh['seaice'][100:, 30:250, :]
seaice = seaice*100
time_date_seaice = fh['time_date']
### Load 1981-1996
fh = np.load('seaice_19811996_10km.npz', allow_pickle=True)
#lat_ = fh['lat']
#lon = fh['lon']
seaice_19811996 = fh['seaice'][100:, 30:250, :]
seaice_19811996 = seaice_19811996*100
#time_date_19811996 = fh['time_date']
#%% WEDi
leapyears_list = [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
## Sea ice
seaice_wedi_19972021 = seaice[clusters == 1,:]
seaice_wedi_19972021 = np.nanmean(seaice_wedi_19972021,0)
seaice_wedi_19811996 = seaice_19811996[clusters == 1,:]
seaice_wedi_19811996 = np.nanmean(seaice_wedi_19811996,0)
seaice_wedi_19812021 = np.hstack((seaice_wedi_19811996, seaice_wedi_19972021))
time_date_seaice_19812021 = np.hstack((time_date_19811996, time_date_seaice))
time_date_years = np.empty_like(time_date_seaice_19812021)
time_date_months = np.empty_like(time_date_seaice_19812021)
time_date_days = np.empty_like(time_date_seaice_19812021)
for i in range(0, len(time_date_seaice_19812021)):
    time_date_seaice_19812021[i] = datetime.datetime(year=time_date_seaice_19812021[i].year,
                                                  month=time_date_seaice_19812021[i].month,
                                                  day=time_date_seaice_19812021[i].day)
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day
for i in np.arange(1997, 2021):
    # Extract august to may
    seaice_wedi_sep = seaice_wedi_19812021[(time_date_years == i) & (time_date_months == 9)]
    seaice_wedi_oct = seaice_wedi_19812021[(time_date_years == i) & (time_date_months == 10)]
    seaice_wedi_nov = seaice_wedi_19812021[(time_date_years == i) & (time_date_months == 11)]
    seaice_wedi_spring_all = np.hstack((seaice_wedi_sep, seaice_wedi_oct, seaice_wedi_nov))
    seaice_wedi_spring_temp = np.nanmean(seaice_wedi_spring_all)
    if i == 1997:
        seaice_wedi_spring = seaice_wedi_spring_temp
    else:
        seaice_wedi_spring = np.vstack((seaice_wedi_spring, seaice_wedi_spring_temp))
#%% WEDo
## Sea ice
seaice_wedo_19972021 = seaice[clusters == 5,:]
seaice_wedo_19972021 = np.nanmean(seaice_wedo_19972021,0)
seaice_wedo_19811996 = seaice_19811996[clusters == 5,:]
seaice_wedo_19811996 = np.nanmean(seaice_wedo_19811996,0)
seaice_wedo_19812021 = np.hstack((seaice_wedo_19811996, seaice_wedo_19972021))
time_date_seaice_19812021 = np.hstack((time_date_19811996, time_date_seaice))
time_date_years = np.empty_like(time_date_seaice_19812021)
time_date_months = np.empty_like(time_date_seaice_19812021)
time_date_days = np.empty_like(time_date_seaice_19812021)
for i in range(0, len(time_date_seaice_19812021)):
    time_date_seaice_19812021[i] = datetime.datetime(year=time_date_seaice_19812021[i].year,
                                                  month=time_date_seaice_19812021[i].month,
                                                  day=time_date_seaice_19812021[i].day)
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day

for i in np.arange(1997, 2021):
    # Extract august to may
    seaice_wedo_sep = seaice_wedo_19812021[(time_date_years == i) & (time_date_months == 9)]
    seaice_wedo_oct = seaice_wedo_19812021[(time_date_years == i) & (time_date_months == 10)]
    seaice_wedo_nov = seaice_wedo_19812021[(time_date_years == i) & (time_date_months == 11)]
    seaice_wedo_spring_all = np.hstack((seaice_wedo_sep, seaice_wedo_oct, seaice_wedo_nov))
    seaice_wedo_spring_temp = np.nanmean(seaice_wedo_spring_all)
    if i == 1997:
        seaice_wedo_spring = seaice_wedo_spring_temp
    else:
        seaice_wedo_spring = np.vstack((seaice_wedo_spring, seaice_wedo_spring_temp))

#%% GES
## Sea ice
seaice_ges_19972021 = seaice[clusters == 2,:]
seaice_ges_19972021 = np.nanmean(seaice_ges_19972021,0)
seaice_ges_19811996 = seaice_19811996[clusters == 2,:]
seaice_ges_19811996 = np.nanmean(seaice_ges_19811996,0)
seaice_ges_19812021 = np.hstack((seaice_ges_19811996, seaice_ges_19972021))
time_date_seaice_19812021 = np.hstack((time_date_19811996, time_date_seaice))
time_date_years = np.empty_like(time_date_seaice_19812021)
time_date_months = np.empty_like(time_date_seaice_19812021)
time_date_days = np.empty_like(time_date_seaice_19812021)
for i in range(0, len(time_date_seaice_19812021)):
    time_date_seaice_19812021[i] = datetime.datetime(year=time_date_seaice_19812021[i].year,
                                                  month=time_date_seaice_19812021[i].month,
                                                  day=time_date_seaice_19812021[i].day)
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day
for i in np.arange(1997, 2021):
    # Extract august to may
    seaice_ges_sep = seaice_ges_19812021[(time_date_years == i) & (time_date_months == 9)]
    seaice_ges_oct = seaice_ges_19812021[(time_date_years == i) & (time_date_months == 10)]
    seaice_ges_nov = seaice_ges_19812021[(time_date_years == i) & (time_date_months == 11)]
    seaice_ges_spring_all = np.hstack((seaice_ges_sep, seaice_ges_oct, seaice_ges_nov))
    seaice_ges_spring_temp = np.nanmean(seaice_ges_spring_all)
    if i == 1997:
        seaice_ges_spring = seaice_ges_spring_temp
    else:
        seaice_ges_spring = np.vstack((seaice_ges_spring, seaice_ges_spring_temp))

#%% DRA
## Sea ice
seaice_dra_19972021 = seaice[clusters == 3,:]
seaice_dra_19972021 = np.nanmean(seaice_dra_19972021,0)
seaice_dra_19811996 = seaice_19811996[clusters == 3,:]
seaice_dra_19811996 = np.nanmean(seaice_dra_19811996,0)
seaice_dra_19812021 = np.hstack((seaice_dra_19811996, seaice_dra_19972021))
time_date_seaice_19812021 = np.hstack((time_date_19811996, time_date_seaice))
time_date_years = np.empty_like(time_date_seaice_19812021)
time_date_months = np.empty_like(time_date_seaice_19812021)
time_date_days = np.empty_like(time_date_seaice_19812021)
for i in range(0, len(time_date_seaice_19812021)):
    time_date_seaice_19812021[i] = datetime.datetime(year=time_date_seaice_19812021[i].year,
                                                  month=time_date_seaice_19812021[i].month,
                                                  day=time_date_seaice_19812021[i].day)
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day

for i in np.arange(1997, 2021):
    # Extract august to may
    seaice_dra_sep = seaice_dra_19812021[(time_date_years == i) & (time_date_months == 9)]
    seaice_dra_oct = seaice_dra_19812021[(time_date_years == i) & (time_date_months == 10)]
    seaice_dra_nov = seaice_dra_19812021[(time_date_years == i) & (time_date_months == 11)]
    seaice_dra_spring_all = np.hstack((seaice_dra_sep, seaice_dra_oct, seaice_dra_nov))
    seaice_dra_spring_temp = np.nanmean(seaice_dra_spring_all)
    if i == 1997:
        seaice_dra_spring = seaice_dra_spring_temp
    else:
        seaice_dra_spring = np.vstack((seaice_dra_spring, seaice_dra_spring_temp))

#%% BRS
## Sea ice
seaice_brs_19972021 = seaice[clusters == 4,:]
seaice_brs_19972021 = np.nanmean(seaice_brs_19972021,0)
seaice_brs_19811996 = seaice_19811996[clusters == 4,:]
seaice_brs_19811996 = np.nanmean(seaice_brs_19811996,0)
seaice_brs_19812021 = np.hstack((seaice_brs_19811996, seaice_brs_19972021))
time_date_seaice_19812021 = np.hstack((time_date_19811996, time_date_seaice))
time_date_years = np.empty_like(time_date_seaice_19812021)
time_date_months = np.empty_like(time_date_seaice_19812021)
time_date_days = np.empty_like(time_date_seaice_19812021)
for i in range(0, len(time_date_seaice_19812021)):
    time_date_seaice_19812021[i] = datetime.datetime(year=time_date_seaice_19812021[i].year,
                                                  month=time_date_seaice_19812021[i].month,
                                                  day=time_date_seaice_19812021[i].day)
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day
for i in np.arange(1997, 2021):
    # Extract august to may
    seaice_brs_sep = seaice_brs_19812021[(time_date_years == i) & (time_date_months == 9)]
    seaice_brs_oct = seaice_brs_19812021[(time_date_years == i) & (time_date_months == 10)]
    seaice_brs_nov = seaice_brs_19812021[(time_date_years == i) & (time_date_months == 11)]
    seaice_brs_spring_all = np.hstack((seaice_brs_sep, seaice_brs_oct, seaice_brs_nov))
    seaice_brs_spring_temp = np.nanmean(seaice_brs_spring_all)
    if i == 1997:
        seaice_brs_spring = seaice_brs_spring_temp
    else:
        seaice_brs_spring = np.vstack((seaice_brs_spring, seaice_brs_spring_temp))
#%% Create two vectors with Sea Ice Spring and Bloom Peak
# Order: DRA, BRS, WEDO, GES, WEDI
seaice_spring_allregions = np.hstack((seaice_dra_spring.ravel(), seaice_brs_spring.ravel(), seaice_wedo_spring.ravel(),
                                      seaice_ges_spring.ravel(), seaice_wedi_spring.ravel()))
bloompeak_allregions = np.hstack((dra_bpeak, brs_bpeak, wedo_bpeak, ges_bpeak, wedi_bpeak))
plt.scatter(seaice_spring_allregions, bloompeak_allregions)
# Without WEDO
seaice_spring_withoutwedo = np.hstack((seaice_dra_spring.ravel(), seaice_brs_spring.ravel(),
                                      seaice_ges_spring.ravel(), seaice_wedi_spring.ravel()))
bloompeak_withoutwedo = np.hstack((dra_bpeak, brs_bpeak, ges_bpeak, wedi_bpeak))
plt.scatter(seaice_spring_withoutwedo, bloompeak_withoutwedo)
# Without BRS
seaice_spring_withoutbrs = np.hstack((seaice_dra_spring.ravel(), seaice_wedo_spring.ravel(),
                                      seaice_ges_spring.ravel(), seaice_wedi_spring.ravel()))
bloompeak_withoutbrs = np.hstack((dra_bpeak, wedo_bpeak, ges_bpeak, wedi_bpeak))
plt.scatter(seaice_spring_withoutbrs, bloompeak_withoutbrs)

stats.linregress(seaice_spring_allregions, bloompeak_allregions)
stats.linregress(seaice_spring_withoutbrs, bloompeak_withoutbrs)
#%% Create two vectors with Sea Ice Spring and Bloom Init
# Order: DRA, BRS, WEDO, GES, WEDI
seaice_spring_allregions = np.hstack((seaice_dra_spring.ravel(), seaice_brs_spring.ravel(), seaice_wedo_spring.ravel(),
                                      seaice_ges_spring.ravel(), seaice_wedi_spring.ravel()))
bloominit_allregions = np.hstack((dra_binit, brs_binit, wedo_binit, ges_binit, wedi_binit))
plt.scatter(seaice_spring_allregions, bloominit_allregions)
# Without WEDO
seaice_spring_withoutwedo = np.hstack((seaice_dra_spring.ravel(), seaice_brs_spring.ravel(),
                                      seaice_ges_spring.ravel(), seaice_wedi_spring.ravel()))
bloominit_withoutwedo = np.hstack((dra_binit, brs_binit, ges_binit, wedi_binit))
plt.scatter(seaice_spring_withoutwedo, bloominit_withoutwedo)
# Without BRS
seaice_spring_withoutbrs = np.hstack((seaice_dra_spring.ravel(), seaice_wedo_spring.ravel(),
                                      seaice_ges_spring.ravel(), seaice_wedi_spring.ravel()))
bloominit_withoutbrs = np.hstack((dra_binit, wedo_binit, ges_binit, wedi_binit))
plt.scatter(seaice_spring_withoutbrs, bloominit_withoutbrs)

#stats.linregress(seaice_spring_allregions, bloominit_allregions)
slope_binit, intercept_binit, rvalue_binit, pvalue_binit, _ = stats.linregress(seaice_spring_withoutbrs, bloominit_withoutbrs)
slope_bpeak, intercept_bpeak, rvalue_bpeak, pvalue_bpeak, _ = stats.linregress(seaice_spring_withoutbrs, bloompeak_withoutbrs)

#%%
fig, axs = plt.subplots(1, 1, figsize=(5,3.5))
# Bloom Init
axs.scatter(seaice_dra_spring.ravel(), dra_binit, s=35, c=[243/256, 167/256, 18/256, 1], marker='^', alpha=0.8, label='DRA')
axs.scatter(seaice_brs_spring.ravel(), brs_binit, s=30, c=[106/256, 153/256, 78/256, 1], marker='s', alpha=0.5, label='BRS')
axs.scatter(seaice_wedo_spring.ravel(), wedo_binit, s=25, c=[83/256, 77/256, 65/256, 1], marker='D', alpha=0.8, label='WEDo')
axs.scatter(seaice_ges_spring.ravel(), ges_binit, s=30, c=[41/256, 51/256, 92/256, 1], marker='*', alpha=0.8, label='GES')
axs.scatter(seaice_wedi_spring.ravel(), wedi_binit, s=30, c=[219/256, 43/256, 57/256, 1], marker='o', alpha=0.8, label='WEDi')
axs.set_ylabel('Bloom Initiation Date', fontsize=14)
axs.set_xlabel('Spring Sea Ice Concentration (%)', fontsize=14)
axs.set_yticks(ticks= [5, 9, 13, 17, 21, 24], labels=['SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB'], fontsize=14)
axs.tick_params(axis='x', labelsize=14)
axs.plot(seaice_spring_withoutbrs, seaice_spring_withoutbrs*slope_binit+intercept_binit, c='k')
axs.legend(loc=1, ncol=3, labelspacing=0.2, handletextpad=.2, fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\updatedclusters_environmental\\Binit.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()











#%% Separate and calculate changes to yearly cycles
chl_sepapr_19982003 = np.nanmean(chl_sepapr_19982021[:6, :],0)
chl_sepapr_20042009 = np.nanmean(chl_sepapr_19982021[6:12, :],0)
chl_sepapr_20102015 = np.nanmean(chl_sepapr_19982021[12:18, :],0)
chl_sepapr_20162021 = np.nanmean(chl_sepapr_19982021[18:, :],0)
chl_sepapr_19982005 = np.nanmean(chl_sepapr_19982021[:8, :],0)
chl_sepapr_20062021 = np.nanmean(chl_sepapr_19982021[8:, :],0)
# Plot Chl
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982003, marker='^', c='g', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20042009, marker='s', c='g', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanic_sepapr_climatology_comparison5years.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982005, marker='^', c='g', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20062021, marker='s', c='g', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanic_sepapr_climatology_comparison_prepost2005.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
par_sepapr_19982003 = np.nanmean(par_sepapr_19982021[:6, :],0)
par_sepapr_20042009 = np.nanmean(par_sepapr_19982021[6:12, :],0)
par_sepapr_20102015 = np.nanmean(par_sepapr_19982021[12:18, :],0)
par_sepapr_20162021 = np.nanmean(par_sepapr_19982021[18:, :],0)
par_sepapr_19982005 = np.nanmean(par_sepapr_19982021[:8, :],0)
par_sepapr_20062021 = np.nanmean(par_sepapr_19982021[8:, :],0)
# Plot par
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982003, marker='^', c='y', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20042009, marker='s', c='y', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='y', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='y', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanic_sepapr_climatology_comparison5years_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982005, marker='^', c='y', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20062021, marker='s', c='y', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanic_sepapr_climatology_comparison_prepost2005_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
sst_sepapr_19821991 = np.nanmean(sst_sepapr_19982021[:10, :],0)
sst_sepapr_19922001 = np.nanmean(sst_sepapr_19982021[10:20, :],0)
sst_sepapr_20022011 = np.nanmean(sst_sepapr_19982021[20:30, :],0)
sst_sepapr_20122021 = np.nanmean(sst_sepapr_19982021[30:, :],0)
sst_sepapr_19822005 = np.nanmean(sst_sepapr_19982021[:24, :],0)
sst_sepapr_20062021 =  np.nanmean(sst_sepapr_19982021[24:, :],0)
# Plot sst
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19821991, marker='^', c='r', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_19922001, marker='s', c='r', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20022011, linestyle='--', c='r', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20122021, marker='*', linestyle=':', c='r', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanic_sepapr_climatology_comparison10years_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19822005, marker='^', c='r', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20062021, marker='s', c='r', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanic_sepapr_climatology_comparison_prepost2005_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
seaice_sepapr_19821991 = np.nanmean(seaice_sepapr_19982021[:10, :],0)
seaice_sepapr_19922001 = np.nanmean(seaice_sepapr_19982021[10:20, :],0)
seaice_sepapr_20022011 = np.nanmean(seaice_sepapr_19982021[20:30, :],0)
seaice_sepapr_20122021 = np.nanmean(seaice_sepapr_19982021[30:, :],0)
seaice_sepapr_19822005 = np.nanmean(seaice_sepapr_19982021[:24, :],0)
seaice_sepapr_20062021 =  np.nanmean(seaice_sepapr_19982021[24:, :],0)
# Plot seaice
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19821991, marker='^', c='grey', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_19922001, marker='s', c='grey', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20022011, linestyle='--', c='grey', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20122021, marker='*', linestyle=':', c='grey', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanic_sepapr_climatology_comparison10years_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19822005, marker='^', c='grey', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20062021, marker='s', c='grey', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('seaice (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanic_sepapr_climatology_comparison_prepost2005_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% GERLACHE
leapyears_list = [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
## CHL #1998-2021
chl_gerlache = chl[clusters == 2,:]
chl_gerlache = np.nanmean(chl_gerlache,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = chl_gerlache[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_chl[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 5))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = chl_gerlache[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_chl[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1998:
        chl_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        chl_sepapr_19982021 = np.vstack((chl_sepapr_19982021, yeartemp_augmay_pd_8day.values))
chl_sepapr_19982021_mean = np.nanmean(chl_sepapr_19982021,0)
## SST
sst_gerlache_19972021 = sst[clusters == 2,:]
sst_gerlache_19972021 = np.nanmean(sst_gerlache_19972021,0)
sst_gerlache_19811996 = sst_19811996[clusters == 2,:]
sst_gerlache_19811996 = np.nanmean(sst_gerlache_19811996,0)
sst_gerlache_19812021 = np.hstack((sst_gerlache_19811996, sst_gerlache_19972021))
time_date_sst_19812021 = np.hstack((time_date_19811996, time_date_sst))
time_date_years = np.empty_like(time_date_sst_19812021)
time_date_months = np.empty_like(time_date_sst_19812021)
time_date_days = np.empty_like(time_date_sst_19812021)
for i in range(0, len(time_date_sst_19812021)):
    time_date_sst_19812021[i] = datetime.datetime(year=time_date_sst_19812021[i].year,
                                                  month=time_date_sst_19812021[i].month,
                                                  day=time_date_sst_19812021[i].day)
    time_date_years[i] = time_date_sst_19812021[i].year
    time_date_months[i] = time_date_sst_19812021[i].month
    time_date_days[i] = time_date_sst_19812021[i].day

for i in np.arange(1982, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1982:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = sst_gerlache_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_sst_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = sst_gerlache_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_sst_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1982:
        sst_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        sst_sepapr_19982021 = np.vstack((sst_sepapr_19982021, yeartemp_augmay_pd_8day.values))
sst_sepapr_19982021_mean = np.nanmean(sst_sepapr_19982021,0)
## Sea ice
seaice_gerlache_19972021 = seaice[clusters == 2,:]
seaice_gerlache_19972021 = np.nanmean(seaice_gerlache_19972021,0)
seaice_gerlache_19811996 = seaice_19811996[clusters == 2,:]
seaice_gerlache_19811996 = np.nanmean(seaice_gerlache_19811996,0)
seaice_gerlache_19812021 = np.hstack((seaice_gerlache_19811996, seaice_gerlache_19972021))
time_date_seaice_19812021 = np.hstack((time_date_19811996, time_date_seaice))
time_date_years = np.empty_like(time_date_seaice_19812021)
time_date_months = np.empty_like(time_date_seaice_19812021)
time_date_days = np.empty_like(time_date_seaice_19812021)
for i in range(0, len(time_date_seaice_19812021)):
    time_date_seaice_19812021[i] = datetime.datetime(year=time_date_seaice_19812021[i].year,
                                                  month=time_date_seaice_19812021[i].month,
                                                  day=time_date_seaice_19812021[i].day)
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day

for i in np.arange(1982, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1982:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = seaice_gerlache_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_seaice_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = seaice_gerlache_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_seaice_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1982:
        seaice_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        seaice_sepapr_19982021 = np.vstack((seaice_sepapr_19982021, yeartemp_augmay_pd_8day.values))
seaice_sepapr_19982021_mean = np.nanmean(seaice_sepapr_19982021,0)
## Sea ice advance
seaice_advanceday_19812021_mean = np.round(np.nanmean(seaice_advanceday.values),0)
## Sea ice retreat
seaice_retreatday_19812021_mean = np.round(np.nanmean(seaice_retreatday.values),0)
## PAR
par_gerlache = par[clusters == 2,:]
par_gerlache = np.nanmean(par_gerlache,0)
par_gerlache_df = pd.Series(data=par_gerlache, index=time_date_par)
par_gerlache_df = par_gerlache_df.resample('D').mean()
par_gerlache = par_gerlache_df.values
time_date_par_daily = par_gerlache_df.index
time_date_years = np.empty_like(time_date_par_daily)
time_date_months = np.empty_like(time_date_par_daily)
#for i in range(0, len(time_date_par_daily)):
#    time_date_years[i] = time_date_par_daily.year
#    time_date_months[i] = time_date_par_daily[i].month
for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_par_daily.year == i) & (time_date_par_daily.month == 6))[-1][-1]
        yeartemp_augmay = par_gerlache[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_par_daily[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_par_daily.year == i-1) & (time_date_par_daily.month == 6))[0][0]
        yeartemp_may = np.where((time_date_par_daily.year == i) & (time_date_par_daily.month == 6))[-1][-1]
        yeartemp_augmay = par_gerlache[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_par_daily[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1998:
        par_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        par_sepapr_19982021 = np.vstack((par_sepapr_19982021, yeartemp_augmay_pd_8day.values))
par_sepapr_19982021_mean = np.nanmean(par_sepapr_19982021,0)
#%% Plot figure
# SEAICE CONC
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982021_mean, marker='^', c='g', label = 'Chla', markersize=5, alpha=.5, zorder=10)
#axs.set_zorder(10)
ax2 = axs.twinx()
ax2.plot(np.arange(1,43), sst_sepapr_19982021_mean, marker='s', c='r', label = 'SST', markersize=5, alpha=.5,zorder=9)
ax3 = axs.twinx()
ax3.plot(np.arange(1,43), seaice_sepapr_19982021_mean, linestyle='-', c='grey', label = 'Sea Ice', markersize=5, alpha=.5)
ax4 = axs.twinx()
ax4.plot(np.arange(1,43), par_sepapr_19982021_mean, marker='*', c='y', label = 'PAR', markersize=5, alpha=.5)
#ax5 = axs.twinx()
#ax5.plot(np.arange(1,43), windspeed_sepapr_19982021_mean, marker='.', c='purple', label = 'Wind Speed', markersize=5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2.spines['right'].set_color('r')
ax3.spines['right'].set_position(('outward', 60))
ax3.yaxis.label.set_color('grey')
ax3.tick_params(axis='y', colors='grey')
ax3.spines['right'].set_color('grey')
ax4.spines['right'].set_position(('outward', 110))
ax4.yaxis.label.set_color('y')
ax4.tick_params(axis='y', colors='y')
ax4.spines['right'].set_color('y')
#ax5.spines['right'].set_position(('outward', 160))
#ax5.yaxis.label.set_color('purple')
#ax5.tick_params(axis='y', colors='purple')
#ax5.spines['right'].set_color('purple')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
ax2.set_ylabel('SST (°C)', fontsize=12)
ax3.set_ylabel('Sea Ice (%)', fontsize=12)
ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax5.set_ylabel('Wind Speed (ms$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
plt.axvline(21, linestyle='--', c='grey', alpha=0.3)
plt.axvline(38, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1.15), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'gerlache_sepapr_climatology.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
chl_sepapr_19982003 = np.nanmean(chl_sepapr_19982021[:6, :],0)
chl_sepapr_20042009 = np.nanmean(chl_sepapr_19982021[6:12, :],0)
chl_sepapr_20102015 = np.nanmean(chl_sepapr_19982021[12:18, :],0)
chl_sepapr_20162021 = np.nanmean(chl_sepapr_19982021[18:, :],0)
chl_sepapr_19982005 = np.nanmean(chl_sepapr_19982021[:8, :],0)
chl_sepapr_20062021 = np.nanmean(chl_sepapr_19982021[8:, :],0)
# Plot Chl
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982003, marker='^', c='g', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20042009, marker='s', c='g', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison5years.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982005, marker='^', c='g', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20062021, marker='s', c='g', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison_prepost2005.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
par_sepapr_19982003 = np.nanmean(par_sepapr_19982021[:6, :],0)
par_sepapr_20042009 = np.nanmean(par_sepapr_19982021[6:12, :],0)
par_sepapr_20102015 = np.nanmean(par_sepapr_19982021[12:18, :],0)
par_sepapr_20162021 = np.nanmean(par_sepapr_19982021[18:, :],0)
par_sepapr_19982005 = np.nanmean(par_sepapr_19982021[:8, :],0)
par_sepapr_20062021 = np.nanmean(par_sepapr_19982021[8:, :],0)
# Plot par
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982003, marker='^', c='y', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20042009, marker='s', c='y', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='y', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='y', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison5years_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982005, marker='^', c='y', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20062021, marker='s', c='y', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison_prepost2005_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
sst_sepapr_19821991 = np.nanmean(sst_sepapr_19982021[:10, :],0)
sst_sepapr_19922001 = np.nanmean(sst_sepapr_19982021[10:20, :],0)
sst_sepapr_20022011 = np.nanmean(sst_sepapr_19982021[20:30, :],0)
sst_sepapr_20122021 = np.nanmean(sst_sepapr_19982021[30:, :],0)
sst_sepapr_19822005 = np.nanmean(sst_sepapr_19982021[:24, :],0)
sst_sepapr_20062021 =  np.nanmean(sst_sepapr_19982021[24:, :],0)
# Plot sst
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19821991, marker='^', c='r', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_19922001, marker='s', c='r', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20022011, linestyle='--', c='r', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20122021, marker='*', linestyle=':', c='r', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison10years_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19822005, marker='^', c='r', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20062021, marker='s', c='r', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison_prepost2005_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
seaice_sepapr_19821991 = np.nanmean(seaice_sepapr_19982021[:10, :],0)
seaice_sepapr_19922001 = np.nanmean(seaice_sepapr_19982021[10:20, :],0)
seaice_sepapr_20022011 = np.nanmean(seaice_sepapr_19982021[20:30, :],0)
seaice_sepapr_20122021 = np.nanmean(seaice_sepapr_19982021[30:, :],0)
seaice_sepapr_19822005 = np.nanmean(seaice_sepapr_19982021[:24, :],0)
seaice_sepapr_20062021 =  np.nanmean(seaice_sepapr_19982021[24:, :],0)
# Plot seaice
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19821991, marker='^', c='grey', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_19922001, marker='s', c='grey', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20022011, linestyle='--', c='grey', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20122021, marker='*', linestyle=':', c='grey', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison10years_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19822005, marker='^', c='grey', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20062021, marker='s', c='grey', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('seaice (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison_prepost2005_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% WEDDELL
leapyears_list = [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
## CHL #1998-2021
chl_weddell = chl[clusters == 1,:]
chl_weddell = np.nanmean(chl_weddell,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = chl_weddell[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_chl[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 5))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = chl_weddell[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_chl[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1998:
        chl_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        chl_sepapr_19982021 = np.vstack((chl_sepapr_19982021, yeartemp_augmay_pd_8day.values))
chl_sepapr_19982021_mean = np.nanmean(chl_sepapr_19982021,0)
## SST
sst_weddell_19972021 = sst[clusters == 1,:]
sst_weddell_19972021 = np.nanmean(sst_weddell_19972021,0)
sst_weddell_19811996 = sst_19811996[clusters == 1,:]
sst_weddell_19811996 = np.nanmean(sst_weddell_19811996,0)
sst_weddell_19812021 = np.hstack((sst_weddell_19811996, sst_weddell_19972021))
time_date_sst_19812021 = np.hstack((time_date_19811996, time_date_sst))
time_date_years = np.empty_like(time_date_sst_19812021)
time_date_months = np.empty_like(time_date_sst_19812021)
time_date_days = np.empty_like(time_date_sst_19812021)
for i in range(0, len(time_date_sst_19812021)):
    time_date_sst_19812021[i] = datetime.datetime(year=time_date_sst_19812021[i].year,
                                                  month=time_date_sst_19812021[i].month,
                                                  day=time_date_sst_19812021[i].day)
    time_date_years[i] = time_date_sst_19812021[i].year
    time_date_months[i] = time_date_sst_19812021[i].month
    time_date_days[i] = time_date_sst_19812021[i].day

for i in np.arange(1982, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1982:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = sst_weddell_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_sst_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = sst_weddell_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_sst_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1982:
        sst_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        sst_sepapr_19982021 = np.vstack((sst_sepapr_19982021, yeartemp_augmay_pd_8day.values))
sst_sepapr_19982021_mean = np.nanmean(sst_sepapr_19982021,0)
## Sea ice
seaice_weddell_19972021 = seaice[clusters == 1,:]
seaice_weddell_19972021 = np.nanmean(seaice_weddell_19972021,0)
seaice_weddell_19811996 = seaice_19811996[clusters == 1,:]
seaice_weddell_19811996 = np.nanmean(seaice_weddell_19811996,0)
seaice_weddell_19812021 = np.hstack((seaice_weddell_19811996, seaice_weddell_19972021))
time_date_seaice_19812021 = np.hstack((time_date_19811996, time_date_seaice))
time_date_years = np.empty_like(time_date_seaice_19812021)
time_date_months = np.empty_like(time_date_seaice_19812021)
time_date_days = np.empty_like(time_date_seaice_19812021)
for i in range(0, len(time_date_seaice_19812021)):
    time_date_seaice_19812021[i] = datetime.datetime(year=time_date_seaice_19812021[i].year,
                                                  month=time_date_seaice_19812021[i].month,
                                                  day=time_date_seaice_19812021[i].day)
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day

for i in np.arange(1982, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1982:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = seaice_weddell_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_seaice_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = seaice_weddell_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_seaice_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1982:
        seaice_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        seaice_sepapr_19982021 = np.vstack((seaice_sepapr_19982021, yeartemp_augmay_pd_8day.values))
seaice_sepapr_19982021_mean = np.nanmean(seaice_sepapr_19982021,0)
## Sea ice advance
seaice_advanceday_19812021_mean = np.round(np.nanmean(seaice_advanceday.values),0)
## Sea ice retreat
seaice_retreatday_19812021_mean = np.round(np.nanmean(seaice_retreatday.values),0)
## PAR
par_weddell = par[clusters == 1,:]
par_weddell = np.nanmean(par_weddell,0)
par_weddell_df = pd.Series(data=par_weddell, index=time_date_par)
par_weddell_df = par_weddell_df.resample('D').mean()
par_weddell = par_weddell_df.values
time_date_par_daily = par_weddell_df.index
time_date_years = np.empty_like(time_date_par_daily)
time_date_months = np.empty_like(time_date_par_daily)
#for i in range(0, len(time_date_par_daily)):
#    time_date_years[i] = time_date_par_daily.year
#    time_date_months[i] = time_date_par_daily[i].month
for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_par_daily.year == i) & (time_date_par_daily.month == 6))[-1][-1]
        yeartemp_augmay = par_weddell[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_par_daily[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_par_daily.year == i-1) & (time_date_par_daily.month == 6))[0][0]
        yeartemp_may = np.where((time_date_par_daily.year == i) & (time_date_par_daily.month == 6))[-1][-1]
        yeartemp_augmay = par_weddell[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_par_daily[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1998:
        par_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        par_sepapr_19982021 = np.vstack((par_sepapr_19982021, yeartemp_augmay_pd_8day.values))
par_sepapr_19982021_mean = np.nanmean(par_sepapr_19982021,0)
#%% Plot figure
# SEAICE CONC
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982021_mean, marker='^', c='g', label = 'Chla', markersize=5, alpha=.5, zorder=10)
#axs.set_zorder(10)
ax2 = axs.twinx()
ax2.plot(np.arange(1,43), sst_sepapr_19982021_mean, marker='s', c='r', label = 'SST', markersize=5, alpha=.5,zorder=9)
ax3 = axs.twinx()
ax3.plot(np.arange(1,43), seaice_sepapr_19982021_mean, linestyle='-', c='grey', label = 'Sea Ice', markersize=5, alpha=.5)
ax4 = axs.twinx()
ax4.plot(np.arange(1,43), par_sepapr_19982021_mean, marker='*', c='y', label = 'PAR', markersize=5, alpha=.5)
#ax5 = axs.twinx()
#ax5.plot(np.arange(1,43), windspeed_sepapr_19982021_mean, marker='.', c='purple', label = 'Wind Speed', markersize=5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2.spines['right'].set_color('r')
ax3.spines['right'].set_position(('outward', 60))
ax3.yaxis.label.set_color('grey')
ax3.tick_params(axis='y', colors='grey')
ax3.spines['right'].set_color('grey')
ax4.spines['right'].set_position(('outward', 110))
ax4.yaxis.label.set_color('y')
ax4.tick_params(axis='y', colors='y')
ax4.spines['right'].set_color('y')
#ax5.spines['right'].set_position(('outward', 160))
#ax5.yaxis.label.set_color('purple')
#ax5.tick_params(axis='y', colors='purple')
#ax5.spines['right'].set_color('purple')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
ax2.set_ylabel('SST (°C)', fontsize=12)
ax3.set_ylabel('Sea Ice (%)', fontsize=12)
ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax5.set_ylabel('Wind Speed (ms$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(13, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(32, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1.15), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'weddell_sepapr_climatology.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
chl_sepapr_19982003 = np.nanmean(chl_sepapr_19982021[:6, :],0)
chl_sepapr_20042009 = np.nanmean(chl_sepapr_19982021[6:12, :],0)
chl_sepapr_20102015 = np.nanmean(chl_sepapr_19982021[12:18, :],0)
chl_sepapr_20162021 = np.nanmean(chl_sepapr_19982021[18:, :],0)
chl_sepapr_19982005 = np.nanmean(chl_sepapr_19982021[:8, :],0)
chl_sepapr_20062021 = np.nanmean(chl_sepapr_19982021[8:, :],0)
# Plot Chl
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982003, marker='^', c='g', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20042009, marker='s', c='g', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'weddell_sepapr_climatology_comparison5years.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982005, marker='^', c='g', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20062021, marker='s', c='g', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'weddell_sepapr_climatology_comparison_prepost2005.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
par_sepapr_19982003 = np.nanmean(par_sepapr_19982021[:6, :],0)
par_sepapr_20042009 = np.nanmean(par_sepapr_19982021[6:12, :],0)
par_sepapr_20102015 = np.nanmean(par_sepapr_19982021[12:18, :],0)
par_sepapr_20162021 = np.nanmean(par_sepapr_19982021[18:, :],0)
par_sepapr_19982005 = np.nanmean(par_sepapr_19982021[:8, :],0)
par_sepapr_20062021 = np.nanmean(par_sepapr_19982021[8:, :],0)
# Plot par
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982003, marker='^', c='y', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20042009, marker='s', c='y', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='y', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='y', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'weddell_sepapr_climatology_comparison5years_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982005, marker='^', c='y', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20062021, marker='s', c='y', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'weddell_sepapr_climatology_comparison_prepost2005_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
sst_sepapr_19821991 = np.nanmean(sst_sepapr_19982021[:10, :],0)
sst_sepapr_19922001 = np.nanmean(sst_sepapr_19982021[10:20, :],0)
sst_sepapr_20022011 = np.nanmean(sst_sepapr_19982021[20:30, :],0)
sst_sepapr_20122021 = np.nanmean(sst_sepapr_19982021[30:, :],0)
sst_sepapr_19822005 = np.nanmean(sst_sepapr_19982021[:24, :],0)
sst_sepapr_20062021 =  np.nanmean(sst_sepapr_19982021[24:, :],0)
# Plot sst
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19821991, marker='^', c='r', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_19922001, marker='s', c='r', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20022011, linestyle='--', c='r', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20122021, marker='*', linestyle=':', c='r', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'weddell_sepapr_climatology_comparison10years_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19822005, marker='^', c='r', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20062021, marker='s', c='r', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'weddell_sepapr_climatology_comparison_prepost2005_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
seaice_sepapr_19821991 = np.nanmean(seaice_sepapr_19982021[:10, :],0)
seaice_sepapr_19922001 = np.nanmean(seaice_sepapr_19982021[10:20, :],0)
seaice_sepapr_20022011 = np.nanmean(seaice_sepapr_19982021[20:30, :],0)
seaice_sepapr_20122021 = np.nanmean(seaice_sepapr_19982021[30:, :],0)
seaice_sepapr_19822005 = np.nanmean(seaice_sepapr_19982021[:24, :],0)
seaice_sepapr_20062021 =  np.nanmean(seaice_sepapr_19982021[24:, :],0)
# Plot seaice
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19821991, marker='^', c='grey', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_19922001, marker='s', c='grey', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20022011, linestyle='--', c='grey', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20122021, marker='*', linestyle=':', c='grey', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'weddell_sepapr_climatology_comparison10years_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19822005, marker='^', c='grey', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20062021, marker='s', c='grey', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('seaice (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'weddell_sepapr_climatology_comparison_prepost2005_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% BRANSFIELD
leapyears_list = [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
## CHL #1998-2021
chl_bransfield = chl[clusters == 4,:]
chl_bransfield = np.nanmean(chl_bransfield,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = chl_bransfield[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_chl[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 5))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = chl_bransfield[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_chl[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1998:
        chl_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        chl_sepapr_19982021 = np.vstack((chl_sepapr_19982021, yeartemp_augmay_pd_8day.values))
chl_sepapr_19982021_mean = np.nanmean(chl_sepapr_19982021,0)
## SST
sst_bransfield_19972021 = sst[clusters == 4,:]
sst_bransfield_19972021 = np.nanmean(sst_bransfield_19972021,0)
sst_bransfield_19811996 = sst_19811996[clusters == 4,:]
sst_bransfield_19811996 = np.nanmean(sst_bransfield_19811996,0)
sst_bransfield_19812021 = np.hstack((sst_bransfield_19811996, sst_bransfield_19972021))
time_date_sst_19812021 = np.hstack((time_date_19811996, time_date_sst))
time_date_years = np.empty_like(time_date_sst_19812021)
time_date_months = np.empty_like(time_date_sst_19812021)
time_date_days = np.empty_like(time_date_sst_19812021)
for i in range(0, len(time_date_sst_19812021)):
    time_date_sst_19812021[i] = datetime.datetime(year=time_date_sst_19812021[i].year,
                                                  month=time_date_sst_19812021[i].month,
                                                  day=time_date_sst_19812021[i].day)
    time_date_years[i] = time_date_sst_19812021[i].year
    time_date_months[i] = time_date_sst_19812021[i].month
    time_date_days[i] = time_date_sst_19812021[i].day

for i in np.arange(1982, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1982:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = sst_bransfield_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_sst_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = sst_bransfield_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_sst_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1982:
        sst_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        sst_sepapr_19982021 = np.vstack((sst_sepapr_19982021, yeartemp_augmay_pd_8day.values))
sst_sepapr_19982021_mean = np.nanmean(sst_sepapr_19982021,0)
## Sea ice
seaice_bransfield_19972021 = seaice[clusters == 4,:]
seaice_bransfield_19972021 = np.nanmean(seaice_bransfield_19972021,0)
seaice_bransfield_19811996 = seaice_19811996[clusters == 4,:]
seaice_bransfield_19811996 = np.nanmean(seaice_bransfield_19811996,0)
seaice_bransfield_19812021 = np.hstack((seaice_bransfield_19811996, seaice_bransfield_19972021))
time_date_seaice_19812021 = np.hstack((time_date_19811996, time_date_seaice))
time_date_years = np.empty_like(time_date_seaice_19812021)
time_date_months = np.empty_like(time_date_seaice_19812021)
time_date_days = np.empty_like(time_date_seaice_19812021)
for i in range(0, len(time_date_seaice_19812021)):
    time_date_seaice_19812021[i] = datetime.datetime(year=time_date_seaice_19812021[i].year,
                                                  month=time_date_seaice_19812021[i].month,
                                                  day=time_date_seaice_19812021[i].day)
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day

for i in np.arange(1982, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1982:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = seaice_bransfield_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_seaice_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = seaice_bransfield_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_seaice_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1982:
        seaice_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        seaice_sepapr_19982021 = np.vstack((seaice_sepapr_19982021, yeartemp_augmay_pd_8day.values))
seaice_sepapr_19982021_mean = np.nanmean(seaice_sepapr_19982021,0)
## Sea ice advance
seaice_advanceday_19812021_mean = np.round(np.nanmean(seaice_advanceday.values),0)
## Sea ice retreat
seaice_retreatday_19812021_mean = np.round(np.nanmean(seaice_retreatday.values),0)
## PAR
par_bransfield = par[clusters == 4,:]
par_bransfield = np.nanmean(par_bransfield,0)
par_bransfield_df = pd.Series(data=par_bransfield, index=time_date_par)
par_bransfield_df = par_bransfield_df.resample('D').mean()
par_bransfield = par_bransfield_df.values
time_date_par_daily = par_bransfield_df.index
time_date_years = np.empty_like(time_date_par_daily)
time_date_months = np.empty_like(time_date_par_daily)
#for i in range(0, len(time_date_par_daily)):
#    time_date_years[i] = time_date_par_daily.year
#    time_date_months[i] = time_date_par_daily[i].month
for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_par_daily.year == i) & (time_date_par_daily.month == 6))[-1][-1]
        yeartemp_augmay = par_bransfield[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_par_daily[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_par_daily.year == i-1) & (time_date_par_daily.month == 6))[0][0]
        yeartemp_may = np.where((time_date_par_daily.year == i) & (time_date_par_daily.month == 6))[-1][-1]
        yeartemp_augmay = par_bransfield[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_par_daily[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1998:
        par_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        par_sepapr_19982021 = np.vstack((par_sepapr_19982021, yeartemp_augmay_pd_8day.values))
par_sepapr_19982021_mean = np.nanmean(par_sepapr_19982021,0)
#%% Plot figure
# SEAICE CONC
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982021_mean, marker='^', c='g', label = 'Chla', markersize=5, alpha=.5, zorder=10)
#axs.set_zorder(10)
ax2 = axs.twinx()
ax2.plot(np.arange(1,43), sst_sepapr_19982021_mean, marker='s', c='r', label = 'SST', markersize=5, alpha=.5,zorder=9)
ax3 = axs.twinx()
ax3.plot(np.arange(1,43), seaice_sepapr_19982021_mean, linestyle='-', c='grey', label = 'Sea Ice', markersize=5, alpha=.5)
ax4 = axs.twinx()
ax4.plot(np.arange(1,43), par_sepapr_19982021_mean, marker='*', c='y', label = 'PAR', markersize=5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2.spines['right'].set_color('r')
ax3.spines['right'].set_position(('outward', 60))
ax3.yaxis.label.set_color('grey')
ax3.tick_params(axis='y', colors='grey')
ax3.spines['right'].set_color('grey')
ax4.spines['right'].set_position(('outward', 110))
ax4.yaxis.label.set_color('y')
ax4.tick_params(axis='y', colors='y')
ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
ax2.set_ylabel('SST (°C)', fontsize=12)
ax3.set_ylabel('Sea Ice (%)', fontsize=12)
ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1.15), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'bransfield_sepapr_climatology.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
chl_sepapr_19982003 = np.nanmean(chl_sepapr_19982021[:6, :],0)
chl_sepapr_20042009 = np.nanmean(chl_sepapr_19982021[6:12, :],0)
chl_sepapr_20102015 = np.nanmean(chl_sepapr_19982021[12:18, :],0)
chl_sepapr_20162021 = np.nanmean(chl_sepapr_19982021[18:, :],0)
chl_sepapr_19982005 = np.nanmean(chl_sepapr_19982021[:8, :],0)
chl_sepapr_20062021 = np.nanmean(chl_sepapr_19982021[8:, :],0)
# Plot Chl
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982003, marker='^', c='g', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20042009, marker='s', c='g', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'bransfield_sepapr_climatology_comparison5years.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982005, marker='^', c='g', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20062021, marker='s', c='g', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'bransfield_sepapr_climatology_comparison_prepost2005.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
par_sepapr_19982003 = np.nanmean(par_sepapr_19982021[:6, :],0)
par_sepapr_20042009 = np.nanmean(par_sepapr_19982021[6:12, :],0)
par_sepapr_20102015 = np.nanmean(par_sepapr_19982021[12:18, :],0)
par_sepapr_20162021 = np.nanmean(par_sepapr_19982021[18:, :],0)
par_sepapr_19982005 = np.nanmean(par_sepapr_19982021[:8, :],0)
par_sepapr_20062021 = np.nanmean(par_sepapr_19982021[8:, :],0)
# Plot par
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982003, marker='^', c='y', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20042009, marker='s', c='y', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='y', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='y', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'bransfield_sepapr_climatology_comparison5years_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982005, marker='^', c='y', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20062021, marker='s', c='y', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'bransfield_sepapr_climatology_comparison_prepost2005_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
sst_sepapr_19821991 = np.nanmean(sst_sepapr_19982021[:10, :],0)
sst_sepapr_19922001 = np.nanmean(sst_sepapr_19982021[10:20, :],0)
sst_sepapr_20022011 = np.nanmean(sst_sepapr_19982021[20:30, :],0)
sst_sepapr_20122021 = np.nanmean(sst_sepapr_19982021[30:, :],0)
sst_sepapr_19822005 = np.nanmean(sst_sepapr_19982021[:24, :],0)
sst_sepapr_20062021 =  np.nanmean(sst_sepapr_19982021[24:, :],0)
# Plot sst
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19821991, marker='^', c='r', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_19922001, marker='s', c='r', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20022011, linestyle='--', c='r', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20122021, marker='*', linestyle=':', c='r', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'bransfield_sepapr_climatology_comparison10years_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19822005, marker='^', c='r', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20062021, marker='s', c='r', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'bransfield_sepapr_climatology_comparison_prepost2005_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
seaice_sepapr_19821991 = np.nanmean(seaice_sepapr_19982021[:10, :],0)
seaice_sepapr_19922001 = np.nanmean(seaice_sepapr_19982021[10:20, :],0)
seaice_sepapr_20022011 = np.nanmean(seaice_sepapr_19982021[20:30, :],0)
seaice_sepapr_20122021 = np.nanmean(seaice_sepapr_19982021[30:, :],0)
seaice_sepapr_19822005 = np.nanmean(seaice_sepapr_19982021[:24, :],0)
seaice_sepapr_20062021 =  np.nanmean(seaice_sepapr_19982021[24:, :],0)
# Plot seaice
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19821991, marker='^', c='grey', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_19922001, marker='s', c='grey', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20022011, linestyle='--', c='grey', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20122021, marker='*', linestyle=':', c='grey', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'bransfield_sepapr_climatology_comparison10years_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19822005, marker='^', c='grey', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20062021, marker='s', c='grey', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('seaice (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'bransfield_sepapr_climatology_comparison_prepost2005_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% OCEANIC SOUTH
leapyears_list = [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
## CHL #1998-2021
chl_oceanic = chl[clusters == 5,:]
chl_oceanic = np.nanmean(chl_oceanic,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = chl_oceanic[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_chl[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 5))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = chl_oceanic[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_chl[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1998:
        chl_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        chl_sepapr_19982021 = np.vstack((chl_sepapr_19982021, yeartemp_augmay_pd_8day.values))
chl_sepapr_19982021_mean = np.nanmean(chl_sepapr_19982021,0)
## SST
sst_oceanic_19972021 = sst[clusters == 5,:]
sst_oceanic_19972021 = np.nanmean(sst_oceanic_19972021,0)
sst_oceanic_19811996 = sst_19811996[clusters == 5,:]
sst_oceanic_19811996 = np.nanmean(sst_oceanic_19811996,0)
sst_oceanic_19812021 = np.hstack((sst_oceanic_19811996, sst_oceanic_19972021))
time_date_sst_19812021 = np.hstack((time_date_19811996, time_date_sst))
time_date_years = np.empty_like(time_date_sst_19812021)
time_date_months = np.empty_like(time_date_sst_19812021)
time_date_days = np.empty_like(time_date_sst_19812021)
for i in range(0, len(time_date_sst_19812021)):
    time_date_sst_19812021[i] = datetime.datetime(year=time_date_sst_19812021[i].year,
                                                  month=time_date_sst_19812021[i].month,
                                                  day=time_date_sst_19812021[i].day)
    time_date_years[i] = time_date_sst_19812021[i].year
    time_date_months[i] = time_date_sst_19812021[i].month
    time_date_days[i] = time_date_sst_19812021[i].day

for i in np.arange(1982, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1982:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = sst_oceanic_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_sst_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = sst_oceanic_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_sst_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1982:
        sst_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        sst_sepapr_19982021 = np.vstack((sst_sepapr_19982021, yeartemp_augmay_pd_8day.values))
sst_sepapr_19982021_mean = np.nanmean(sst_sepapr_19982021,0)
## Sea ice
seaice_oceanic_19972021 = seaice[clusters == 5,:]
seaice_oceanic_19972021 = np.nanmean(seaice_oceanic_19972021,0)
seaice_oceanic_19811996 = seaice_19811996[clusters == 5,:]
seaice_oceanic_19811996 = np.nanmean(seaice_oceanic_19811996,0)
seaice_oceanic_19812021 = np.hstack((seaice_oceanic_19811996, seaice_oceanic_19972021))
time_date_seaice_19812021 = np.hstack((time_date_19811996, time_date_seaice))
time_date_years = np.empty_like(time_date_seaice_19812021)
time_date_months = np.empty_like(time_date_seaice_19812021)
time_date_days = np.empty_like(time_date_seaice_19812021)
for i in range(0, len(time_date_seaice_19812021)):
    time_date_seaice_19812021[i] = datetime.datetime(year=time_date_seaice_19812021[i].year,
                                                  month=time_date_seaice_19812021[i].month,
                                                  day=time_date_seaice_19812021[i].day)
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day

for i in np.arange(1982, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1982:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = seaice_oceanic_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_seaice_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = seaice_oceanic_19812021[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_seaice_19812021[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1982:
        seaice_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        seaice_sepapr_19982021 = np.vstack((seaice_sepapr_19982021, yeartemp_augmay_pd_8day.values))
seaice_sepapr_19982021_mean = np.nanmean(seaice_sepapr_19982021,0)
## Sea ice advance
seaice_advanceday_19812021_mean = np.round(np.nanmean(seaice_advanceday.values),0)
## Sea ice retreat
seaice_retreatday_19812021_mean = np.round(np.nanmean(seaice_retreatday.values),0)
## PAR
par_oceanic = par[clusters == 5,:]
par_oceanic = np.nanmean(par_oceanic,0)
par_oceanic_df = pd.Series(data=par_oceanic, index=time_date_par)
par_oceanic_df = par_oceanic_df.resample('D').mean()
par_oceanic = par_oceanic_df.values
time_date_par_daily = par_oceanic_df.index
time_date_years = np.empty_like(time_date_par_daily)
time_date_months = np.empty_like(time_date_par_daily)
#for i in range(0, len(time_date_par_daily)):
#    time_date_years[i] = time_date_par_daily.year
#    time_date_months[i] = time_date_par_daily[i].month
for i in np.arange(1998, 2022):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_par_daily.year == i) & (time_date_par_daily.month == 6))[-1][-1]
        yeartemp_augmay = par_oceanic[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_par_daily[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_par_daily.year == i-1) & (time_date_par_daily.month == 6))[0][0]
        yeartemp_may = np.where((time_date_par_daily.year == i) & (time_date_par_daily.month == 6))[-1][-1]
        yeartemp_augmay = par_oceanic[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_par_daily[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1998:
        par_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        par_sepapr_19982021 = np.vstack((par_sepapr_19982021, yeartemp_augmay_pd_8day.values))
par_sepapr_19982021_mean = np.nanmean(par_sepapr_19982021,0)
#%% Plot figure
# SEAICE CONC
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982021_mean, marker='^', c='g', label = 'Chla', markersize=5, alpha=.5, zorder=10)
#axs.set_zorder(10)
ax2 = axs.twinx()
ax2.plot(np.arange(1,43), sst_sepapr_19982021_mean, marker='s', c='r', label = 'SST', markersize=5, alpha=.5,zorder=9)
ax3 = axs.twinx()
ax3.plot(np.arange(1,43), seaice_sepapr_19982021_mean, linestyle='-', c='grey', label = 'Sea Ice', markersize=5, alpha=.5)
ax4 = axs.twinx()
ax4.plot(np.arange(1,43), par_sepapr_19982021_mean, marker='*', c='y', label = 'PAR', markersize=5, alpha=.5)
#ax5 = axs.twinx()
#ax5.plot(np.arange(1,43), windspeed_sepapr_19982021_mean, marker='.', c='purple', label = 'Wind Speed', markersize=5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
ax2.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='red')
ax2.spines['right'].set_color('r')
ax3.spines['right'].set_position(('outward', 60))
ax3.yaxis.label.set_color('grey')
ax3.tick_params(axis='y', colors='grey')
ax3.spines['right'].set_color('grey')
ax4.spines['right'].set_position(('outward', 110))
ax4.yaxis.label.set_color('y')
ax4.tick_params(axis='y', colors='y')
ax4.spines['right'].set_color('y')
#ax5.spines['right'].set_position(('outward', 160))
#ax5.yaxis.label.set_color('purple')
#ax5.tick_params(axis='y', colors='purple')
#ax5.spines['right'].set_color('purple')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
ax2.set_ylabel('SST (°C)', fontsize=12)
ax3.set_ylabel('Sea Ice (%)', fontsize=12)
ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax5.set_ylabel('Wind Speed (ms$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
plt.axvline(13, linestyle='--', c='grey', alpha=0.3)
plt.axvline(32, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1.15), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanicsouth_sepapr_climatology.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
chl_sepapr_19982003 = np.nanmean(chl_sepapr_19982021[:6, :],0)
chl_sepapr_20042009 = np.nanmean(chl_sepapr_19982021[6:12, :],0)
chl_sepapr_20102015 = np.nanmean(chl_sepapr_19982021[12:18, :],0)
chl_sepapr_20162021 = np.nanmean(chl_sepapr_19982021[18:, :],0)
chl_sepapr_19982005 = np.nanmean(chl_sepapr_19982021[:8, :],0)
chl_sepapr_20062021 = np.nanmean(chl_sepapr_19982021[8:, :],0)
# Plot Chl
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982003, marker='^', c='g', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20042009, marker='s', c='g', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanicsouth_sepapr_climatology_comparison5years.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), chl_sepapr_19982005, marker='^', c='g', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), chl_sepapr_20062021, marker='s', c='g', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), chl_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanicsouth_sepapr_climatology_comparison_prepost2005.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
par_sepapr_19982003 = np.nanmean(par_sepapr_19982021[:6, :],0)
par_sepapr_20042009 = np.nanmean(par_sepapr_19982021[6:12, :],0)
par_sepapr_20102015 = np.nanmean(par_sepapr_19982021[12:18, :],0)
par_sepapr_20162021 = np.nanmean(par_sepapr_19982021[18:, :],0)
par_sepapr_19982005 = np.nanmean(par_sepapr_19982021[:8, :],0)
par_sepapr_20062021 = np.nanmean(par_sepapr_19982021[8:, :],0)
# Plot par
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982003, marker='^', c='y', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20042009, marker='s', c='y', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='y', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='y', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanicsouth_sepapr_climatology_comparison5years_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), par_sepapr_19982005, marker='^', c='y', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), par_sepapr_20062021, marker='s', c='y', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanicsouth_sepapr_climatology_comparison_prepost2005_par.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
sst_sepapr_19821991 = np.nanmean(sst_sepapr_19982021[:10, :],0)
sst_sepapr_19922001 = np.nanmean(sst_sepapr_19982021[10:20, :],0)
sst_sepapr_20022011 = np.nanmean(sst_sepapr_19982021[20:30, :],0)
sst_sepapr_20122021 = np.nanmean(sst_sepapr_19982021[30:, :],0)
sst_sepapr_19822005 = np.nanmean(sst_sepapr_19982021[:24, :],0)
sst_sepapr_20062021 =  np.nanmean(sst_sepapr_19982021[24:, :],0)
# Plot sst
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19821991, marker='^', c='r', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_19922001, marker='s', c='r', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20022011, linestyle='--', c='r', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20122021, marker='*', linestyle=':', c='r', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanicsouth_sepapr_climatology_comparison10years_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), sst_sepapr_19822005, marker='^', c='r', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), sst_sepapr_20062021, marker='s', c='r', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('SST (°C)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanicsouth_sepapr_climatology_comparison_prepost2005_sst.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Separate and calculate changes to yearly cycles
seaice_sepapr_19821991 = np.nanmean(seaice_sepapr_19982021[:10, :],0)
seaice_sepapr_19922001 = np.nanmean(seaice_sepapr_19982021[10:20, :],0)
seaice_sepapr_20022011 = np.nanmean(seaice_sepapr_19982021[20:30, :],0)
seaice_sepapr_20122021 = np.nanmean(seaice_sepapr_19982021[30:, :],0)
seaice_sepapr_19822005 = np.nanmean(seaice_sepapr_19982021[:24, :],0)
seaice_sepapr_20062021 =  np.nanmean(seaice_sepapr_19982021[24:, :],0)
# Plot seaice
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19821991, marker='^', c='grey', label = '1982-1991', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_19922001, marker='s', c='grey', label = '1992-2001', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20022011, linestyle='--', c='grey', label = '2002-2011', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20122021, marker='*', linestyle=':', c='grey', label = '2012-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('SST (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanicsouth_sepapr_climatology_comparison10years_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), seaice_sepapr_19822005, marker='^', c='grey', label = '1982-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), seaice_sepapr_20062021, marker='s', c='grey', label = '2006-2021', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20102015, linestyle='--', c='g', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), par_sepapr_20162021, marker='*', linestyle=':', c='g', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
#ax2.yaxis.label.set_color('red')
#ax2.tick_params(axis='y', colors='red')
#ax2.spines['right'].set_color('r')
#ax3.spines['right'].set_position(('outward', 60))
#ax3.yaxis.label.set_color('grey')
#ax3.tick_params(axis='y', colors='grey')
#ax3.spines['right'].set_color('grey')
#ax4.spines['right'].set_position(('outward', 110))
#ax4.yaxis.label.set_color('y')
#ax4.tick_params(axis='y', colors='y')
#ax4.spines['right'].set_color('y')
axs.set_ylabel('Sea Ice (%)', fontsize=12)
#ax2.set_ylabel('seaice (°C)', fontsize=12)
#ax3.set_ylabel('Sea Ice (%)', fontsize=12)
#ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
axs.legend(loc=0)
plt.xlim(1, 42)
#fig.legend(loc="upper right", ncol=2)
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\clusters_withoutseaice_min50\\')
graphs_dir = 'oceanicsouth_sepapr_climatology_comparison_prepost2005_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()