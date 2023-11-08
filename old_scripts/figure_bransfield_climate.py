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
fh = np.load('antarcticpeninsula_cluster.npz',allow_pickle = True)
clusters = fh['clusters']
# CHL
fh = np.load('chloc4so_19972021_10km.npz', allow_pickle=True)
lat_chl = fh['lat'][100:]
lon_chl = fh['lon'][30:250]
chl = fh['chl'][100:, 30:250, :]
time_date_chl = fh['time_date']
# Correct values
chl[chl > 50] = 50
# BLOOM METRICS BRANSFIELD
fh = np.load('phenology_bransfield_10km.npz', allow_pickle=True)
b_init = fh['b_init']
b_term = fh['b_term']
b_peak = fh['b_peak']
chl_max = fh['chl_max']
b_area = fh['b_area']
b_dur = fh['b_dur']
b_years = fh['time_years']
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
#PAR
### Load data 1998-2020
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\par\\')
fh = np.load('par_19972022_10km.npz', allow_pickle=True)
lat_par = fh['lat'][100:]
lon_par = fh['lon'][30:250]
par = fh['par'][100:, 30:250, :]
time_date_par = fh['time_date']
# MEI
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\elnino\\')
mei_pd = pd.read_csv('meiv2.csv', sep=';')
mei_monthly_19972021 = mei_pd['MEI2'][8:-8].values
mei_months = mei_pd['Month'][8:-8].values
mei_years = mei_pd['Year'][8:-8].values
# SAM
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sam\\')
sam_pd = pd.read_csv('sam.csv', sep=';')
sam_monthly_19972021 = sam_pd['SAM'].values
sam_months = sam_pd['Month'].values
sam_years = sam_pd['Year'].values
# WINDS
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\winds\\')
fh = np.load('winds_19972019_daily.npz', allow_pickle=True)
lat_winds = fh['lat']
lon_winds = fh['lon']
winds_northward = fh['northward_wind']
winds_eastward = fh['eastward_wind']
time_date_winds = fh['time_date']
# Sea Ice duration and timings from PALMER LTER
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaicetimings')
ice_timing = pd.read_csv('ice_timing.csv', sep=';')
ice_timing_years = ice_timing['Ice Year']
seaice_advanceday = ice_timing['Pori adv']
seaice_retreatday = ice_timing['Pori ret']
seaice_duration = ice_timing['Pori dur']
# Sea Ice Extent from PALMER LTER
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaiceextent')
seaice_extent_df = pd.read_csv('seaiceextent.csv', sep=';')
seaice_extent_years = seaice_extent_df['Year']
seaice_extent = seaice_extent_df['Ori_Ext']
#%% Create dataframe for bransfield using spring-summer data (September-April means)
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
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs.transAxes)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepapr_climatology.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Setember-Dec Mean from 1981 to 2021
# CHL
chl_bransfield = chl[clusters == 4,:]
chl_bransfield = np.nanmean(chl_bransfield,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1997, 2022):
    yeartemp_sep = chl_bransfield[(time_date_years == i) & (time_date_months == 9)]
    yeartemp_oct = chl_bransfield[(time_date_years == i) & (time_date_months == 10)]
    yeartemp_nov = chl_bransfield[(time_date_years == i) & (time_date_months == 11)]
    yeartemp_dec = chl_bransfield[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_bransfield[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_bransfield[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = chl_bransfield[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = chl_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1997:
        chl_octdec19982021 = yeartemp_summermean
#        chl_summerstds19982021 = yeartemp_summerstd
    else:
        chl_octdec19982021 = np.hstack((chl_octdec19982021, yeartemp_summermean))
#        chl_summerstds19982021 = np.hstack((chl_summerstds19982021, yeartemp_summerstd))
# SST
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
    yeartemp_sep = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 9)]
    yeartemp_oct = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 10)]
    yeartemp_nov = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 11)]
    yeartemp_dec = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_bransfield[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_bransfield[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = chl_bransfield[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = chl_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1982:
        sst_octdec19812021 = yeartemp_summermean
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        sst_octdec19812021 = np.hstack((sst_octdec19812021, yeartemp_summermean))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# PAR
par_bransfield = par[clusters == 4,:]
par_bransfield = np.nanmean(par_bransfield,0)
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1997, 2022):
    yeartemp_sep = par_bransfield[(time_date_years == i) & (time_date_months == 9)]
    yeartemp_oct = par_bransfield[(time_date_years == i) & (time_date_months == 10)]
    yeartemp_nov = par_bransfield[(time_date_years == i) & (time_date_months == 11)]
    yeartemp_dec = par_bransfield[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = par_bransfield[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = par_bransfield[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = par_bransfield[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = par_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1997:
        par_octdec19982021 = yeartemp_summermean
#        par_summerstds19982021 = yeartemp_summerstd
    else:
        par_octdec19982021 = np.hstack((par_octdec19982021, yeartemp_summermean))
#        par_summerstds19982021 = np.hstack((par_summerstds19982021, yeartemp_summerstd))
#%% SEPTEMBER-DECEMBER
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1997,2022), chl_octdec19982021, marker='o', c='g', label = 'Chla', s=50, alpha=.5, zorder=10)
ax2 = axs.twinx()
ax2.plot(np.arange(1982,2022), sst_octdec19812021, marker='s', c='r', label = 'SST', markersize=5, alpha=.5,zorder=9)
ax3 = axs.twinx()
ax3.plot(np.arange(1980,2022), seaice_retreatday, linestyle='-', c='grey', label = 'Sea Ice Retreat', markersize=5, alpha=.5)
ax4 = axs.twinx()
ax4.plot(np.arange(1997,2022), par_octdec19982021, marker='*', c='y', label = 'PAR', markersize=5, alpha=.5)
ax5 = axs.twinx()
ax5.plot(np.arange(1979,2021), seaice_extent.values, linestyle='--', c='grey', label = 'Sea Ice Extent', markersize=5, alpha=.5)
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
ax5.spines['right'].set_position(('outward', 160))
ax5.yaxis.label.set_color('grey')
ax5.tick_params(axis='y', colors='grey')
ax5.spines['right'].set_color('grey')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
ax2.set_ylabel('SST (°C)', fontsize=12)
ax3.set_ylabel('Sea Ice Ret. Day', fontsize=12)
ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
ax5.set_ylabel('Sea Ice Extent', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepdec_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Setember Mean from 1981 to 2021
# CHL
chl_bransfield = chl[clusters == 4,:]
chl_bransfield = np.nanmean(chl_bransfield,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1997, 2022):
    yeartemp_sep = chl_bransfield[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = chl_bransfield[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = chl_bransfield[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = chl_bransfield[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_bransfield[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_bransfield[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = chl_bransfield[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = chl_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep,
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1997:
        chl_sep19982021 = yeartemp_summermean
#        chl_summerstds19982021 = yeartemp_summerstd
    else:
        chl_sep19982021 = np.hstack((chl_sep19982021, yeartemp_summermean))
#        chl_summerstds19982021 = np.hstack((chl_summerstds19982021, yeartemp_summerstd))
# SST
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
    yeartemp_sep = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_bransfield[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_bransfield[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = chl_bransfield[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = chl_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1982:
        sst_sep19812021 = yeartemp_summermean
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        sst_sep19812021 = np.hstack((sst_sep19812021, yeartemp_summermean))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# PAR
par_bransfield = par[clusters == 4,:]
par_bransfield = np.nanmean(par_bransfield,0)
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1997, 2022):
    yeartemp_sep = par_bransfield[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = par_bransfield[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = par_bransfield[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = par_bransfield[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = par_bransfield[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = par_bransfield[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = par_bransfield[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = par_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1997:
        par_sep19982021 = yeartemp_summermean
#        par_summerstds19982021 = yeartemp_summerstd
    else:
        par_sep19982021 = np.hstack((par_sep19982021, yeartemp_summermean))
#        par_summerstds19982021 = np.hstack((par_summerstds19982021, yeartemp_summerstd))

#%% SEPTEMBER ONLY
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1997,2022), chl_sep19982021, marker='o', c='g', label = 'Chla', s=50, alpha=.5, zorder=10)
ax2 = axs.twinx()
ax2.plot(np.arange(1982,2022), sst_sep19812021, marker='s', c='r', label = 'SST', markersize=5, alpha=.5,zorder=9)
ax3 = axs.twinx()
ax3.plot(np.arange(1980,2022), seaice_retreatday, linestyle='-', c='grey', label = 'Sea Ice Retreat', markersize=5, alpha=.5)
ax4 = axs.twinx()
ax4.plot(np.arange(1997,2022), par_sep19982021, marker='*', c='y', label = 'PAR', markersize=5, alpha=.5)
ax5 = axs.twinx()
ax5.plot(np.arange(1979,2021), seaice_extent.values, linestyle='--', c='grey', label = 'Sea Ice Extent', markersize=5, alpha=.5)
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
ax5.spines['right'].set_position(('outward', 160))
ax5.yaxis.label.set_color('grey')
ax5.tick_params(axis='y', colors='grey')
ax5.spines['right'].set_color('grey')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
ax2.set_ylabel('SST (°C)', fontsize=12)
ax3.set_ylabel('Sea Ice Ret. Day', fontsize=12)
ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
ax5.set_ylabel('Sea Ice Extent', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()

#%%
stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_sep19982021)], chl_sep19982021[~np.isnan(chl_sep19982021)])
stats.linregress(np.arange(1997, 2022)[~np.isnan(par_sep19982021)], par_sep19982021[~np.isnan(par_sep19982021)])
stats.linregress(np.arange(2005, 2022)[~np.isnan(sst_sep19812021[23:])], sst_sep19812021[23:][~np.isnan(sst_sep19812021[23:])])
stats.linregress(np.arange(2005, 2022)[~np.isnan(chl_sep19982021[8:])], chl_sep19982021[8:][~np.isnan(chl_sep19982021[8:])])

#%% Apr-May Mean from 1981 to 2021
# CHL
chl_bransfield = chl[clusters == 4,:]
chl_bransfield = np.nanmean(chl_bransfield,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1998, 2022):
#    yeartemp_sep = chl_bransfield[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = chl_bransfield[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = chl_bransfield[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = chl_bransfield[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_bransfield[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = chl_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = chl_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr)))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1998:
        chl_marapr19982021 = yeartemp_summermean
#        chl_summerstds19982021 = yeartemp_summerstd
    else:
        chl_marapr19982021 = np.hstack((chl_marapr19982021, yeartemp_summermean))
#        chl_summerstds19982021 = np.hstack((chl_summerstds19982021, yeartemp_summerstd))
# SST
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
#    yeartemp_sep = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_bransfield[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1982:
        sst_marapr19812021 = yeartemp_summermean
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        sst_marapr19812021 = np.hstack((sst_marapr19812021, yeartemp_summermean))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# PAR
par_bransfield = par[clusters == 4,:]
par_bransfield = np.nanmean(par_bransfield,0)
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
#    yeartemp_sep = par_bransfield[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = par_bransfield[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = par_bransfield[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = par_bransfield[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = par_bransfield[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = par_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = par_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = par_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_mar, yeartemp_apr
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1998:
        par_marapr19982021 = yeartemp_summermean
#        par_summerstds19982021 = yeartemp_summerstd
    else:
        par_marapr19982021 = np.hstack((par_marapr19982021, yeartemp_summermean))
#        par_summerstds19982021 = np.hstack((par_summerstds19982021, yeartemp_summerstd))
#%% MARCH-APRIL
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2022), chl_marapr19982021, marker='o', c='g', label = 'Chla', s=50, alpha=.5, zorder=10)
ax2 = axs.twinx()
ax2.plot(np.arange(1982,2022), sst_marapr19812021, marker='s', c='r', label = 'SST', markersize=5, alpha=.5,zorder=9)
ax3 = axs.twinx()
ax3.plot(np.arange(1980,2022), seaice_advanceday, linestyle='-', c='grey', label = 'Sea Ice Advance', markersize=5, alpha=.5)
ax4 = axs.twinx()
ax4.plot(np.arange(1998,2022), par_marapr19982021, marker='*', c='y', label = 'PAR', markersize=5, alpha=.5)
ax5 = axs.twinx()
ax5.plot(np.arange(1979,2021), seaice_extent.values, linestyle='--', c='grey', label = 'Sea Ice Extent', markersize=5, alpha=.5)
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
ax5.spines['right'].set_position(('outward', 160))
ax5.yaxis.label.set_color('grey')
ax5.tick_params(axis='y', colors='grey')
ax5.spines['right'].set_color('grey')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
ax2.set_ylabel('SST (°C)', fontsize=12)
ax3.set_ylabel('Sea Ice Ret. Day', fontsize=12)
ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
ax5.set_ylabel('Sea Ice Extent', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_marapr_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Individual plots
## SEPTEMBER
# CHL
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1997,2022), chl_sep19982021, marker='o', c='g', label = 'Chla', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_sep19982021)], chl_sep19982021[~np.isnan(chl_sep19982021)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='g', label = 'Sep 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(chl_sep19982021[:9])], chl_sep19982021[:9][~np.isnan(chl_sep19982021[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Sep 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_sep19982021[9:])], chl_sep19982021[9:][~np.isnan(chl_sep19982021[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Sep 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# PAR
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1997,2022), par_sep19982021, marker='o', c='y', label = 'PAR', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(par_sep19982021)], par_sep19982021[~np.isnan(par_sep19982021)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='y', label = 'Sep 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(par_sep19982021[:9])], par_sep19982021[:9][~np.isnan(par_sep19982021[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Sep 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_sep19982021[9:])], par_sep19982021[9:][~np.isnan(par_sep19982021[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Sep 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SST
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1982,2022), sst_sep19812021, marker='o', c='r', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1982, 2022)[~np.isnan(sst_sep19812021)], sst_sep19812021[~np.isnan(sst_sep19812021)])
axs.plot(np.arange(1982,2022), np.arange(1982,2022)*slope+intercept, c='r', label = 'Sep 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1982, 2006)[~np.isnan(sst_sep19812021[:24])], sst_sep19812021[:24][~np.isnan(sst_sep19812021[:24])])
axs.plot(np.arange(1982,2006), np.arange(1982,2006)*slope+intercept, c='b', label = 'Sep 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_sep19812021[24:])], sst_sep19812021[24:][~np.isnan(sst_sep19812021[24:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Sep 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Sea Ice Retreat
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1980,2022), seaice_retreatday.values, marker='o', c='gray', label = 'Sea Ice Retreat', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1980, 2022)[~np.isnan(seaice_retreatday)], seaice_retreatday[~np.isnan(seaice_retreatday)])
axs.plot(np.arange(1980,2022), np.arange(1980,2022)*slope+intercept, c='gray', label = 'Sep 80-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1980, 2006)[~np.isnan(seaice_retreatday[:26])], seaice_retreatday[:26][~np.isnan(seaice_retreatday[:26])])
axs.plot(np.arange(1980,2006), np.arange(1980,2006)*slope+intercept, c='b', label = 'Sep 80-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(seaice_retreatday[26:])], seaice_retreatday[26:][~np.isnan(seaice_retreatday[26:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Sep 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
axs.set_ylabel('Sea Ice Ret. Day', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1980, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_seaiceretreat_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Sea Ice Extent
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1979,2021), seaice_extent.values, marker='o', c='gray', label = 'Sea Ice Extent', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1979, 2021)[~np.isnan(seaice_extent)], seaice_extent[~np.isnan(seaice_extent)])
axs.plot(np.arange(1979,2021), np.arange(1979,2021)*slope+intercept, c='gray', label = 'Sep 79-20', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1979, 2006)[~np.isnan(seaice_extent[:27])], seaice_extent[:27][~np.isnan(seaice_extent[:27])])
axs.plot(np.arange(1979,2006), np.arange(1979,2006)*slope+intercept, c='b', label = 'Sep 80-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2021)[~np.isnan(seaice_extent[27:])], seaice_extent[27:][~np.isnan(seaice_extent[27:])])
axs.plot(np.arange(2006,2021), np.arange(2006,2021)*slope+intercept, c='r', label = 'Sep 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
axs.set_ylabel('Sea Ice Extent', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1979, 2021)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_seaiceextent_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% APRIL-MAY
# CHL
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2022), chl_marapr19982021, marker='o', c='g', label = 'Chla', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(chl_marapr19982021)], chl_marapr19982021[~np.isnan(chl_marapr19982021)])
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, c='g', label = 'marapr 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(chl_marapr19982021[:8])], chl_marapr19982021[:8][~np.isnan(chl_marapr19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'marapr 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_marapr19982021[8:])], chl_marapr19982021[8:][~np.isnan(chl_marapr19982021[8:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'marapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'marapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_marapr_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# PAR
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2022), par_marapr19982021, marker='o', c='y', label = 'PAR', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(par_marapr19982021)], par_marapr19982021[~np.isnan(par_marapr19982021)])
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, c='y', label = 'marapr 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(par_marapr19982021[:8])], par_marapr19982021[:8][~np.isnan(par_marapr19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'marapr 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_marapr19982021[8:])], par_marapr19982021[8:][~np.isnan(par_marapr19982021[8:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'marapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'marapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_marapr_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SST
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1982,2022), sst_marapr19812021, marker='o', c='r', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1982, 2022)[~np.isnan(sst_marapr19812021)], sst_marapr19812021[~np.isnan(sst_marapr19812021)])
axs.plot(np.arange(1982,2022), np.arange(1982,2022)*slope+intercept, c='r', label = 'marapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1982, 2006)[~np.isnan(sst_marapr19812021[:24])], sst_marapr19812021[:24][~np.isnan(sst_marapr19812021[:24])])
axs.plot(np.arange(1982,2006), np.arange(1982,2006)*slope+intercept, c='b', label = 'marapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_marapr19812021[24:])], sst_marapr19812021[24:][~np.isnan(sst_marapr19812021[24:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'marapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'marapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_marapr_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Sea Ice Advance
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1980,2022), seaice_advanceday.values, marker='o', c='gray', label = 'Sea Ice Advance', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1980, 2022)[~np.isnan(seaice_advanceday)], seaice_advanceday[~np.isnan(seaice_advanceday)])
axs.plot(np.arange(1980,2022), np.arange(1980,2022)*slope+intercept, c='gray', label = 'marapr 80-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1980, 2006)[~np.isnan(seaice_advanceday[:26])], seaice_advanceday[:26][~np.isnan(seaice_advanceday[:26])])
axs.plot(np.arange(1980,2006), np.arange(1980,2006)*slope+intercept, c='b', label = 'marapr 80-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(seaice_advanceday[26:])], seaice_advanceday[26:][~np.isnan(seaice_advanceday[26:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'marapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
axs.set_ylabel('Sea Ice Adv. Day', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'marapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1980, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_marapr_seaiceadvance_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Sea Ice Extent
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1979,2021), seaice_extent.values, marker='o', c='gray', label = 'Sea Ice Extent', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1979, 2021)[~np.isnan(seaice_extent)], seaice_extent[~np.isnan(seaice_extent)])
axs.plot(np.arange(1979,2021), np.arange(1979,2021)*slope+intercept, c='gray', label = 'marapr 79-20', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1979, 2006)[~np.isnan(seaice_extent[:27])], seaice_extent[:27][~np.isnan(seaice_extent[:27])])
axs.plot(np.arange(1979,2006), np.arange(1979,2006)*slope+intercept, c='b', label = 'marapr 80-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2021)[~np.isnan(seaice_extent[27:])], seaice_extent[27:][~np.isnan(seaice_extent[27:])])
axs.plot(np.arange(2006,2021), np.arange(2006,2021)*slope+intercept, c='r', label = 'marapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('grey')
axs.tick_params(axis='y', colors='grey')
axs.spines['left'].set_color('grey')
axs.set_ylabel('Sea Ice Extent', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'marapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1979, 2021)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_marapr_seaiceextent_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Setember-April Mean from 1981 to 2021
# CHL
chl_bransfield = chl[clusters == 4,:]
chl_bransfield = np.nanmean(chl_bransfield,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = chl_bransfield[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = chl_bransfield[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = chl_bransfield[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = chl_bransfield[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = chl_bransfield[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = chl_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = chl_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1998:
        chl_sepapr19982021 = yeartemp_summermean
#        chl_summerstds19982021 = yeartemp_summerstd
    else:
        chl_sepapr19982021 = np.hstack((chl_sepapr19982021, yeartemp_summermean))
#        chl_summerstds19982021 = np.hstack((chl_summerstds19982021, yeartemp_summerstd))
# SST
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
for i in np.arange(1983, 2022):
    yeartemp_sep = sst_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = sst_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = sst_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = sst_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1983:
        sst_sepapr19812021 = yeartemp_summermean
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        sst_sepapr19812021 = np.hstack((sst_sepapr19812021, yeartemp_summermean))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# PAR
par_bransfield = par[clusters == 4,:]
par_bransfield = np.nanmean(par_bransfield,0)
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = par_bransfield[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = par_bransfield[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = par_bransfield[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = par_bransfield[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = par_bransfield[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = par_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = par_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = par_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr
                                                )))
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
#                                                )))    
    if i == 1998:
        par_sepapr19982021 = yeartemp_summermean
#        par_summerstds19982021 = yeartemp_summerstd
    else:
        par_sepapr19982021 = np.hstack((par_sepapr19982021, yeartemp_summermean))
#        par_summerstds19982021 = np.hstack((par_summerstds19982021, yeartemp_summerstd))
#%% PLOT
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2022), chl_sepapr19982021, marker='o', c='g', label = 'Chla', s=50, alpha=.5, zorder=10)
ax2 = axs.twinx()
ax2.plot(np.arange(1983,2022), sst_sepapr19812021, marker='s', c='r', label = 'SST', markersize=5, alpha=.5,zorder=9)
ax3 = axs.twinx()
ax3.plot(np.arange(1980,2022), seaice_advanceday, linestyle='-', c='grey', label = 'Sea Ice Advance', markersize=5, alpha=.5)
ax4 = axs.twinx()
ax4.plot(np.arange(1998,2022), par_sepapr19982021, marker='*', c='y', label = 'PAR', markersize=5, alpha=.5)
ax5 = axs.twinx()
ax5.plot(np.arange(1979,2021), seaice_extent.values, linestyle='--', c='grey', label = 'Sea Ice Extent', markersize=5, alpha=.5)
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
ax5.spines['right'].set_position(('outward', 160))
ax5.yaxis.label.set_color('grey')
ax5.tick_params(axis='y', colors='grey')
ax5.spines['right'].set_color('grey')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
ax2.set_ylabel('SST (°C)', fontsize=12)
ax3.set_ylabel('Sea Ice Ret. Day', fontsize=12)
ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
ax5.set_ylabel('Sea Ice Extent', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper right", ncol=2)
fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepapr_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%%
# CHL
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2022), chl_sepapr19982021, marker='o', c='g', label = 'Chla', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(chl_sepapr19982021)], chl_sepapr19982021[~np.isnan(chl_sepapr19982021)])
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, c='g', label = 'sepapr 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(chl_sepapr19982021[:8])], chl_sepapr19982021[:8][~np.isnan(chl_sepapr19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'sepapr 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_sepapr19982021[8:])], chl_sepapr19982021[8:][~np.isnan(chl_sepapr19982021[8:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepapr_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# PAR
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2022), par_sepapr19982021, marker='o', c='y', label = 'PAR', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(par_sepapr19982021)], par_sepapr19982021[~np.isnan(par_sepapr19982021)])
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, c='y', label = 'sepapr 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(par_sepapr19982021[:8])], par_sepapr19982021[:8][~np.isnan(par_sepapr19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'sepapr 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_sepapr19982021[8:])], par_sepapr19982021[8:][~np.isnan(par_sepapr19982021[8:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepapr_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SST
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1983,2022), sst_sepapr19812021, marker='o', c='r', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1983, 2022)[~np.isnan(sst_sepapr19812021)], sst_sepapr19812021[~np.isnan(sst_sepapr19812021)])
axs.plot(np.arange(1983,2022), np.arange(1983,2022)*slope+intercept, c='r', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1983, 2006)[~np.isnan(sst_sepapr19812021[:23])], sst_sepapr19812021[:23][~np.isnan(sst_sepapr19812021[:23])])
axs.plot(np.arange(1983,2006), np.arange(1983,2006)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_sepapr19812021[23:])], sst_sepapr19812021[23:][~np.isnan(sst_sepapr19812021[23:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1983, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepapr_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% SAM
## SAM #1998-2021
for i in np.arange(1980, 2022):
    yeartemp_sep = sam_monthly_19972021[(sam_years == i-1) & (sam_months == 9)]
    yeartemp_oct = sam_monthly_19972021[(sam_years == i-1) & (sam_months == 10)]
    yeartemp_nov = sam_monthly_19972021[(sam_years == i-1) & (sam_months == 11)]
    yeartemp_dec = sam_monthly_19972021[(sam_years == i-1) & (sam_months == 12)]
    yeartemp_jan = sam_monthly_19972021[(sam_years == i) & (sam_months == 1)]
    yeartemp_feb = sam_monthly_19972021[(sam_years == i) & (sam_months == 2)]
    yeartemp_mar = sam_monthly_19972021[(sam_years == i) & (sam_months == 3)]
    yeartemp_apr = sam_monthly_19972021[(sam_years == i) & (sam_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1980:
        sam_summermeans19982021 = yeartemp_summermean
        sam_summerstds19982021 = yeartemp_summerstd
    else:
        sam_summermeans19982021 = np.hstack((sam_summermeans19982021, yeartemp_summermean))
        sam_summerstds19982021 = np.hstack((sam_summerstds19982021, yeartemp_summerstd)) 
# SAM sam_monthly_19972021
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1980,2022), sam_summermeans19982021, marker='o', c='k', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1980, 2022)[~np.isnan(sam_summermeans19982021)], sam_summermeans19982021[~np.isnan(sam_summermeans19982021)])
axs.plot(np.arange(1980,2022), np.arange(1980,2022)*slope+intercept, c='k', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1980, 2006)[~np.isnan(sam_summermeans19982021[:26])], sam_summermeans19982021[:26][~np.isnan(sam_summermeans19982021[:26])])
axs.plot(np.arange(1980,2006), np.arange(1980,2006)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sam_summermeans19982021[26:])], sam_summermeans19982021[26:][~np.isnan(sam_summermeans19982021[26:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('SAM', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1980, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepapr_SAM_19802022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Wind Speed
# Define region of interest to average
Bransfield_winds_verts = [(-65, -64.5),
             (-65, -63),
             (-52.5, -60),
             (-52.5, -61.6)]
#plt.figure()
#map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
#map.coastlines(resolution='10m', color='black', linewidth=1)
#map.set_extent([-68, -50, -67, -60])
#map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
#                                        edgecolor='k',
#                                        facecolor=cartopy.feature.COLORS['land']))
#gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
#poly_CBS = Polygon(list(Bransfield_winds_verts), facecolor=[1,1,1,0], edgecolor='k', linewidth=1, linestyle='--', zorder=2, transform=ccrs.PlateCarree())
#plt.gca().add_patch(poly_CBS)
#plt.tight_layout()
## Northward wind
x, y = np.meshgrid(lon_winds, lat_winds) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(Bransfield_winds_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon_winds), len(lat_winds))
mask3d = np.repeat([mask], np.size(winds_northward,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
winds_northward_bransfield = np.ma.array(winds_northward, mask=~mask3d)
winds_northward_bransfield = np.nanmean(winds_northward_bransfield, (0,1))
## Eastward wind
x, y = np.meshgrid(lon_winds, lat_winds) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(Bransfield_winds_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon_winds), len(lat_winds))
mask3d = np.repeat([mask], np.size(winds_eastward,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
winds_eastward_bransfield = np.ma.array(winds_eastward, mask=~mask3d)
winds_eastward_bransfield = np.nanmean(winds_eastward_bransfield, (0,1))
## Calculate wind speed
wind_speed = np.sqrt(winds_eastward_bransfield**2 + winds_northward_bransfield**2)
wind_dir = np.empty_like(wind_speed)
for k in range(0, len(wind_dir)):
    wind_dir_trig_to = math.atan2(winds_eastward_bransfield[k]/wind_speed[k], winds_northward_bransfield[k]/wind_speed[k]) 
    wind_dir_trig_to_degrees = wind_dir_trig_to * 180/math.pi
    wind_dir[k] = wind_dir_trig_to_degrees + 180
time_date_years = time_date_winds.astype('datetime64[Y]').astype(int) + 1970
time_date_months = time_date_winds.astype('datetime64[M]').astype(int) % 12 + 1 
## WIND SPEED #1998-2019
for i in np.arange(1998, 2020):
    yeartemp_sep = wind_speed[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wind_speed[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wind_speed[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wind_speed[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wind_speed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wind_speed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wind_speed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wind_speed[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1998:
        windspeed_summermeans19982021 = yeartemp_summermean
        windspeed_summerstds19982021 = yeartemp_summerstd
    else:
        windspeed_summermeans19982021 = np.hstack((windspeed_summermeans19982021, yeartemp_summermean))
        windspeed_summerstds19982021 = np.hstack((windspeed_summerstds19982021, yeartemp_summerstd)) 
## WIND DIR #1998-2019
for i in np.arange(1998, 2020):
    yeartemp_sep = wind_dir[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wind_dir[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wind_dir[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wind_dir[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wind_dir[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wind_dir[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wind_dir[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wind_dir[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1998:
        winddir_summermeans19982021 = yeartemp_summermean
        winddir_summerstds19982021 = yeartemp_summerstd
    else:
        winddir_summermeans19982021 = np.hstack((winddir_summermeans19982021, yeartemp_summermean))
        winddir_summerstds19982021 = np.hstack((winddir_summerstds19982021, yeartemp_summerstd)) 

fig, axs = plt.subplots(1, 1, figsize=(11,2))
# Wind Speed
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2020), windspeed_summermeans19982021, marker='o', c='purple', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2020)[~np.isnan(windspeed_summermeans19982021)], windspeed_summermeans19982021[~np.isnan(windspeed_summermeans19982021)])
axs.plot(np.arange(1998,2020), np.arange(1998,2020)*slope+intercept, c='purple', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(windspeed_summermeans19982021[:8])], windspeed_summermeans19982021[:8][~np.isnan(windspeed_summermeans19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_summermeans19982021[8:])], windspeed_summermeans19982021[8:][~np.isnan(windspeed_summermeans19982021[8:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (m/s)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepapr_windspeed_19802022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
fig, axs = plt.subplots(1, 1, figsize=(11,2))
# Wind Direction
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2020), winddir_summermeans19982021, marker='o', c='purple', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2020)[~np.isnan(winddir_summermeans19982021)], winddir_summermeans19982021[~np.isnan(winddir_summermeans19982021)])
axs.plot(np.arange(1998,2020), np.arange(1998,2020)*slope+intercept, c='purple', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(winddir_summermeans19982021[:8])], winddir_summermeans19982021[:8][~np.isnan(winddir_summermeans19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(winddir_summermeans19982021[8:])], winddir_summermeans19982021[8:][~np.isnan(winddir_summermeans19982021[8:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Dir (Degrees)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepapr_winddir_19802022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Wind Speed and Direction March-April
## WIND SPEED #1998-2019
for i in np.arange(1998, 2020):
    yeartemp_sep = wind_speed[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wind_speed[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wind_speed[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wind_speed[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wind_speed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wind_speed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wind_speed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wind_speed[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1998:
        windspeed_marapr19982021 = yeartemp_summermean
#        windspeed_summerstds19982021 = yeartemp_summerstd
    else:
        windspeed_marapr19982021 = np.hstack((windspeed_marapr19982021, yeartemp_summermean))
#        windspeed_summerstds19982021 = np.hstack((windspeed_summerstds19982021, yeartemp_summerstd)) 
## WIND DIR #1998-2019
for i in np.arange(1998, 2020):
    yeartemp_sep = wind_dir[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wind_dir[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wind_dir[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wind_dir[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wind_dir[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wind_dir[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wind_dir[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wind_dir[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1998:
        winddir_marapr19982021 = yeartemp_summermean
#        winddir_summerstds19982021 = yeartemp_summerstd
    else:
        winddir_marapr19982021 = np.hstack((winddir_marapr19982021, yeartemp_summermean))
#        winddir_summerstds19982021 = np.hstack((winddir_summerstds19982021, yeartemp_summerstd)) 

fig, axs = plt.subplots(1, 1, figsize=(11,2))
# Wind Speed
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2020), windspeed_marapr19982021, marker='o', c='purple', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2020)[~np.isnan(windspeed_marapr19982021)], windspeed_marapr19982021[~np.isnan(windspeed_marapr19982021)])
axs.plot(np.arange(1998,2020), np.arange(1998,2020)*slope+intercept, c='purple', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(windspeed_marapr19982021[:8])], windspeed_marapr19982021[:8][~np.isnan(windspeed_marapr19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_marapr19982021[8:])], windspeed_marapr19982021[8:][~np.isnan(windspeed_marapr19982021[8:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (m/s)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_marapr_windspeed_19802022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
fig, axs = plt.subplots(1, 1, figsize=(11,2))
# Wind Direction
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2020), winddir_marapr19982021, marker='o', c='purple', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2020)[~np.isnan(winddir_marapr19982021)], winddir_marapr19982021[~np.isnan(winddir_marapr19982021)])
axs.plot(np.arange(1998,2020), np.arange(1998,2020)*slope+intercept, c='purple', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(winddir_marapr19982021[:8])], winddir_marapr19982021[:8][~np.isnan(winddir_marapr19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(winddir_marapr19982021[8:])], winddir_marapr19982021[8:][~np.isnan(winddir_marapr19982021[8:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Dir (Degrees)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_marapr_winddir_19802022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Wind Speed and Direction September
## WIND SPEED #1998-2019
for i in np.arange(1998, 2020):
    yeartemp_sep = wind_speed[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wind_speed[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wind_speed[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wind_speed[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wind_speed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wind_speed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wind_speed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wind_speed[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(yeartemp_sep)
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_mar,
#                                                yeartemp_apr)))    
    if i == 1998:
        windspeed_sep19982021 = yeartemp_summermean
#        windspeed_summerstds19982021 = yeartemp_summerstd
    else:
        windspeed_sep19982021 = np.hstack((windspeed_sep19982021, yeartemp_summermean))
#        windspeed_summerstds19982021 = np.hstack((windspeed_summerstds19982021, yeartemp_summerstd)) 
## WIND DIR #1998-2019
for i in np.arange(1998, 2020):
    yeartemp_sep = wind_dir[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wind_dir[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wind_dir[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wind_dir[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wind_dir[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wind_dir[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wind_dir[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wind_dir[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(yeartemp_sep)
#    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_mar,
#                                                yeartemp_apr)))    
    if i == 1998:
        winddir_sep19982021 = yeartemp_summermean
#        winddir_summerstds19982021 = yeartemp_summerstd
    else:
        winddir_sep19982021 = np.hstack((winddir_sep19982021, yeartemp_summermean))
#        winddir_summerstds19982021 = np.hstack((winddir_summerstds19982021, yeartemp_summerstd)) 

fig, axs = plt.subplots(1, 1, figsize=(11,2))
# Wind Speed
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2020), windspeed_sep19982021, marker='o', c='purple', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2020)[~np.isnan(windspeed_sep19982021)], windspeed_sep19982021[~np.isnan(windspeed_sep19982021)])
axs.plot(np.arange(1998,2020), np.arange(1998,2020)*slope+intercept, c='purple', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(windspeed_sep19982021[:8])], windspeed_sep19982021[:8][~np.isnan(windspeed_sep19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_sep19982021[8:])], windspeed_sep19982021[8:][~np.isnan(windspeed_sep19982021[8:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (m/s)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_windspeed_19802022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
fig, axs = plt.subplots(1, 1, figsize=(11,2))
# Wind Direction
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2020), winddir_sep19982021, marker='o', c='purple', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2020)[~np.isnan(winddir_sep19982021)], winddir_sep19982021[~np.isnan(winddir_sep19982021)])
axs.plot(np.arange(1998,2020), np.arange(1998,2020)*slope+intercept, c='purple', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(winddir_sep19982021[:8])], winddir_sep19982021[:8][~np.isnan(winddir_sep19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(winddir_sep19982021[8:])], winddir_sep19982021[8:][~np.isnan(winddir_sep19982021[8:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Dir (Degrees)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1982, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_winddir_19802022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()

#%%
sam_timedate = np.empty(len(sam_years), dtype=object)
for i in range(0, len(sam_timedate)):
    sam_timedate[i] = datetime.datetime(year=sam_years[i], month=sam_months[i], day=15)

fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1,285), sam_monthly_19972021, marker='o', c='r', label = 'SST', s=25, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1,285)[~np.isnan(sam_monthly_19972021)], sam_monthly_19972021[~np.isnan(sam_monthly_19972021)])
axs.plot(np.arange(1,285), np.arange(1,285)*slope+intercept, c='r', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1,101)[~np.isnan(sam_monthly_19972021[:100])], sam_monthly_19972021[:100][~np.isnan(sam_monthly_19972021[:100])])
axs.plot(np.arange(1,101), np.arange(1,101)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(101,285)[~np.isnan(sam_monthly_19972021[100:])], sam_monthly_19972021[100:][~np.isnan(sam_monthly_19972021[100:])])
axs.plot(np.arange(101,285), np.arange(101,285)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SAM', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
#plt.xlim(1983, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sepapr_SAM_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%%




## SST #19982021
sst_bransfield_19972021 = sst[clusters == 4,:]
sst_bransfield_19972021 = np.nanmean(sst_bransfield_19972021,0)
sst_bransfield_19811996 = sst_19811996[clusters == 4,:]
sst_bransfield_19811996 = np.nanmean(sst_bransfield_19811996,0)
sst_bransfield_19812021 = np.hstack((sst_bransfield_19811996, sst_bransfield_19972021))
time_date_sst_19812021 = np.hstack((time_date_19811996, time_date_sst))
time_date_years = np.empty_like(time_date_sst_19812021)
time_date_months = np.empty_like(time_date_sst_19812021)
for i in range(0, len(time_date_sst_19812021)):
    time_date_years[i] = time_date_sst_19812021[i].year
    time_date_months[i] = time_date_sst_19812021[i].month
for i in np.arange(1983, 2022): #
    yeartemp_sep = sst_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = sst_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = sst_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = sst_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = sst_bransfield_19812021[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1983:
        sst_summermeans19832021 = yeartemp_summermean
        sst_summerstds19832021 = yeartemp_summerstd
    else:
        sst_summermeans19832021 = np.hstack((sst_summermeans19832021, yeartemp_summermean))
        sst_summerstds19832021 = np.hstack((sst_summerstds19832021, yeartemp_summerstd))
## SST #19811996
sst_bransfield_19811996 = sst_19811996[clusters == 4,:]
sst_bransfield_19811996 = np.nanmean(sst_bransfield_19811996,0)
time_date_years = np.empty_like(time_date_19811996)
time_date_months = np.empty_like(time_date_19811996)
for i in range(0, len(time_date_19811996)):
    time_date_years[i] = time_date_19811996[i].year
    time_date_months[i] = time_date_19811996[i].month
for i in np.arange(1983, 1997): #
    yeartemp_sep = sst_bransfield_19811996[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = sst_bransfield_19811996[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = sst_bransfield_19811996[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = sst_bransfield_19811996[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = sst_bransfield_19811996[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = sst_bransfield_19811996[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = sst_bransfield_19811996[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = sst_bransfield_19811996[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1983:
        sst_summermeans19811996 = yeartemp_summermean
        sst_summerstds19811996 = yeartemp_summerstd
    else:
        sst_summermeans19811996 = np.hstack((sst_summermeans19811996, yeartemp_summermean))
        sst_summerstds19811996 = np.hstack((sst_summerstds19811996, yeartemp_summerstd))
## SEAICE #1998-2021
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
    time_date_years[i] = time_date_seaice_19812021[i].year
    time_date_months[i] = time_date_seaice_19812021[i].month
    time_date_days[i] = time_date_seaice_19812021[i].day
for i in np.arange(1983, 2022): #
    yeartemp_sep = seaice_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = seaice_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = seaice_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = seaice_bransfield_19812021[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = seaice_bransfield_19812021[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = seaice_bransfield_19812021[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = seaice_bransfield_19812021[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = seaice_bransfield_19812021[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1983:
        seaice_summermeans19832021 = yeartemp_summermean
        seaice_summerstds19832021 = yeartemp_summerstd
    else:
        seaice_summermeans19832021 = np.hstack((seaice_summermeans19832021, yeartemp_summermean))
        seaice_summerstds19832021 = np.hstack((seaice_summerstds19832021, yeartemp_summerstd))
## SEAICE #19811996
seaice_bransfield_19811996 = seaice_19811996[clusters == 4,:]
seaice_bransfield_19811996 = np.nanmean(seaice_bransfield_19811996,0)
time_date_years = np.empty_like(time_date_19811996)
time_date_months = np.empty_like(time_date_19811996)
for i in range(0, len(time_date_19811996)):
    time_date_years[i] = time_date_19811996[i].year
    time_date_months[i] = time_date_19811996[i].month
for i in np.arange(1983, 1997): #
    yeartemp_sep = seaice_bransfield_19811996[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = seaice_bransfield_19811996[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = seaice_bransfield_19811996[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = seaice_bransfield_19811996[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = seaice_bransfield_19811996[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = seaice_bransfield_19811996[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = seaice_bransfield_19811996[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = seaice_bransfield_19811996[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1983:
        seaice_summermeans19811996 = yeartemp_summermean
        seaice_summerstds19811996 = yeartemp_summerstd
    else:
        seaice_summermeans19811996 = np.hstack((seaice_summermeans19811996, yeartemp_summermean))
        seaice_summerstds19811996 = np.hstack((seaice_summerstds19811996, yeartemp_summerstd))


## SEA ICE ADVANCE FROM SATELLITE
for i in np.arange(1998, 2022):
    # Find index of 15 February (begins and ends annual sea ice year)
    february15_prevyear_ind = np.argwhere(time_date_seaice == datetime.datetime(year=i-1, month=2, day=15, hour=12))[0][0]
    february15_follyear_ind = np.argwhere(time_date_seaice == datetime.datetime(year=i, month=2, day=15, hour=12))[0][0]
    # Retrieve sea ice concentration data
    seaicetemp = seaice_bransfield[february15_prevyear_ind:february15_follyear_ind]
    seaicetemp_date = time_date_seaice[february15_prevyear_ind:february15_follyear_ind]
    # Find first 5 consecutives instances of seaice above 15%        
    ice_index = seaicetemp > 15
    breaker_a = 0
    for l, ind in enumerate(ice_index):
        if ind == False:
            continue
        elif (breaker_a == 0) & (all(x == True for x in ice_index[l:l+5])):
            seaice_init_index = l
            seaice_init_date = seaicetemp_date[l].timetuple().tm_yday
            if seaice_init_date < 46:
                seaice_init_date = 365 + seaice_init_date
            breaker_a=1    
    if i == 1998:
        seaice_advance19982021 = seaice_init_date
    else:
        seaice_advance19982021 = np.hstack((seaice_advance19982021, seaice_init_date))
## SEA ICE RETREAT FROM SATELLITE
for i in np.arange(1998, 2022):
    # Find index of 15 February (begins and ends annual sea ice year)
    february15_prevyear_ind = np.argwhere(time_date_seaice == datetime.datetime(year=i-1, month=2, day=15, hour=12))[0][0]
    february15_follyear_ind = np.argwhere(time_date_seaice == datetime.datetime(year=i, month=2, day=15, hour=12))[0][0]
    # Retrieve sea ice concentration data
    seaicetemp = seaice_bransfield[february15_prevyear_ind:february15_follyear_ind]
    seaicetemp = seaicetemp[::-1]
    seaicetemp_date = time_date_seaice[february15_prevyear_ind:february15_follyear_ind]
    seaicetemp_date = seaicetemp_date[::-1]
    # Find first 5 consecutives instances of seaice above 15%        
    ice_index = seaicetemp > 15
    breaker_a = 0
    for l, ind in enumerate(ice_index):
        if ind == False:
            continue
        elif (breaker_a == 0) & (all(x == True for x in ice_index[l:l+5])):
            seaice_term_index = l
            seaice_term_date = seaicetemp_date[l].timetuple().tm_yday
            if seaice_term_date < 46:
                seaice_term_date = 365 + seaice_term_date
            breaker_a=1    
    if i == 1998:
        seaice_retreat19982021 = seaice_term_date
    else:
        seaice_retreat19982021 = np.hstack((seaice_retreat19982021, seaice_term_date))        
## SEA ICE DURATION FROM SATELLITE
seaice_duration_19982021 = seaice_retreat19982021 - seaice_advance19982021    
## PAR #1998-2021
par_bransfield = par[clusters == 4,:]
par_bransfield = np.nanmean(par_bransfield,0)
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = par_bransfield[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = par_bransfield[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = par_bransfield[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = par_bransfield[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = par_bransfield[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = par_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = par_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = par_bransfield[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1998:
        par_summermeans19982021 = yeartemp_summermean
        par_summerstds19982021 = yeartemp_summerstd
    else:
        par_summermeans19982021 = np.hstack((par_summermeans19982021, yeartemp_summermean))
        par_summerstds19982021 = np.hstack((par_summerstds19982021, yeartemp_summerstd))        
## MEI #1998-2021
for i in np.arange(1998, 2022):
    yeartemp_sep = mei_monthly_19972021[(mei_years == i-1) & (mei_months == 9)]
    yeartemp_oct = mei_monthly_19972021[(mei_years == i-1) & (mei_months == 10)]
    yeartemp_nov = mei_monthly_19972021[(mei_years == i-1) & (mei_months == 11)]
    yeartemp_dec = mei_monthly_19972021[(mei_years == i-1) & (mei_months == 12)]
    yeartemp_jan = mei_monthly_19972021[(mei_years == i) & (mei_months == 1)]
    yeartemp_feb = mei_monthly_19972021[(mei_years == i) & (mei_months == 2)]
    yeartemp_mar = mei_monthly_19972021[(mei_years == i) & (mei_months == 3)]
    yeartemp_apr = mei_monthly_19972021[(mei_years == i) & (mei_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1998:
        mei_summermeans19982021 = yeartemp_summermean
        mei_summerstds19982021 = yeartemp_summerstd
    else:
        mei_summermeans19982021 = np.hstack((mei_summermeans19982021, yeartemp_summermean))
        mei_summerstds19982021 = np.hstack((mei_summerstds19982021, yeartemp_summerstd)) 

## WINDS #1998-2021
# Define region of interest to average
Bransfield_winds_verts = [(-65, -64.5),
             (-65, -63),
             (-52.5, -60),
             (-52.5, -61.6)]
#plt.figure()
#map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
#map.coastlines(resolution='10m', color='black', linewidth=1)
#map.set_extent([-68, -50, -67, -60])
#map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
#                                        edgecolor='k',
#                                        facecolor=cartopy.feature.COLORS['land']))
#gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
#poly_CBS = Polygon(list(Bransfield_winds_verts), facecolor=[1,1,1,0], edgecolor='k', linewidth=1, linestyle='--', zorder=2, transform=ccrs.PlateCarree())
#plt.gca().add_patch(poly_CBS)
#plt.tight_layout()
## Northward wind
x, y = np.meshgrid(lon_winds, lat_winds) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(Bransfield_winds_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon_winds), len(lat_winds))
mask3d = np.repeat([mask], np.size(winds_northward,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
winds_northward_bransfield = np.ma.array(winds_northward, mask=~mask3d)
winds_northward_bransfield = np.nanmean(winds_northward_bransfield, (0,1))
## Eastward wind
x, y = np.meshgrid(lon_winds, lat_winds) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(Bransfield_winds_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon_winds), len(lat_winds))
mask3d = np.repeat([mask], np.size(winds_eastward,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
winds_eastward_bransfield = np.ma.array(winds_eastward, mask=~mask3d)
winds_eastward_bransfield = np.nanmean(winds_eastward_bransfield, (0,1))
## Calculate wind speed
wind_speed = np.sqrt(winds_eastward_bransfield**2 + winds_northward_bransfield**2)
wind_dir = np.empty_like(wind_speed)
for k in range(0, len(wind_dir)):
    wind_dir_trig_to = math.atan2(winds_eastward_bransfield[k]/wind_speed[k], winds_northward_bransfield[k]/wind_speed[k]) 
    wind_dir_trig_to_degrees = wind_dir_trig_to * 180/math.pi
    wind_dir[k] = wind_dir_trig_to_degrees + 180
time_date_years = time_date_winds.astype('datetime64[Y]').astype(int) + 1970
time_date_months = time_date_winds.astype('datetime64[M]').astype(int) % 12 + 1 
## WIND SPEED #1998-2019
for i in np.arange(1998, 2020):
    yeartemp_sep = wind_speed[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wind_speed[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wind_speed[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wind_speed[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wind_speed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wind_speed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wind_speed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wind_speed[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1998:
        windspeed_summermeans19982021 = yeartemp_summermean
        windspeed_summerstds19982021 = yeartemp_summerstd
    else:
        windspeed_summermeans19982021 = np.hstack((windspeed_summermeans19982021, yeartemp_summermean))
        windspeed_summerstds19982021 = np.hstack((windspeed_summerstds19982021, yeartemp_summerstd)) 
## WIND DIR #1998-2019
for i in np.arange(1998, 2020):
    yeartemp_sep = wind_dir[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wind_dir[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wind_dir[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wind_dir[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wind_dir[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wind_dir[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wind_dir[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wind_dir[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_sep, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar,
                                                yeartemp_apr)))    
    if i == 1998:
        winddir_summermeans19982021 = yeartemp_summermean
        winddir_summerstds19982021 = yeartemp_summerstd
    else:
        winddir_summermeans19982021 = np.hstack((winddir_summermeans19982021, yeartemp_summermean))
        winddir_summerstds19982021 = np.hstack((winddir_summerstds19982021, yeartemp_summerstd)) 
## Make all the same lenght and add to dataframe (1998-2019)
dataframe_bransfield = pd.DataFrame(data=chl_summermeans19982021[:-2], index=np.arange(1998,2020), columns=['Chla Mean'])
#dataframe_bransfield['Chla STD'] = chl_summerstds19982021[:-2]
dataframe_bransfield['SST Mean'] = sst_summermeans19982021[:-2]
#dataframe_bransfield['SST STD'] = sst_summerstds19982021[:-2]
dataframe_bransfield['SEAICE Mean'] = seaice_summermeans19982021[:-2]
#dataframe_bransfield['SEAICE STD'] = seaice_summerstds19982021[:-2]
dataframe_bransfield['PAR Mean'] = par_summermeans19982021[:-2]
#dataframe_bransfield['PAR STD'] = par_summerstds19982021[:-2]
dataframe_bransfield['SEAICE ADV SAT'] = seaice_advance19982021[:-2]
dataframe_bransfield['SEAICE RET SAT'] = seaice_retreat19982021[:-2]
#dataframe_bransfield['SEAICE DUR SAT'] = seaice_duration_19982021[:-2]
dataframe_bransfield['WSpeed Mean'] = windspeed_summermeans19982021
#dataframe_bransfield['WSpeed STD'] = windspeed_summerstds19982021
dataframe_bransfield['WDir Mean'] = winddir_summermeans19982021
#dataframe_bransfield['WDir STD'] = winddir_summerstds19982021
dataframe_bransfield['MEI Mean'] = mei_summermeans19982021[:-2]
#dataframe_bransfield['MEI STD'] = mei_summerstds19982021[:-2]
dataframe_bransfield['SAM Mean'] = sam_summermeans19982021[:-2]
#dataframe_bransfield['SAM STD'] = sam_summerstds19982021[:-2]
#dataframe_bransfield['SEAICE ADV PALMER'] = seaice_advanceday.values[18:-2]
#dataframe_bransfield['SEAICE RET PALMER'] = seaice_retreatday.values[18:-2]
#dataframe_bransfield['SEAICE DUR PALMER'] = seaice_duration.values[18:-2]
#dataframe_bransfield['SEAICE EXT PALMER'] = seaice_extent.values[18:-2]
#dataframe_bransfield['B Init'] = b_init[:-2]
#dataframe_bransfield['B Term'] = b_term[:-2]
#dataframe_bransfield['B Peak'] = b_peak[:-2]
#dataframe_bransfield['B Dur'] = b_dur[:-2]
#dataframe_bransfield['B Area'] = b_area[:-2]
#dataframe_bransfield['Chla Max'] = chl_max[:-2]
#%%
## Plot Correlation Matrix
corr = dataframe_bransfield.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, mask = mask, cmap=cmocean.cm.balance, vmin=-1, vmax=1)
#plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\randomforests\\bransfield\\annual\\')
graphs_dir = 'bransfield.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Run model
## Check co-correlation between variables
correlated_features = set()
## Keep only non correlated variables
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) >= 0.7:
            colname = corr.columns[i]
            correlated_features.add(colname)
#dataframe_all_original = dataframe_all
dataframe_all = dataframe_bransfield.dropna()
#f, ax = plt.subplots(figsize=(9, 6))
#sns.heatmap(correlation_matrix, annot=True, linewidths=.5, ax=ax,
#            vmin=-1,vmax=1,cmap=sns.diverging_palette(220, 20, n=256))
#b, t = plt.ylim() # discover the values for bottom and top
#b += 0.5 # Add 0.5 to the bottom
#t -= 0.5 # Subtract 0.5 from the top
#plt.ylim(b, t) # update the ylim(bottom, top) values
#plt.tight_layout()
### REMOVE RESPONSE VARIABLE ###
features = dataframe_all
X = features.drop(['Chla Mean'],axis=1)
features_list = features.columns
y = dataframe_all['Chla Mean']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500, random_state=0)
regressor.fit(X, y)
y_pred = regressor.predict(X)
from sklearn import metrics
# Metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))
print('Score Train: ', regressor.score(X_train, y_train))
print('Score Full: ', regressor.score(X, y))
#print('Score Test: ', regressor.score(X_test, y_test))
#%%
# Variable importance
from sklearn.inspection import permutation_importance
scoring = ['r2']
r_multi = permutation_importance(regressor, X, y,
                           n_repeats=100,
                           random_state=23, scoring=scoring)
for metric in r_multi:
    print(f"{metric}")
    r = r_multi[metric]
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X.columns[i]:<8}  " 
                  f"{r.importances_mean[i]:.3f}  " 
                  f" +/- {r.importances_std[i]:.3f}")
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\randomforests\\bransfield\\annual')
currentmodel_permut = pd.read_csv('chlmean_permutation_bransfield.csv', sep=";")
# Load sardine data
variables_names = currentmodel_permut['Variable'].values
variables_importances = currentmodel_permut['Importance'].values
variables_stds = currentmodel_permut['Std'].values
#%%
fig, axs = plt.subplots(1, 2, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs[1].plot(dataframe_all.index, dataframe_all['Chla Mean'].values, marker='^', c='#36454F', label = 'Observed Chla', markersize=2, alpha=0.5)
# Plot the predicted values
axs[1].scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
axs[1].legend()
axs[1].set_xlabel('Years', fontsize=14)
axs[1].set_ylabel('Chla', fontsize=14)
#axs[1].set_title('Predictions', fontsize=14)
#from partial_dependence import PartialDependenceExplainer
axs[0].barh(variables_names[::-1], variables_importances[::-1], xerr=variables_stds[::-1], facecolor='#9800cb', edgecolor='k')
#axs[0].axhline(y=10.5, c='#36454F', alpha=0.5)
#axs[0].axhspan(10.4, 15, facecolor='#36454F', alpha=0.2)
#axs[0].set_ylim(-1, 15)
axs[0].text(0.1, 0, f"R2={regressor.score(X, y):.2f}", fontsize=14)
axs[0].text(0.1, -.5, f"MAE={metrics.mean_absolute_error(y, y_pred):.2f}", fontsize=14)
#axs[0].axvline(x=np.mean(predictor_importances['Importance']), c='#A50021', linestyle='--')
axs[0].set_xlabel('Variable Importance', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_randomforestresults.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()


#%% Plot Figure 7
from sklearn.inspection import plot_partial_dependence
#fig = plt.figure(figsize=(9,9))
plot_partial_dependence(regressor,X, ['SST Mean'], kind='average', 
                        line_kw={"color": "k", "linewidth": 4}
                        )
#plt.yticks(ticks=[np.log10(800), np.log10(1000), np.log10(1200), np.log10(1400), np.log10(1600), np.log10(1800), np.log10(2000)],
#           labels=[800, 1000, 1200, 1400, 1600, 1800, 2000])
#plt.ylim(2.9,3.2)
#plt.xlabel(fontsize=14)
#plt.ylabel(fontsize=14)
#axs[0].set_xlabel('Fav. Window (n)', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_randomforest_SST.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Plot interannual means
# CHL
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), chl_summermeans19982021, marker='^', c='#36454F', label = 'Observed Chla', markersize=2, alpha=0.5)
[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(chl_summermeans19982021)], chl_summermeans19982021[~np.isnan(chl_summermeans19982021)])
axs.text(1998, 1.6, f"R={r:.2f}", fontsize=14)
axs.text(1998, 1.5, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Chla', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_chl19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SST
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), sst_summermeans19982021, marker='^', c='#36454F', label = 'Observed Chla', markersize=2, alpha=0.5)
[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(sst_summermeans19982021)], sst_summermeans19982021[~np.isnan(sst_summermeans19982021)])
axs.text(1998, .5, f"R={r:.2f}", fontsize=14)
axs.text(1998, .45, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('SST', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_sst19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SEAICE CONC
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), seaice_summermeans19982021, marker='^', c='#36454F', label = 'Observed Chla', markersize=2, alpha=0.5)
[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_summermeans19982021)], seaice_summermeans19982021[~np.isnan(seaice_summermeans19982021)])
axs.text(1998, 4, f"R={r:.2f}", fontsize=14)
axs.text(1998, 3, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Sea Ice Concentration', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_seaice19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# PAR
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), par_summermeans19982021, marker='^', c='#36454F', label = 'Observed Chla', markersize=2, alpha=0.5)
[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(par_summermeans19982021)], par_summermeans19982021[~np.isnan(par_summermeans19982021)])
axs.text(1998, 28, f"R={r:.2f}", fontsize=14)
axs.text(1998, 27.5, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('PAR', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_par19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SEA ICE ADVANCE
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), seaice_advance19982021, marker='^', c='#36454F', label = 'Sea Ice Adv Bra', markersize=2, alpha=0.5)
axs.plot(np.arange(1998,2022), seaice_advanceday.values[18:], marker='s', c='r', label = 'Sea Ice Adv Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advance19982021)], seaice_advance19982021[~np.isnan(seaice_advance19982021)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1998, 135, f"R={r:.2f}", fontsize=14)
axs.text(1998, 125, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Sea Ice Adv Day', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_seaiceadvance19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SEA ICE RETREAT
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), seaice_retreat19982021, marker='^', c='#36454F', label = 'Sea Ice Ret Bra', markersize=2, alpha=0.5)
axs.plot(np.arange(1998,2022), seaice_retreatday.values[18:], marker='s', c='r', label = 'Sea Ice Ret Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_retreat19982021)], seaice_retreat19982021[~np.isnan(seaice_retreat19982021)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1998, 265, f"R={r:.2f}", fontsize=14)
axs.text(1998, 255, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Sea Ice Ret Day', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_seaiceretreat19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SEA ICE DURATION
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), seaice_duration_19982021, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
axs.plot(np.arange(1998,2022), seaice_duration.values[18:], marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_duration_19982021)], seaice_duration_19982021[~np.isnan(seaice_duration_19982021)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1998, 200, f"R={r:.2f}", fontsize=14)
axs.text(1998, 190, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Sea Ice Dur', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_seaiceduration19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# WIND SPEED
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2020), windspeed_summermeans19982021, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
#axs.plot(np.arange(1998,2022), seaice_duration.values[18:], marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2020)[~np.isnan(windspeed_summermeans19982021)], windspeed_summermeans19982021[~np.isnan(windspeed_summermeans19982021)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1998, 6.75, f"R={r:.2f}", fontsize=14)
axs.text(1998, 6.6, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2020), np.arange(1998,2020)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Wind Speed', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_windspeed19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# WIND DIRECTION
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2020), winddir_summermeans19982021, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
#axs.plot(np.arange(1998,2022), seaice_duration.values[18:], marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2020)[~np.isnan(winddir_summermeans19982021)], winddir_summermeans19982021[~np.isnan(winddir_summermeans19982021)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1998, 210, f"R={r:.2f}", fontsize=14)
axs.text(1998, 205, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2020), np.arange(1998,2020)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Wind Direction', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_winddirection19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# MEI
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), mei_summermeans19982021, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
#axs.plot(np.arange(1998,2022), seaice_duration.values[18:], marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(mei_summermeans19982021)], mei_summermeans19982021[~np.isnan(mei_summermeans19982021)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

#axs.text(1998, 210, f"R={r:.2f}", fontsize=14)
#axs.text(1998, 205, f"P-val={pval:.2f}", fontsize=14)
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('MEI', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_mei19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SAM
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), sam_summermeans19982021, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
#axs.plot(np.arange(1998,2022), seaice_duration.values[18:], marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(sam_summermeans19982021)], sam_summermeans19982021[~np.isnan(sam_summermeans19982021)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

#axs.text(1998, 210, f"R={r:.2f}", fontsize=14)
#axs.text(1998, 205, f"P-val={pval:.2f}", fontsize=14)
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('SAM', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_sam19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# BLOOM INIT
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), b_init, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
#axs.plot(np.arange(1998,2022), seaice_duration.values[18:], marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(b_init)], b_init[~np.isnan(b_init)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1998, 22, f"R={r:.2f}", fontsize=14)
axs.text(1998, 21, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Bloom Init Week', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_bloominit19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# BLOOM TERM
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), b_term, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
#axs.plot(np.arange(1998,2022), seaice_duration.values[18:], marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(b_term)], b_term[~np.isnan(b_term)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1998.5, 22, f"R={r:.2f}", fontsize=14)
axs.text(1998.5, 21.2, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Bloom Term Week', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_bloomterm19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# BLOOM DUR
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), b_dur, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
#axs.plot(np.arange(1998,2022), seaice_duration.values[18:], marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(b_dur)], b_dur[~np.isnan(b_dur)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1998, 19, f"R={r:.2f}", fontsize=14)
axs.text(1998, 18, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Bloom Dur', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_bloomdur19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# BLOOM AREA
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), b_area, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
#axs.plot(np.arange(1998,2022), seaice_duration.values[18:], marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(b_area)], b_area[~np.isnan(b_area)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1998, 1.8, f"R={r:.2f}", fontsize=14)
axs.text(1998, 1.7, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Bloom Area', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_bloomarea19982021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Larger dataset trends
# SEA ICE ADVANCE
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), seaice_advance19982021, marker='^', c='#36454F', label = 'Sea Ice Adv Bra', markersize=2, alpha=0.5)
axs.plot(np.arange(1980,2022), seaice_advanceday.values, marker='s', c='r', label = 'Sea Ice Adv Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1980, 2022)[~np.isnan(seaice_advanceday)], seaice_advanceday[~np.isnan(seaice_advanceday)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1980, 190, f"R={r:.2f}", fontsize=14)
axs.text(1980, 180, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1980,2022), np.arange(1980,2022)*slope+intercept, linestyle='--', c='r')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Sea Ice Adv Day', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_seaiceadvancepalmer19802021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SEA ICE RETREAT
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), seaice_retreat19982021, marker='^', c='#36454F', label = 'Sea Ice Ret Bra', markersize=2, alpha=0.5)
axs.plot(np.arange(1980,2022), seaice_retreatday.values, marker='s', c='r', label = 'Sea Ice Adv Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1980, 2022)[~np.isnan(seaice_retreatday)], seaice_retreatday[~np.isnan(seaice_retreatday)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1980, 265, f"R={r:.2f}", fontsize=14)
axs.text(1980, 255, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1980,2022), np.arange(1980,2022)*slope+intercept, linestyle='--', c='r')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Sea Ice Ret Day', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_seaiceretreat19802021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SEA ICE DURATION
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1998,2022), seaice_duration_19982021, marker='^', c='#36454F', label = 'Sea Ice Dur Bra', markersize=2, alpha=0.5)
axs.plot(np.arange(1980,2022), seaice_duration.values, marker='s', c='r', label = 'Sea Ice Dur Palmer', markersize=2, alpha=0.5)


[slope,intercept,r,pval,_] = stats.linregress(np.arange(1980, 2022)[~np.isnan(seaice_duration)], seaice_duration[~np.isnan(seaice_duration)])
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(seaice_advanceday.values[18:])], seaice_advanceday.values[18:][~np.isnan(seaice_advanceday.values[18:])])

axs.text(1980, 100, f"R={r:.2f}", fontsize=14)
axs.text(1980, 85, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1980,2022), np.arange(1980,2022)*slope+intercept, linestyle='--', c='r')
#axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, linestyle='--', c='b')

# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
axs.legend(loc=4)
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Sea Ice Dur', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_seaiceduration19802021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SST 1982-2021
#sst_summermeans19812021 = np.hstack((sst_summermeans19811996, sst_summermeans19982021))
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1983,2022), sst_summermeans19832021, marker='^', c='#36454F', label = 'Observed Chla', markersize=2, alpha=0.5)
[slope,intercept,r,pval,_] = stats.linregress(np.arange(1983, 2022)[~np.isnan(sst_summermeans19832021)], sst_summermeans19832021[~np.isnan(sst_summermeans19832021)])
axs.text(1983, .5, f"R={r:.2f}", fontsize=14)
axs.text(1983, .45, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1983,2022), np.arange(1983,2022)*slope+intercept, linestyle='--', c='b')
# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('SST', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_sst19832021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# SEAICE CONC
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1983,2022), seaice_summermeans19832021, marker='^', c='#36454F', label = 'Observed Chla', markersize=2, alpha=0.5)
[slope,intercept,r,pval,_] = stats.linregress(np.arange(1983, 2022)[~np.isnan(seaice_summermeans19832021)], seaice_summermeans19832021[~np.isnan(seaice_summermeans19832021)])
axs.text(1983, 4, f"R={r:.2f}", fontsize=14)
axs.text(1983, 3, f"P-val={pval:.2f}", fontsize=14)
axs.plot(np.arange(1983,2022), np.arange(1983,2022)*slope+intercept, linestyle='--', c='b')
# Plot the predicted values
#axs.scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c='#9800cb', s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
#axs.legend()
axs.set_xlabel('Years', fontsize=14)
axs.set_ylabel('Sea Ice Concentration', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'bransfield_seaice19832021_sepaprmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()