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
# BLOOM METRICS gerlache
fh = np.load('phenology_gerlache_10km.npz', allow_pickle=True)
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
sam_monthly_19972021 = sam_pd['SAM'][8:-8].values
sam_months = sam_pd['Month'][8:-8].values
sam_years = sam_pd['Year'][8:-8].values
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
seaice_advanceday = ice_timing['Pdsr adv']
seaice_retreatday = ice_timing['Pdsr ret']
seaice_duration = ice_timing['Pdsr dur']
# Sea Ice Extent from PALMER LTER
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaiceextent')
seaice_extent_df = pd.read_csv('seaiceextent.csv', sep=';')
seaice_extent_years = seaice_extent_df['Year']
seaice_extent = seaice_extent_df['Dsr_Ext']
#%% Create dataframe for gerlache using spring-summer data (September-April means)
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
# WIND SPEED
gerlache_winds_verts = [(-68, -66),
                        (-64.5, -64.6),
                        (-64, -65),
             (-61.5, -64),
             (-61, -63.8),
             (-61, -64.25),
             (-64, -65.2),
             (-67.25, -66.8)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-68, -50, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
poly_CBS = Polygon(list(gerlache_winds_verts), facecolor=[1,1,1,0], edgecolor='k', linewidth=1, linestyle='--', zorder=2, transform=ccrs.PlateCarree())
plt.gca().add_patch(poly_CBS)
plt.tight_layout()
## Northward wind
x, y = np.meshgrid(lon_winds, lat_winds) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(gerlache_winds_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon_winds), len(lat_winds))
mask3d = np.repeat([mask], np.size(winds_northward,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
winds_northward_gerlache = np.ma.array(winds_northward, mask=~mask3d)
winds_northward_gerlache = np.nanmean(winds_northward_gerlache, (0,1))
## Eastward wind
x, y = np.meshgrid(lon_winds, lat_winds) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(gerlache_winds_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon_winds), len(lat_winds))
mask3d = np.repeat([mask], np.size(winds_eastward,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
winds_eastward_gerlache = np.ma.array(winds_eastward, mask=~mask3d)
winds_eastward_gerlache = np.nanmean(winds_eastward_gerlache, (0,1))
## Calculate wind speed
wind_speed = np.sqrt(winds_eastward_gerlache**2 + winds_northward_gerlache**2)
wind_dir = np.empty_like(wind_speed)
for k in range(0, len(wind_dir)):
    wind_dir_trig_to = math.atan2(winds_eastward_gerlache[k]/wind_speed[k], winds_northward_gerlache[k]/wind_speed[k]) 
    wind_dir_trig_to_degrees = wind_dir_trig_to * 180/math.pi
    wind_dir[k] = wind_dir_trig_to_degrees + 180
time_date_years = time_date_winds.astype('datetime64[Y]').astype(int) + 1970
time_date_months = time_date_winds.astype('datetime64[M]').astype(int) % 12 + 1 
for i in np.arange(1998, 2020):
    ix = pd.date_range(start=datetime.date(i-1, 8, 1), end=datetime.date(i, 6, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_aug = 0
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = wind_speed[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_winds[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)
    else:
        yeartemp_aug = np.where((time_date_years == i-1) & (time_date_months == 5))[0][0]
        yeartemp_may = np.where((time_date_years == i) & (time_date_months == 6))[-1][-1]
        yeartemp_augmay = wind_speed[yeartemp_aug:yeartemp_may+1]
        yeartemp_augmay_pd = pd.Series(yeartemp_augmay, index=time_date_winds[yeartemp_aug:yeartemp_may+1])
        yeartemp_augmay_pd = yeartemp_augmay_pd.reindex(ix)   
    if (i in leapyears_list):
        yeartemp_augmay_pd = yeartemp_augmay_pd[~((yeartemp_augmay_pd.index.month == 2) & (yeartemp_augmay_pd.index.day == 29))]
    yeartemp_augmay_pd_8day = yeartemp_augmay_pd.resample('8D').mean()
    if i == 1998:
        windspeed_sepapr_19982021 = yeartemp_augmay_pd_8day.values
    else:
        windspeed_sepapr_19982021 = np.vstack((windspeed_sepapr_19982021, yeartemp_augmay_pd_8day.values))
windspeed_sepapr_19982021_mean = np.nanmean(windspeed_sepapr_19982021,0)
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'gerlache_sepapr_climatology.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Plot figure
#
chl_sepapr_19982021_init = np.empty(len(b_init))
chl_sepapr_19982021_term = np.empty(len(b_term))

for i in range(0,np.size(chl_sepapr_19982021,0)):
    chl_sepapr_19982021_init[i] = np.where(chl_sepapr_19982021[i,:] > 0)[0][0]
    chl_sepapr_19982021_term[i] = np.where(chl_sepapr_19982021[i,:] > 0)[0][-1]
# Bloom Fenology
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998, 2022), b_init, marker='o', c='k', label = 'Chla')
axs.scatter(np.arange(1998, 2022), b_term, marker='o', c='k', label = 'Chla')
axs.vlines(x=np.arange(1998, 2022), ymin=b_init, ymax=b_term, colors='k', lw=2, alpha=0.3)
axs.scatter(np.arange(1998, 2022), b_peak, marker='.', c='r', label = 'Chla')
axs.plot(np.arange(1998, 2022), chl_sepapr_19982021_init, c='g', label = 'Chla', linestyle=':', alpha=0.7)
axs.plot(np.arange(1998, 2022), chl_sepapr_19982021_term, c='g', label = 'Chla', linestyle=':', alpha=0.7)
axs.set_yticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                     'Jun', 'Jul'], fontsize=10)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'gerlache_sepapr_phenology.png'
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison_prepost2005_seaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#Wind Speed
windspeed_sepapr_19982003 = np.nanmean(windspeed_sepapr_19982021[:6, :],0)
windspeed_sepapr_20042009 = np.nanmean(windspeed_sepapr_19982021[6:12, :],0)
windspeed_sepapr_20102015 = np.nanmean(windspeed_sepapr_19982021[12:18, :],0)
windspeed_sepapr_20162021 = np.nanmean(windspeed_sepapr_19982021[18:, :],0)
windspeed_sepapr_19982005 = np.nanmean(windspeed_sepapr_19982021[:8, :],0)
windspeed_sepapr_20062021 = np.nanmean(windspeed_sepapr_19982021[8:, :],0)
# Plot windspeed
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), windspeed_sepapr_19982003, marker='^', c='purple', label = '1998-2003', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), windspeed_sepapr_20042009, marker='s', c='purple', label = '2004-2009', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
axs.plot(np.arange(1,43), windspeed_sepapr_20102015, linestyle='--', c='purple', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
axs.plot(np.arange(1,43), windspeed_sepapr_20162021, marker='*', linestyle=':', c='purple', label = '2016-2019', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
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
axs.set_ylabel('Wind Speed (ms$^{-1}$)', fontsize=12)
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison5years_windpseed.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Comparing just prior and post 2005
fig, axs = plt.subplots(1, 1, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.plot(np.arange(1,43), windspeed_sepapr_19982005, marker='^', c='purple', label = '1998-2005', markersize=7.5, alpha=.5, zorder=10)
#axs.set_zorder(10)
#ax2 = axs.twinx()
axs.plot(np.arange(1,43), windspeed_sepapr_20062021, marker='s', c='purple', label = '2006-2019', markersize=7.5, alpha=.5,zorder=9)
#ax3 = axs.twinx()
#axs.plot(np.arange(1,43), windspeed_sepapr_20102015, linestyle='--', c='purple', label = '2010-2015', markersize=7.5, alpha=.5)
#ax4 = axs.twinx()
#axs.plot(np.arange(1,43), windspeed_sepapr_20162021, marker='*', linestyle=':', c='purple', label = '2016-2021', markersize=7.5, alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
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
axs.set_ylabel('Wind Speed (ms$^{-1}$)', fontsize=12)
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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'gerlache_sepapr_climatology_comparison_prepost2005_windspeed.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()

#%% Setember-Dec Mean from 1981 to 2021
# CHL
chl_gerlache = chl[clusters == 2,:]
chl_gerlache = np.nanmean(chl_gerlache,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1997, 2022):
    yeartemp_sep = chl_gerlache[(time_date_years == i) & (time_date_months == 9)]
    yeartemp_oct = chl_gerlache[(time_date_years == i) & (time_date_months == 10)]
    yeartemp_nov = chl_gerlache[(time_date_years == i) & (time_date_months == 11)]
    yeartemp_dec = chl_gerlache[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_gerlache[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_gerlache[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = chl_gerlache[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = chl_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
    yeartemp_sep = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 9)]
    yeartemp_oct = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 10)]
    yeartemp_nov = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 11)]
    yeartemp_dec = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_gerlache[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_gerlache[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = chl_gerlache[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = chl_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
par_gerlache = par[clusters == 2,:]
par_gerlache = np.nanmean(par_gerlache,0)
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1997, 2022):
    yeartemp_sep = par_gerlache[(time_date_years == i) & (time_date_months == 9)]
    yeartemp_oct = par_gerlache[(time_date_years == i) & (time_date_months == 10)]
    yeartemp_nov = par_gerlache[(time_date_years == i) & (time_date_months == 11)]
    yeartemp_dec = par_gerlache[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = par_gerlache[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = par_gerlache[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = par_gerlache[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = par_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
graphs_dir = 'gerlache_sepdec_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Setember Mean from 1981 to 2021
# CHL
chl_gerlache = chl[clusters == 2,:]
chl_gerlache = np.nanmean(chl_gerlache,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1997, 2022):
    yeartemp_sep = chl_gerlache[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = chl_gerlache[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = chl_gerlache[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = chl_gerlache[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_gerlache[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_gerlache[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = chl_gerlache[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = chl_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
    yeartemp_sep = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_gerlache[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_gerlache[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = chl_gerlache[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = chl_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
par_gerlache = par[clusters == 2,:]
par_gerlache = np.nanmean(par_gerlache,0)
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1997, 2022):
    yeartemp_sep = par_gerlache[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = par_gerlache[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = par_gerlache[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = par_gerlache[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = par_gerlache[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = par_gerlache[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = par_gerlache[(time_date_years == i) & (time_date_months == 3)]
#    yeartemp_apr = par_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
graphs_dir = 'gerlache_sep_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()

#%%
stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_sep19982021)], chl_sep19982021[~np.isnan(chl_sep19982021)])
stats.linregress(np.arange(1997, 2022)[~np.isnan(par_sep19982021)], par_sep19982021[~np.isnan(par_sep19982021)])
stats.linregress(np.arange(2005, 2022)[~np.isnan(sst_sep19812021[23:])], sst_sep19812021[23:][~np.isnan(sst_sep19812021[23:])])
stats.linregress(np.arange(2005, 2022)[~np.isnan(chl_sep19982021[8:])], chl_sep19982021[8:][~np.isnan(chl_sep19982021[8:])])

#%% Apr-May Mean from 1981 to 2021
# CHL
chl_gerlache = chl[clusters == 2,:]
chl_gerlache = np.nanmean(chl_gerlache,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1998, 2022):
#    yeartemp_sep = chl_gerlache[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = chl_gerlache[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = chl_gerlache[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = chl_gerlache[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_gerlache[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_gerlache[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = chl_gerlache[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = chl_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
#    yeartemp_sep = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = chl_gerlache[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = chl_gerlache[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 4)]
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
par_gerlache = par[clusters == 2,:]
par_gerlache = np.nanmean(par_gerlache,0)
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
#    yeartemp_sep = par_gerlache[(time_date_years == i) & (time_date_months == 9)]
#    yeartemp_oct = par_gerlache[(time_date_years == i) & (time_date_months == 10)]
#    yeartemp_nov = par_gerlache[(time_date_years == i) & (time_date_months == 11)]
#    yeartemp_dec = par_gerlache[(time_date_years == i) & (time_date_months == 12)]
#    yeartemp_jan = par_gerlache[(time_date_years == i) & (time_date_months == 1)]
#    yeartemp_feb = par_gerlache[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = par_gerlache[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = par_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
graphs_dir = 'gerlache_marapr_19822022.png'
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
graphs_dir = 'gerlache_sep_chl_19822022.png'
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
graphs_dir = 'gerlache_sep_par_19822022.png'
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
graphs_dir = 'gerlache_sep_sst_19822022.png'
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
graphs_dir = 'gerlache_sep_seaiceretreat_19822022.png'
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
graphs_dir = 'gerlache_sep_seaiceextent_19822022.png'
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
graphs_dir = 'gerlache_marapr_chl_19822022.png'
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
graphs_dir = 'gerlache_marapr_par_19822022.png'
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
graphs_dir = 'gerlache_marapr_sst_19822022.png'
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
axs.set_ylabel('Sea Ice Ret. Day', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'marapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1980, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'gerlache_marapr_seaiceadvance_19822022.png'
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
graphs_dir = 'gerlache_marapr_seaiceextent_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Setember-April Mean from 1981 to 2021
# CHL
chl_gerlache = chl[clusters == 2,:]
chl_gerlache = np.nanmean(chl_gerlache,0)
time_date_years = np.empty_like(time_date_chl)
time_date_months = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years[i] = time_date_chl[i].year
    time_date_months[i] = time_date_chl[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = chl_gerlache[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = chl_gerlache[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = chl_gerlache[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = chl_gerlache[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = chl_gerlache[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl_gerlache[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = chl_gerlache[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = chl_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
for i in np.arange(1983, 2022):
    yeartemp_sep = sst_gerlache_19812021[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = sst_gerlache_19812021[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = sst_gerlache_19812021[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = sst_gerlache_19812021[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = sst_gerlache_19812021[(time_date_years == i) & (time_date_months == 4)]
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
par_gerlache = par[clusters == 2,:]
par_gerlache = np.nanmean(par_gerlache,0)
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = par_gerlache[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = par_gerlache[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = par_gerlache[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = par_gerlache[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = par_gerlache[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = par_gerlache[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = par_gerlache[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = par_gerlache[(time_date_years == i) & (time_date_months == 4)]
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
graphs_dir = 'gerlache_sepapr_19822022.png'
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
graphs_dir = 'gerlache_sepapr_chl_19822022.png'
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
graphs_dir = 'gerlache_sepapr_par_19822022.png'
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
graphs_dir = 'gerlache_sepapr_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% SAM
## SAM #1998-2021
for i in np.arange(1998, 2022):
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
    if i == 1998:
        sam_summermeans19982021 = yeartemp_summermean
        sam_summerstds19982021 = yeartemp_summerstd
    else:
        sam_summermeans19982021 = np.hstack((sam_summermeans19982021, yeartemp_summermean))
        sam_summerstds19982021 = np.hstack((sam_summerstds19982021, yeartemp_summerstd)) 
# SAM sam_monthly_19972021
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs.scatter(np.arange(1998,2022), sam_summermeans19982021, marker='o', c='r', label = 'SST', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2022)[~np.isnan(sam_summermeans19982021)], sam_summermeans19982021[~np.isnan(sam_summermeans19982021)])
axs.plot(np.arange(1998,2022), np.arange(1998,2022)*slope+intercept, c='r', label = 'sepapr 81-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1998, 2006)[~np.isnan(sam_summermeans19982021[:8])], sam_summermeans19982021[:8][~np.isnan(sam_summermeans19982021[:8])])
axs.plot(np.arange(1998,2006), np.arange(1998,2006)*slope+intercept, c='b', label = 'sepapr 81-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sam_summermeans19982021[8:])], sam_summermeans19982021[8:][~np.isnan(sam_summermeans19982021[8:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'sepapr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SAM', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'sepapr', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1983, 2022)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'gerlache_sepapr_SAM_19822022.png'
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
graphs_dir = 'gerlache_sepapr_SAM_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()