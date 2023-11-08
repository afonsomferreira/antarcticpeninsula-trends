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
seaice_advanceday = ice_timing['Pori adv']
seaice_retreatday = ice_timing['Pori ret']
seaice_duration = ice_timing['Pori dur']
# Sea Ice Extent from PALMER LTER
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaiceextent')
seaice_extent_df = pd.read_csv('seaiceextent.csv', sep=';')
seaice_extent_years = seaice_extent_df['Year']
seaice_extent = seaice_extent_df['Ori_Ext']
#%% Calculate mean and stds for each month
#CHL
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
    yeartemp_jan = chl_bransfield[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = chl_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = chl_bransfield[(time_date_years == i) & (time_date_months == 4)]   
    if i == 1997:
        chl_sep19982021_mean = np.nanmean(yeartemp_sep)
        chl_sep19982021_std = np.nanstd(yeartemp_sep)
        chl_oct19982021_mean = np.nanmean(yeartemp_oct)
        chl_oct19982021_std = np.nanstd(yeartemp_oct)
        chl_nov19982021_mean = np.nanmean(yeartemp_nov)
        chl_nov19982021_std = np.nanstd(yeartemp_nov)
        chl_dec19982021_mean = np.nanmean(yeartemp_dec)
        chl_dec19982021_std = np.nanstd(yeartemp_dec)
        chl_jan19982021_mean = np.nanmean(yeartemp_jan)
        chl_jan19982021_std = np.nanstd(yeartemp_jan)
        chl_feb19982021_mean = np.nanmean(yeartemp_feb)
        chl_feb19982021_std = np.nanstd(yeartemp_feb)
        chl_mar19982021_mean = np.nanmean(yeartemp_mar)
        chl_mar19982021_std = np.nanstd(yeartemp_mar)
        chl_apr19982021_mean = np.nanmean(yeartemp_apr)
        chl_apr19982021_std = np.nanstd(yeartemp_apr)
    else:
        chl_sep19982021_mean = np.hstack((chl_sep19982021_mean, np.nanmean(yeartemp_sep)))
        chl_sep19982021_std = np.hstack((chl_sep19982021_std, np.nanstd(yeartemp_sep)))
        chl_oct19982021_mean = np.hstack((chl_oct19982021_mean, np.nanmean(yeartemp_oct)))
        chl_oct19982021_std = np.hstack((chl_oct19982021_std, np.nanstd(yeartemp_oct)))
        chl_nov19982021_mean = np.hstack((chl_nov19982021_mean, np.nanmean(yeartemp_nov)))
        chl_nov19982021_std = np.hstack((chl_nov19982021_std, np.nanstd(yeartemp_nov)))
        chl_dec19982021_mean = np.hstack((chl_dec19982021_mean, np.nanmean(yeartemp_dec)))
        chl_dec19982021_std = np.hstack((chl_dec19982021_std, np.nanstd(yeartemp_dec)))
        chl_jan19982021_mean = np.hstack((chl_jan19982021_mean, np.nanmean(yeartemp_jan)))
        chl_jan19982021_std = np.hstack((chl_jan19982021_std, np.nanstd(yeartemp_jan)))
        chl_feb19982021_mean = np.hstack((chl_feb19982021_mean, np.nanmean(yeartemp_feb)))
        chl_feb19982021_std = np.hstack((chl_feb19982021_std, np.nanstd(yeartemp_feb)))
        chl_mar19982021_mean = np.hstack((chl_mar19982021_mean, np.nanmean(yeartemp_mar)))
        chl_mar19982021_std = np.hstack((chl_mar19982021_std, np.nanstd(yeartemp_mar)))
        chl_apr19982021_mean = np.hstack((chl_apr19982021_mean, np.nanmean(yeartemp_apr)))
        chl_apr19982021_std = np.hstack((chl_apr19982021_std, np.nanstd(yeartemp_apr)))
#SST
sst_bransfield_19982021 = sst[clusters == 4,:]
sst_bransfield_19982021 = np.nanmean(sst_bransfield_19982021,0)
sst_bransfield_19811996 = sst_19811996[clusters == 4,:]
sst_bransfield_19811996 = np.nanmean(sst_bransfield_19811996,0)
sst_bransfield = np.hstack((sst_bransfield_19811996, sst_bransfield_19982021))
time_date_sst = np.hstack((time_date_19811996, time_date_sst))
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1981, 2022):
    yeartemp_sep = sst_bransfield[(time_date_years == i) & (time_date_months == 9)]
    yeartemp_oct = sst_bransfield[(time_date_years == i) & (time_date_months == 10)]
    yeartemp_nov = sst_bransfield[(time_date_years == i) & (time_date_months == 11)]
    yeartemp_dec = sst_bransfield[(time_date_years == i) & (time_date_months == 12)]
    yeartemp_jan = sst_bransfield[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = sst_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = sst_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = sst_bransfield[(time_date_years == i) & (time_date_months == 4)]   
    if i == 1981:
        sst_sep19982021_mean = np.nanmean(yeartemp_sep)
        sst_sep19982021_std = np.nanstd(yeartemp_sep)
        sst_oct19982021_mean = np.nanmean(yeartemp_oct)
        sst_oct19982021_std = np.nanstd(yeartemp_oct)
        sst_nov19982021_mean = np.nanmean(yeartemp_nov)
        sst_nov19982021_std = np.nanstd(yeartemp_nov)
        sst_dec19982021_mean = np.nanmean(yeartemp_dec)
        sst_dec19982021_std = np.nanstd(yeartemp_dec)
        sst_jan19982021_mean = np.nanmean(yeartemp_jan)
        sst_jan19982021_std = np.nanstd(yeartemp_jan)
        sst_feb19982021_mean = np.nanmean(yeartemp_feb)
        sst_feb19982021_std = np.nanstd(yeartemp_feb)
        sst_mar19982021_mean = np.nanmean(yeartemp_mar)
        sst_mar19982021_std = np.nanstd(yeartemp_mar)
        sst_apr19982021_mean = np.nanmean(yeartemp_apr)
        sst_apr19982021_std = np.nanstd(yeartemp_apr)
    else:
        sst_sep19982021_mean = np.hstack((sst_sep19982021_mean, np.nanmean(yeartemp_sep)))
        sst_sep19982021_std = np.hstack((sst_sep19982021_std, np.nanstd(yeartemp_sep)))
        sst_oct19982021_mean = np.hstack((sst_oct19982021_mean, np.nanmean(yeartemp_oct)))
        sst_oct19982021_std = np.hstack((sst_oct19982021_std, np.nanstd(yeartemp_oct)))
        sst_nov19982021_mean = np.hstack((sst_nov19982021_mean, np.nanmean(yeartemp_nov)))
        sst_nov19982021_std = np.hstack((sst_nov19982021_std, np.nanstd(yeartemp_nov)))
        sst_dec19982021_mean = np.hstack((sst_dec19982021_mean, np.nanmean(yeartemp_dec)))
        sst_dec19982021_std = np.hstack((sst_dec19982021_std, np.nanstd(yeartemp_dec)))
        sst_jan19982021_mean = np.hstack((sst_jan19982021_mean, np.nanmean(yeartemp_jan)))
        sst_jan19982021_std = np.hstack((sst_jan19982021_std, np.nanstd(yeartemp_jan)))
        sst_feb19982021_mean = np.hstack((sst_feb19982021_mean, np.nanmean(yeartemp_feb)))
        sst_feb19982021_std = np.hstack((sst_feb19982021_std, np.nanstd(yeartemp_feb)))
        sst_mar19982021_mean = np.hstack((sst_mar19982021_mean, np.nanmean(yeartemp_mar)))
        sst_mar19982021_std = np.hstack((sst_mar19982021_std, np.nanstd(yeartemp_mar)))
        sst_apr19982021_mean = np.hstack((sst_apr19982021_mean, np.nanmean(yeartemp_apr)))
        sst_apr19982021_std = np.hstack((sst_apr19982021_std, np.nanstd(yeartemp_apr)))
#PAR
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
    yeartemp_jan = par_bransfield[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = par_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = par_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = par_bransfield[(time_date_years == i) & (time_date_months == 4)]   
    if i == 1997:
        par_sep19982021_mean = np.nanmean(yeartemp_sep)
        par_sep19982021_std = np.nanstd(yeartemp_sep)
        par_oct19982021_mean = np.nanmean(yeartemp_oct)
        par_oct19982021_std = np.nanstd(yeartemp_oct)
        par_nov19982021_mean = np.nanmean(yeartemp_nov)
        par_nov19982021_std = np.nanstd(yeartemp_nov)
        par_dec19982021_mean = np.nanmean(yeartemp_dec)
        par_dec19982021_std = np.nanstd(yeartemp_dec)
        par_jan19982021_mean = np.nanmean(yeartemp_jan)
        par_jan19982021_std = np.nanstd(yeartemp_jan)
        par_feb19982021_mean = np.nanmean(yeartemp_feb)
        par_feb19982021_std = np.nanstd(yeartemp_feb)
        par_mar19982021_mean = np.nanmean(yeartemp_mar)
        par_mar19982021_std = np.nanstd(yeartemp_mar)
        par_apr19982021_mean = np.nanmean(yeartemp_apr)
        par_apr19982021_std = np.nanstd(yeartemp_apr)
    else:
        par_sep19982021_mean = np.hstack((par_sep19982021_mean, np.nanmean(yeartemp_sep)))
        par_sep19982021_std = np.hstack((par_sep19982021_std, np.nanstd(yeartemp_sep)))
        par_oct19982021_mean = np.hstack((par_oct19982021_mean, np.nanmean(yeartemp_oct)))
        par_oct19982021_std = np.hstack((par_oct19982021_std, np.nanstd(yeartemp_oct)))
        par_nov19982021_mean = np.hstack((par_nov19982021_mean, np.nanmean(yeartemp_nov)))
        par_nov19982021_std = np.hstack((par_nov19982021_std, np.nanstd(yeartemp_nov)))
        par_dec19982021_mean = np.hstack((par_dec19982021_mean, np.nanmean(yeartemp_dec)))
        par_dec19982021_std = np.hstack((par_dec19982021_std, np.nanstd(yeartemp_dec)))
        par_jan19982021_mean = np.hstack((par_jan19982021_mean, np.nanmean(yeartemp_jan)))
        par_jan19982021_std = np.hstack((par_jan19982021_std, np.nanstd(yeartemp_jan)))
        par_feb19982021_mean = np.hstack((par_feb19982021_mean, np.nanmean(yeartemp_feb)))
        par_feb19982021_std = np.hstack((par_feb19982021_std, np.nanstd(yeartemp_feb)))
        par_mar19982021_mean = np.hstack((par_mar19982021_mean, np.nanmean(yeartemp_mar)))
        par_mar19982021_std = np.hstack((par_mar19982021_std, np.nanstd(yeartemp_mar)))
        par_apr19982021_mean = np.hstack((par_apr19982021_mean, np.nanmean(yeartemp_apr)))
        par_apr19982021_std = np.hstack((par_apr19982021_std, np.nanstd(yeartemp_apr)))
# WINDS
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
for i in np.arange(1997, 2020):
    yeartemp_sep = wind_speed[(time_date_years == i) & (time_date_months == 9)]
    yeartemp_oct = wind_speed[(time_date_years == i) & (time_date_months == 10)]
    yeartemp_nov = wind_speed[(time_date_years == i) & (time_date_months == 11)]
    yeartemp_dec = wind_speed[(time_date_years == i) & (time_date_months == 12)]
    yeartemp_jan = wind_speed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wind_speed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wind_speed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wind_speed[(time_date_years == i) & (time_date_months == 4)]   
    if i == 1997:
        windspeed_sep19982021_mean = np.nanmean(yeartemp_sep)
        windspeed_sep19982021_std = np.nanstd(yeartemp_sep)
        windspeed_oct19982021_mean = np.nanmean(yeartemp_oct)
        windspeed_oct19982021_std = np.nanstd(yeartemp_oct)
        windspeed_nov19982021_mean = np.nanmean(yeartemp_nov)
        windspeed_nov19982021_std = np.nanstd(yeartemp_nov)
        windspeed_dec19982021_mean = np.nanmean(yeartemp_dec)
        windspeed_dec19982021_std = np.nanstd(yeartemp_dec)
        windspeed_jan19982021_mean = np.nanmean(yeartemp_jan)
        windspeed_jan19982021_std = np.nanstd(yeartemp_jan)
        windspeed_feb19982021_mean = np.nanmean(yeartemp_feb)
        windspeed_feb19982021_std = np.nanstd(yeartemp_feb)
        windspeed_mar19982021_mean = np.nanmean(yeartemp_mar)
        windspeed_mar19982021_std = np.nanstd(yeartemp_mar)
        windspeed_apr19982021_mean = np.nanmean(yeartemp_apr)
        windspeed_apr19982021_std = np.nanstd(yeartemp_apr)
    else:
        windspeed_sep19982021_mean = np.hstack((windspeed_sep19982021_mean, np.nanmean(yeartemp_sep)))
        windspeed_sep19982021_std = np.hstack((windspeed_sep19982021_std, np.nanstd(yeartemp_sep)))
        windspeed_oct19982021_mean = np.hstack((windspeed_oct19982021_mean, np.nanmean(yeartemp_oct)))
        windspeed_oct19982021_std = np.hstack((windspeed_oct19982021_std, np.nanstd(yeartemp_oct)))
        windspeed_nov19982021_mean = np.hstack((windspeed_nov19982021_mean, np.nanmean(yeartemp_nov)))
        windspeed_nov19982021_std = np.hstack((windspeed_nov19982021_std, np.nanstd(yeartemp_nov)))
        windspeed_dec19982021_mean = np.hstack((windspeed_dec19982021_mean, np.nanmean(yeartemp_dec)))
        windspeed_dec19982021_std = np.hstack((windspeed_dec19982021_std, np.nanstd(yeartemp_dec)))
        windspeed_jan19982021_mean = np.hstack((windspeed_jan19982021_mean, np.nanmean(yeartemp_jan)))
        windspeed_jan19982021_std = np.hstack((windspeed_jan19982021_std, np.nanstd(yeartemp_jan)))
        windspeed_feb19982021_mean = np.hstack((windspeed_feb19982021_mean, np.nanmean(yeartemp_feb)))
        windspeed_feb19982021_std = np.hstack((windspeed_feb19982021_std, np.nanstd(yeartemp_feb)))
        windspeed_mar19982021_mean = np.hstack((windspeed_mar19982021_mean, np.nanmean(yeartemp_mar)))
        windspeed_mar19982021_std = np.hstack((windspeed_mar19982021_std, np.nanstd(yeartemp_mar)))
        windspeed_apr19982021_mean = np.hstack((windspeed_apr19982021_mean, np.nanmean(yeartemp_apr)))
        windspeed_apr19982021_std = np.hstack((windspeed_apr19982021_std, np.nanstd(yeartemp_apr)))


#%% Plots Chl
# Sep
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), chl_sep19982021_mean, yerr=chl_sep19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='g')
#axs.scatter(np.arange(1997,2022), chl_sep19982021_mean, marker='o', c='g', label = 'Sep', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_sep19982021_mean)], chl_sep19982021_mean[~np.isnan(chl_sep19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Sep 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(chl_sep19982021_mean[:9])], chl_sep19982021_mean[:9][~np.isnan(chl_sep19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Sep 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_sep19982021_mean[9:])], chl_sep19982021_mean[9:][~np.isnan(chl_sep19982021_mean[9:])])
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
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Oct
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), chl_oct19982021_mean, yerr=chl_oct19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='g')
#axs.scatter(np.arange(1997,2022), chl_oct19982021_mean, marker='o', c='g', label = 'oct', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_oct19982021_mean)], chl_oct19982021_mean[~np.isnan(chl_oct19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Oct 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(chl_oct19982021_mean[:9])], chl_oct19982021_mean[:9][~np.isnan(chl_oct19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Oct 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_oct19982021_mean[9:])], chl_oct19982021_mean[9:][~np.isnan(chl_oct19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Oct 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_oct_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Nov
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), chl_nov19982021_mean, yerr=chl_nov19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='g')
#axs.scatter(np.arange(1997,2022), chl_nov19982021_mean, marker='o', c='g', label = 'nov', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_nov19982021_mean)], chl_nov19982021_mean[~np.isnan(chl_nov19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Nov 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(chl_nov19982021_mean[:9])], chl_nov19982021_mean[:9][~np.isnan(chl_nov19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Nov 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_nov19982021_mean[9:])], chl_nov19982021_mean[9:][~np.isnan(chl_nov19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Nov 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_nov_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Dec
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), chl_dec19982021_mean, yerr=chl_dec19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='g')
#axs.scatter(np.arange(1997,2022), chl_dec19982021_mean, marker='o', c='g', label = 'dec', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_dec19982021_mean)], chl_dec19982021_mean[~np.isnan(chl_dec19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Dec 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(chl_dec19982021_mean[:9])], chl_dec19982021_mean[:9][~np.isnan(chl_dec19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Dec 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_dec19982021_mean[9:])], chl_dec19982021_mean[9:][~np.isnan(chl_dec19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Dec 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_dec_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Jan
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), chl_jan19982021_mean, yerr=chl_jan19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='g')
#axs.scatter(np.arange(1997,2022), chl_jan19982021_mean, marker='o', c='g', label = 'jan', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_jan19982021_mean)], chl_jan19982021_mean[~np.isnan(chl_jan19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Jan 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(chl_jan19982021_mean[:9])], chl_jan19982021_mean[:9][~np.isnan(chl_jan19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Jan 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_jan19982021_mean[9:])], chl_jan19982021_mean[9:][~np.isnan(chl_jan19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Jan 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'jan', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_jan_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Feb
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), chl_feb19982021_mean, yerr=chl_feb19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='g')
#axs.scatter(np.arange(1997,2022), chl_feb19982021_mean, marker='o', c='g', label = 'feb', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_feb19982021_mean)], chl_feb19982021_mean[~np.isnan(chl_feb19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Feb 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(chl_feb19982021_mean[:9])], chl_feb19982021_mean[:9][~np.isnan(chl_feb19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Feb 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_feb19982021_mean[9:])], chl_feb19982021_mean[9:][~np.isnan(chl_feb19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Feb 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'feb', 'feb', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_feb_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Mar
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), chl_mar19982021_mean, yerr=chl_mar19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='g')
#axs.scatter(np.arange(1997,2022), chl_mar19982021_mean, marker='o', c='g', label = 'mar', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_mar19982021_mean)], chl_mar19982021_mean[~np.isnan(chl_mar19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Mar 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(chl_mar19982021_mean[:9])], chl_mar19982021_mean[:9][~np.isnan(chl_mar19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Mar 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_mar19982021_mean[9:])], chl_mar19982021_mean[9:][~np.isnan(chl_mar19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Mar 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'mar', 'mar', 'mar', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_mar_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Apr
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, aprker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), chl_apr19982021_mean, yerr=chl_apr19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='g')
#axs.scatter(np.arange(1997,2022), chl_mar19982021_mean, marker='o', c='g', label = 'mar', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(chl_apr19982021_mean)], chl_apr19982021_mean[~np.isnan(chl_apr19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Apr 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(chl_apr19982021_mean[:9])], chl_apr19982021_mean[:9][~np.isnan(chl_apr19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Apr 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(chl_apr19982021_mean[9:])], chl_apr19982021_mean[9:][~np.isnan(chl_apr19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Apr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('g')
axs.tick_params(axis='y', colors='g')
axs.spines['left'].set_color('g')
axs.set_ylabel('Chl-a (mg m$^{-3}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'mar', 'mar', 'mar', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_apr_chl_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Plots SST
# Sep
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1981, 2022), sst_sep19982021_mean, yerr=sst_sep19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='r')
#axs.scatter(np.arange(1981,2022), sst_sep19982021_mean, marker='o', c='r', label = 'Sep', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2022)[~np.isnan(sst_sep19982021_mean)], sst_sep19982021_mean[~np.isnan(sst_sep19982021_mean)])
axs.plot(np.arange(1981,2022), np.arange(1981,2022)*slope+intercept, c='k', label = 'Sep 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2006)[~np.isnan(sst_sep19982021_mean[:25])], sst_sep19982021_mean[:25][~np.isnan(sst_sep19982021_mean[:25])])
axs.plot(np.arange(1981,2006), np.arange(1981,2006)*slope+intercept, c='b', label = 'Sep 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_sep19982021_mean[25:])], sst_sep19982021_mean[25:][~np.isnan(sst_sep19982021_mean[25:])])
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
plt.xlim(1981.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Oct
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1981, 2022), sst_oct19982021_mean, yerr=sst_oct19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='r')
#axs.scatter(np.arange(1981,2022), sst_oct19982021_mean, marker='o', c='r', label = 'oct', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2022)[~np.isnan(sst_oct19982021_mean)], sst_oct19982021_mean[~np.isnan(sst_oct19982021_mean)])
axs.plot(np.arange(1981,2022), np.arange(1981,2022)*slope+intercept, c='k', label = 'Oct 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2006)[~np.isnan(sst_oct19982021_mean[:25])], sst_oct19982021_mean[:25][~np.isnan(sst_oct19982021_mean[:25])])
axs.plot(np.arange(1981,2006), np.arange(1981,2006)*slope+intercept, c='b', label = 'Oct 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_oct19982021_mean[25:])], sst_oct19982021_mean[25:][~np.isnan(sst_oct19982021_mean[25:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Oct 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_oct_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Nov
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1981, 2022), sst_nov19982021_mean, yerr=sst_nov19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='r')
#axs.scatter(np.arange(1981,2022), sst_nov19982021_mean, marker='o', c='r', label = 'nov', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2022)[~np.isnan(sst_nov19982021_mean)], sst_nov19982021_mean[~np.isnan(sst_nov19982021_mean)])
axs.plot(np.arange(1981,2022), np.arange(1981,2022)*slope+intercept, c='k', label = 'Nov 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2006)[~np.isnan(sst_nov19982021_mean[:25])], sst_nov19982021_mean[:25][~np.isnan(sst_nov19982021_mean[:25])])
axs.plot(np.arange(1981,2006), np.arange(1981,2006)*slope+intercept, c='b', label = 'Nov 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_nov19982021_mean[25:])], sst_nov19982021_mean[25:][~np.isnan(sst_nov19982021_mean[25:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Nov 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1981.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_nov_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Dec
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1981, 2022), sst_dec19982021_mean, yerr=sst_dec19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='r')
#axs.scatter(np.arange(1981,2022), sst_dec19982021_mean, marker='o', c='r', label = 'dec', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2022)[~np.isnan(sst_dec19982021_mean)], sst_dec19982021_mean[~np.isnan(sst_dec19982021_mean)])
axs.plot(np.arange(1981,2022), np.arange(1981,2022)*slope+intercept, c='k', label = 'Dec 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2006)[~np.isnan(sst_dec19982021_mean[:25])], sst_dec19982021_mean[:25][~np.isnan(sst_dec19982021_mean[:25])])
axs.plot(np.arange(1981,2006), np.arange(1981,2006)*slope+intercept, c='b', label = 'Dec 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_dec19982021_mean[25:])], sst_dec19982021_mean[25:][~np.isnan(sst_dec19982021_mean[25:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Dec 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1981.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_dec_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Jan
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1981, 2022), sst_jan19982021_mean, yerr=sst_jan19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='r')
#axs.scatter(np.arange(1981,2022), sst_jan19982021_mean, marker='o', c='r', label = 'jan', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2022)[~np.isnan(sst_jan19982021_mean)], sst_jan19982021_mean[~np.isnan(sst_jan19982021_mean)])
axs.plot(np.arange(1981,2022), np.arange(1981,2022)*slope+intercept, c='k', label = 'Jan 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2006)[~np.isnan(sst_jan19982021_mean[:25])], sst_jan19982021_mean[:25][~np.isnan(sst_jan19982021_mean[:25])])
axs.plot(np.arange(1981,2006), np.arange(1981,2006)*slope+intercept, c='b', label = 'Jan 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_jan19982021_mean[25:])], sst_jan19982021_mean[25:][~np.isnan(sst_jan19982021_mean[25:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Jan 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'jan', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1981.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_jan_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Feb
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1981, 2022), sst_feb19982021_mean, yerr=sst_feb19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='r')
#axs.scatter(np.arange(1981,2022), sst_feb19982021_mean, marker='o', c='r', label = 'feb', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2022)[~np.isnan(sst_feb19982021_mean)], sst_feb19982021_mean[~np.isnan(sst_feb19982021_mean)])
axs.plot(np.arange(1981,2022), np.arange(1981,2022)*slope+intercept, c='k', label = 'Feb 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2006)[~np.isnan(sst_feb19982021_mean[:25])], sst_feb19982021_mean[:25][~np.isnan(sst_feb19982021_mean[:25])])
axs.plot(np.arange(1981,2006), np.arange(1981,2006)*slope+intercept, c='b', label = 'Feb 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_feb19982021_mean[25:])], sst_feb19982021_mean[25:][~np.isnan(sst_feb19982021_mean[25:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Feb 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'feb', 'feb', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1981.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_feb_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Mar
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1981, 2022), sst_mar19982021_mean, yerr=sst_mar19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='r')
#axs.scatter(np.arange(1981,2022), sst_mar19982021_mean, marker='o', c='r', label = 'mar', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2022)[~np.isnan(sst_mar19982021_mean)], sst_mar19982021_mean[~np.isnan(sst_mar19982021_mean)])
axs.plot(np.arange(1981,2022), np.arange(1981,2022)*slope+intercept, c='k', label = 'Mar 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2006)[~np.isnan(sst_mar19982021_mean[:25])], sst_mar19982021_mean[:25][~np.isnan(sst_mar19982021_mean[:25])])
axs.plot(np.arange(1981,2006), np.arange(1981,2006)*slope+intercept, c='b', label = 'Mar 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_mar19982021_mean[25:])], sst_mar19982021_mean[25:][~np.isnan(sst_mar19982021_mean[25:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Mar 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'mar', 'mar', 'mar', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1981.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_mar_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Apr
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, aprker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1981, 2022), sst_apr19982021_mean, yerr=sst_apr19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='r')
#axs.scatter(np.arange(1981,2022), sst_mar19982021_mean, marker='o', c='r', label = 'mar', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2022)[~np.isnan(sst_apr19982021_mean)], sst_apr19982021_mean[~np.isnan(sst_apr19982021_mean)])
axs.plot(np.arange(1981,2022), np.arange(1981,2022)*slope+intercept, c='k', label = 'Apr 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1981, 2006)[~np.isnan(sst_apr19982021_mean[:25])], sst_apr19982021_mean[:25][~np.isnan(sst_apr19982021_mean[:25])])
axs.plot(np.arange(1981,2006), np.arange(1981,2006)*slope+intercept, c='b', label = 'Apr 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(sst_apr19982021_mean[25:])], sst_apr19982021_mean[25:][~np.isnan(sst_apr19982021_mean[25:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Apr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('r')
axs.tick_params(axis='y', colors='r')
axs.spines['left'].set_color('r')
axs.set_ylabel('SST (°C)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'mar', 'mar', 'mar', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1981.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_apr_sst_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Plots par
# Sep
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), par_sep19982021_mean, yerr=par_sep19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='y')
#axs.scatter(np.arange(1997,2022), par_sep19982021_mean, marker='o', c='g', label = 'Sep', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(par_sep19982021_mean)], par_sep19982021_mean[~np.isnan(par_sep19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Sep 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(par_sep19982021_mean[:9])], par_sep19982021_mean[:9][~np.isnan(par_sep19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Sep 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_sep19982021_mean[9:])], par_sep19982021_mean[9:][~np.isnan(par_sep19982021_mean[9:])])
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
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Oct
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), par_oct19982021_mean, yerr=par_oct19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='y')
#axs.scatter(np.arange(1997,2022), par_oct19982021_mean, marker='o', c='y', label = 'oct', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(par_oct19982021_mean)], par_oct19982021_mean[~np.isnan(par_oct19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Oct 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(par_oct19982021_mean[:9])], par_oct19982021_mean[:9][~np.isnan(par_oct19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Oct 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_oct19982021_mean[9:])], par_oct19982021_mean[9:][~np.isnan(par_oct19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Oct 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_oct_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Nov
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), par_nov19982021_mean, yerr=par_nov19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='y')
#axs.scatter(np.arange(1997,2022), par_nov19982021_mean, marker='o', c='y', label = 'nov', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(par_nov19982021_mean)], par_nov19982021_mean[~np.isnan(par_nov19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Nov 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(par_nov19982021_mean[:9])], par_nov19982021_mean[:9][~np.isnan(par_nov19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Nov 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_nov19982021_mean[9:])], par_nov19982021_mean[9:][~np.isnan(par_nov19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Nov 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_nov_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Dec
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), par_dec19982021_mean, yerr=par_dec19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='y')
#axs.scatter(np.arange(1997,2022), par_dec19982021_mean, marker='o', c='y', label = 'dec', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(par_dec19982021_mean)], par_dec19982021_mean[~np.isnan(par_dec19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Dec 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(par_dec19982021_mean[:9])], par_dec19982021_mean[:9][~np.isnan(par_dec19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Dec 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_dec19982021_mean[9:])], par_dec19982021_mean[9:][~np.isnan(par_dec19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Dec 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_dec_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Jan
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), par_jan19982021_mean, yerr=par_jan19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='y')
#axs.scatter(np.arange(1997,2022), par_jan19982021_mean, marker='o', c='y', label = 'jan', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(par_jan19982021_mean)], par_jan19982021_mean[~np.isnan(par_jan19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Jan 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(par_jan19982021_mean[:9])], par_jan19982021_mean[:9][~np.isnan(par_jan19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Jan 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_jan19982021_mean[9:])], par_jan19982021_mean[9:][~np.isnan(par_jan19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Jan 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'jan', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_jan_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Feb
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), par_feb19982021_mean, yerr=par_feb19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='y')
#axs.scatter(np.arange(1997,2022), par_feb19982021_mean, marker='o', c='y', label = 'feb', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(par_feb19982021_mean)], par_feb19982021_mean[~np.isnan(par_feb19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Feb 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(par_feb19982021_mean[:9])], par_feb19982021_mean[:9][~np.isnan(par_feb19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Feb 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_feb19982021_mean[9:])], par_feb19982021_mean[9:][~np.isnan(par_feb19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Feb 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'feb', 'feb', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_feb_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Mar
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), par_mar19982021_mean, yerr=par_mar19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='y')
#axs.scatter(np.arange(1997,2022), par_mar19982021_mean, marker='o', c='y', label = 'mar', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(par_mar19982021_mean)], par_mar19982021_mean[~np.isnan(par_mar19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Mar 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(par_mar19982021_mean[:9])], par_mar19982021_mean[:9][~np.isnan(par_mar19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Mar 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_mar19982021_mean[9:])], par_mar19982021_mean[9:][~np.isnan(par_mar19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Mar 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'mar', 'mar', 'mar', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_mar_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Apr
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, aprker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2022), par_apr19982021_mean, yerr=par_apr19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='y')
#axs.scatter(np.arange(1997,2022), par_mar19982021_mean, marker='o', c='y', label = 'mar', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2022)[~np.isnan(par_apr19982021_mean)], par_apr19982021_mean[~np.isnan(par_apr19982021_mean)])
axs.plot(np.arange(1997,2022), np.arange(1997,2022)*slope+intercept, c='k', label = 'Apr 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(par_apr19982021_mean[:9])], par_apr19982021_mean[:9][~np.isnan(par_apr19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Apr 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2022)[~np.isnan(par_apr19982021_mean[9:])], par_apr19982021_mean[9:][~np.isnan(par_apr19982021_mean[9:])])
axs.plot(np.arange(2006,2022), np.arange(2006,2022)*slope+intercept, c='r', label = 'Apr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('y')
axs.tick_params(axis='y', colors='y')
axs.spines['left'].set_color('y')
axs.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'mar', 'mar', 'mar', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_apr_par_19822022.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Plots Wind Speed
# Sep
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2020), windspeed_sep19982021_mean, yerr=windspeed_sep19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='purple')
#axs.scatter(np.arange(1997,2020), windspeed_sep19982021_mean, marker='o', c='g', label = 'Sep', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2020)[~np.isnan(windspeed_sep19982021_mean)], windspeed_sep19982021_mean[~np.isnan(windspeed_sep19982021_mean)])
axs.plot(np.arange(1997,2020), np.arange(1997,2020)*slope+intercept, c='k', label = 'Sep 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(windspeed_sep19982021_mean[:9])], windspeed_sep19982021_mean[:9][~np.isnan(windspeed_sep19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Sep 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_sep19982021_mean[9:])], windspeed_sep19982021_mean[9:][~np.isnan(windspeed_sep19982021_mean[9:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'Sep 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (ms$^{-1}$', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_sep_windspeed_19822020.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Oct
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2020), windspeed_oct19982021_mean, yerr=windspeed_oct19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='purple')
#axs.scatter(np.arange(1997,2020), windspeed_oct19982021_mean, marker='o', c='purple', label = 'oct', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2020)[~np.isnan(windspeed_oct19982021_mean)], windspeed_oct19982021_mean[~np.isnan(windspeed_oct19982021_mean)])
axs.plot(np.arange(1997,2020), np.arange(1997,2020)*slope+intercept, c='k', label = 'Oct 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(windspeed_oct19982021_mean[:9])], windspeed_oct19982021_mean[:9][~np.isnan(windspeed_oct19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Oct 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_oct19982021_mean[9:])], windspeed_oct19982021_mean[9:][~np.isnan(windspeed_oct19982021_mean[9:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'Oct 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (ms$^{-1}$', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_oct_windspeed_19822020.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Nov
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2020), windspeed_nov19982021_mean, yerr=windspeed_nov19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='purple')
#axs.scatter(np.arange(1997,2020), windspeed_nov19982021_mean, marker='o', c='purple', label = 'nov', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2020)[~np.isnan(windspeed_nov19982021_mean)], windspeed_nov19982021_mean[~np.isnan(windspeed_nov19982021_mean)])
axs.plot(np.arange(1997,2020), np.arange(1997,2020)*slope+intercept, c='k', label = 'Nov 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(windspeed_nov19982021_mean[:9])], windspeed_nov19982021_mean[:9][~np.isnan(windspeed_nov19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Nov 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_nov19982021_mean[9:])], windspeed_nov19982021_mean[9:][~np.isnan(windspeed_nov19982021_mean[9:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'Nov 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (ms$^{-1}$', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_nov_windspeed_19822020.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Dec
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2020), windspeed_dec19982021_mean, yerr=windspeed_dec19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='purple')
#axs.scatter(np.arange(1997,2020), windspeed_dec19982021_mean, marker='o', c='purple', label = 'dec', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2020)[~np.isnan(windspeed_dec19982021_mean)], windspeed_dec19982021_mean[~np.isnan(windspeed_dec19982021_mean)])
axs.plot(np.arange(1997,2020), np.arange(1997,2020)*slope+intercept, c='k', label = 'Dec 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(windspeed_dec19982021_mean[:9])], windspeed_dec19982021_mean[:9][~np.isnan(windspeed_dec19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Dec 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_dec19982021_mean[9:])], windspeed_dec19982021_mean[9:][~np.isnan(windspeed_dec19982021_mean[9:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'Dec 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (ms$^{-1}$', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_dec_windspeed_19822020.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Jan
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2020), windspeed_jan19982021_mean, yerr=windspeed_jan19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='purple')
#axs.scatter(np.arange(1997,2020), windspeed_jan19982021_mean, marker='o', c='purple', label = 'jan', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2020)[~np.isnan(windspeed_jan19982021_mean)], windspeed_jan19982021_mean[~np.isnan(windspeed_jan19982021_mean)])
axs.plot(np.arange(1997,2020), np.arange(1997,2020)*slope+intercept, c='k', label = 'Jan 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(windspeed_jan19982021_mean[:9])], windspeed_jan19982021_mean[:9][~np.isnan(windspeed_jan19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Jan 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_jan19982021_mean[9:])], windspeed_jan19982021_mean[9:][~np.isnan(windspeed_jan19982021_mean[9:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'Jan 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (ms$^{-1}$', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'jan', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_jan_windspeed_19822020.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Feb
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2020), windspeed_feb19982021_mean, yerr=windspeed_feb19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='purple')
#axs.scatter(np.arange(1997,2020), windspeed_feb19982021_mean, marker='o', c='purple', label = 'feb', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2020)[~np.isnan(windspeed_feb19982021_mean)], windspeed_feb19982021_mean[~np.isnan(windspeed_feb19982021_mean)])
axs.plot(np.arange(1997,2020), np.arange(1997,2020)*slope+intercept, c='k', label = 'Feb 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(windspeed_feb19982021_mean[:9])], windspeed_feb19982021_mean[:9][~np.isnan(windspeed_feb19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Feb 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_feb19982021_mean[9:])], windspeed_feb19982021_mean[9:][~np.isnan(windspeed_feb19982021_mean[9:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'Feb 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (ms$^{-1}$', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'feb', 'feb', 'Feb', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_feb_windspeed_19822020.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Mar
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2020), windspeed_mar19982021_mean, yerr=windspeed_mar19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='purple')
#axs.scatter(np.arange(1997,2020), windspeed_mar19982021_mean, marker='o', c='purple', label = 'mar', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2020)[~np.isnan(windspeed_mar19982021_mean)], windspeed_mar19982021_mean[~np.isnan(windspeed_mar19982021_mean)])
axs.plot(np.arange(1997,2020), np.arange(1997,2020)*slope+intercept, c='k', label = 'Mar 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(windspeed_mar19982021_mean[:9])], windspeed_mar19982021_mean[:9][~np.isnan(windspeed_mar19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Mar 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_mar19982021_mean[9:])], windspeed_mar19982021_mean[9:][~np.isnan(windspeed_mar19982021_mean[9:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'Mar 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (ms$^{-1}$', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'mar', 'mar', 'mar', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_mar_windspeed_19822020.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
# Apr
fig, axs = plt.subplots(1, 1, figsize=(11,2))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, aprker='^', c='#000080', label = 'Observed recruitment')
(_, caps, _) = axs.errorbar(np.arange(1997, 2020), windspeed_apr19982021_mean, yerr=windspeed_apr19982021_std, capsize=3, elinewidth=1, linestyle='None', marker='o', c='purple')
#axs.scatter(np.arange(1997,2020), windspeed_mar19982021_mean, marker='o', c='purple', label = 'mar', s=50, alpha=.5, zorder=10)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2020)[~np.isnan(windspeed_apr19982021_mean)], windspeed_apr19982021_mean[~np.isnan(windspeed_apr19982021_mean)])
axs.plot(np.arange(1997,2020), np.arange(1997,2020)*slope+intercept, c='k', label = 'Apr 97-21', linestyle='--')
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(1997, 2006)[~np.isnan(windspeed_apr19982021_mean[:9])], windspeed_apr19982021_mean[:9][~np.isnan(windspeed_apr19982021_mean[:9])])
axs.plot(np.arange(1997,2006), np.arange(1997,2006)*slope+intercept, c='b', label = 'Apr 97-05', linestyle='--', marker='s', alpha=.5)
[slope,intercept,r_val,p_val,_] = stats.linregress(np.arange(2006, 2020)[~np.isnan(windspeed_apr19982021_mean[9:])], windspeed_apr19982021_mean[9:][~np.isnan(windspeed_apr19982021_mean[9:])])
axs.plot(np.arange(2006,2020), np.arange(2006,2020)*slope+intercept, c='r', label = 'Apr 05-22', linestyle='--', marker='*', alpha=.5)
axs.yaxis.label.set_color('purple')
axs.tick_params(axis='y', colors='purple')
axs.spines['left'].set_color('purple')
axs.set_ylabel('Wind Speed (ms$^{-1}$', fontsize=12)
#axs.set_xticks(ticks = [1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38, 42],
#           labels = ['Aug', 'oct', 'Oct', 'Nov', 'mar', 'mar', 'mar', 'Mar', 'Apr', 'May',
#                     'Jun', 'Jul'], fontsize=10)
#plt.axvline(39, linestyle='--', c='grey', alpha=0.3)
#plt.axvline(10, linestyle='--', c='grey', alpha=0.3)
plt.xlim(1996.5, 2021.5)
#fig.legend(loc="upper center", bbox_to_anchor=(.5,1.125), bbox_transform=axs.transAxes, ncol=5)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\')
graphs_dir = 'bransfield_apr_windspeed_19822020.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()



