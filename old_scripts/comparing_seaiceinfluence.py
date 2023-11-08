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
fh = np.load('antarcticpeninsula_newclusters.npz',allow_pickle = True)
clusters = fh['clusters']
# CHL
fh = np.load('chloc4so_19972021_10km.npz', allow_pickle=True)
lat_chl = fh['lat'][100:]
lon_chl = fh['lon'][30:250]
chl = fh['chl'][100:, 30:250, :]
time_date_chl = fh['time_date']
# Correct values
chl[chl > 50] = 50
# SEA ICE
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia')
### Load data 1998-2020
fh = np.load('seaice_19972021_10km.npz', allow_pickle=True)
lat_seaice = fh['lat'][100:]
lon_seaice = fh['lon'][30:250]
seaice = fh['seaice'][100:, 30:250, :]
seaice = seaice*100
time_date_seaice = fh['time_date']
time_date_seaice = time_date_seaice[246:]
seaice = seaice[:,:, 246:]
#%%
# Pass chl and seaice to same dates
chl_new = np.empty_like(seaice)
for i in range(0,len(lat_chl)):
    print(i)
    for j in range(0,len(lon_chl)):
        chl_pixel_temp = chl[i,j,:]
        chl_pixel_df = pd.DataFrame(data=chl_pixel_temp, index=time_date_chl)
        chl_pixel_df_daily = np.squeeze(chl_pixel_df.resample('D').mean().values)
        chl_new[i,j,:] = chl_pixel_df_daily

#%%
# Keep only pixels with seaice between 100 and 90
chl_90100seaice_valid = np.count_nonzero(~np.isnan(chl_new[seaice > 90]))
chl_90100seaice_mean = np.nanmean(chl_new[seaice > 90])
chl_90100seaice_std = np.nanstd(chl_new[seaice > 90])
# Keep only pixels with seaice between 90 and 80
chl_8090seaice_valid = np.count_nonzero(~np.isnan(chl_new[(seaice > 80) & (seaice <= 90)]))
chl_8090seaice_mean = np.nanmean(chl_new[(seaice > 80) & (seaice <= 90)])
chl_8090seaice_std = np.nanstd(chl_new[(seaice > 80) & (seaice <= 90)])
# Keep only pixels with seaice between 80 and 70
chl_7080seaice_valid = np.count_nonzero(~np.isnan(chl_new[(seaice > 70) & (seaice <= 80)]))
chl_7080seaice_mean = np.nanmean(chl_new[(seaice > 70) & (seaice <= 80)])
chl_7080seaice_std = np.nanstd(chl_new[(seaice > 70) & (seaice <= 80)])
# Keep only pixels with seaice between 70 and 60
chl_6070seaice_valid = np.count_nonzero(~np.isnan(chl_new[(seaice > 60) & (seaice <= 70)]))
chl_6070seaice_mean = np.nanmean(chl_new[(seaice > 60) & (seaice <= 70)])
chl_6070seaice_std = np.nanstd(chl_new[(seaice > 60) & (seaice <= 70)])
# Keep only pixels with seaice between 60 and 50
chl_5060seaice_valid = np.count_nonzero(~np.isnan(chl_new[(seaice > 50) & (seaice <= 60)]))
chl_5060seaice_mean = np.nanmean(chl_new[(seaice > 50) & (seaice <= 60)])
chl_5060seaice_std = np.nanstd(chl_new[(seaice > 50) & (seaice <= 60)])
# Keep only pixels with seaice between 50 and 40
chl_4050seaice_valid = np.count_nonzero(~np.isnan(chl_new[(seaice > 40) & (seaice <= 50)]))
chl_4050seaice_mean = np.nanmean(chl_new[(seaice > 40) & (seaice <= 50)])
chl_4050seaice_std = np.nanstd(chl_new[(seaice > 40) & (seaice <= 50)])
# Keep only pixels with seaice between 40 and 30
chl_3040seaice_valid = np.count_nonzero(~np.isnan(chl_new[(seaice > 30) & (seaice <= 40)]))
chl_3040seaice_mean = np.nanmean(chl_new[(seaice > 30) & (seaice <= 40)])
chl_3040seaice_std = np.nanstd(chl_new[(seaice > 30) & (seaice <= 40)])
# Keep only pixels with seaice between 30 and 20
chl_2030seaice_valid = np.count_nonzero(~np.isnan(chl_new[(seaice > 20) & (seaice <= 30)]))
chl_2030seaice_mean = np.nanmean(chl_new[(seaice > 20) & (seaice <= 30)])
chl_2030seaice_std = np.nanstd(chl_new[(seaice > 20) & (seaice <= 30)])
# Keep only pixels with seaice between 20 and 10
chl_1020seaice_valid = np.count_nonzero(~np.isnan(chl_new[(seaice > 10) & (seaice <= 20)]))
chl_1020seaice_mean = np.nanmean(chl_new[(seaice > 10) & (seaice <= 20)])
chl_1020seaice_std = np.nanstd(chl_new[(seaice > 10) & (seaice <= 20)])
# Keep only pixels with seaice between 10 and 0
chl_010seaice_valid = np.count_nonzero(~np.isnan(chl_new[(seaice > 0) & (seaice <= 10)]))
chl_010seaice_mean = np.nanmean(chl_new[(seaice > 0) & (seaice <= 10)])
chl_010seaice_std = np.nanstd(chl_new[(seaice > 0) & (seaice <= 10)])
#%% Plot
plt.bar(np.arange(1,11), np.hstack((chl_90100seaice_valid, chl_8090seaice_valid, chl_7080seaice_valid, chl_6070seaice_valid,
                          chl_5060seaice_valid, chl_4050seaice_valid, chl_3040seaice_valid, chl_2030seaice_valid,
                          chl_1020seaice_valid, chl_010seaice_valid)))
plt.xticks(ticks=np.arange(1,11), labels=['100-90', '90-80', '80-70', '70-60', '60-50', '50-40', '40-30',
                                   '30-20', '20-10', '10-0'])
#%%
(_, caps, _) = plt.errorbar(np.arange(1,11), np.hstack((chl_90100seaice_mean, chl_8090seaice_mean, chl_7080seaice_mean, chl_6070seaice_mean,
                          chl_5060seaice_mean, chl_4050seaice_mean, chl_3040seaice_mean, chl_2030seaice_mean,
                          chl_1020seaice_mean, chl_010seaice_mean)),
                            yerr=np.hstack((chl_90100seaice_std, chl_8090seaice_std, chl_7080seaice_std, chl_6070seaice_std,
                                                      chl_5060seaice_std, chl_4050seaice_std, chl_3040seaice_std, chl_2030seaice_std,
                                                      chl_1020seaice_std, chl_010seaice_std)), capsize=3, elinewidth=1, linestyle='None', marker='o', c='k')
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1997, 2019)[~np.isnan(chlinsitu_janfeb19812021)], chlinsitu_janfeb19812021[~np.isnan(chlinsitu_janfeb19812021)])
#plt.plot(np.arange(1997,2019), np.arange(1997,2019)*slope+intercept, linestyle='--', c='b', alpha=0.5)
#plt.text(1998, 10, f"R={r:.2f}", fontsize=14)
#plt.text(1998, 7.5, f"p-val={pval:.2f}", fontsize=14)
plt.xticks(ticks=np.arange(1,11), labels=['100-90', '90-80', '80-70', '70-60', '60-50', '50-40', '40-30',
                                   '30-20', '20-10', '10-0'])
plt.ylabel('Chla', fontsize=14)
plt.tight_layout()
graphs_dir = 'seaice_meanstd.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%%
(_, caps, _) = plt.errorbar(np.arange(1,11), np.hstack((chl_90100seaice_mean, chl_8090seaice_mean, chl_7080seaice_mean, chl_6070seaice_mean,
                          chl_5060seaice_mean, chl_4050seaice_mean, chl_3040seaice_mean, chl_2030seaice_mean,
                          chl_1020seaice_mean, chl_010seaice_mean)),
                            yerr=np.hstack((chl_90100seaice_std/chl_90100seaice_mean, chl_8090seaice_std/chl_8090seaice_mean, chl_7080seaice_std/chl_7080seaice_mean, chl_6070seaice_std/chl_6070seaice_mean,
                                                      chl_5060seaice_std/chl_5060seaice_mean, chl_4050seaice_std/chl_4050seaice_mean, chl_3040seaice_std/chl_3040seaice_mean, chl_2030seaice_std/chl_2030seaice_mean,
                                                      chl_1020seaice_std/chl_1020seaice_mean, chl_010seaice_std/chl_010seaice_mean)), capsize=3, elinewidth=1, linestyle='None', marker='o', c='k')
#[slope,intercept,r,pval,_] = stats.linregress(np.arange(1997, 2019)[~np.isnan(chlinsitu_janfeb19812021)], chlinsitu_janfeb19812021[~np.isnan(chlinsitu_janfeb19812021)])
#plt.plot(np.arange(1997,2019), np.arange(1997,2019)*slope+intercept, linestyle='--', c='b', alpha=0.5)
#plt.text(1998, 10, f"R={r:.2f}", fontsize=14)
#plt.text(1998, 7.5, f"p-val={pval:.2f}", fontsize=14)
plt.xticks(ticks=np.arange(1,11), labels=['100-90', '90-80', '80-70', '70-60', '60-50', '50-40', '40-30',
                                   '30-20', '20-10', '10-0'])
plt.ylabel('Chla', fontsize=14)
plt.tight_layout()
graphs_dir = 'seaice_meanstd.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()