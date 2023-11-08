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
#%% Load SAM
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sam\\')
sam_pd = pd.read_csv('norm.daily.aao.cdas.z700.19790101_current.csv', sep=',')
sam_daily = sam_pd['aao_index_cdas'].values
time_date_years = sam_pd['year'].values
time_date_months = sam_pd['month'].values
time_date_days = sam_pd['day'].values
#%% Average monthly
time_date_sam_daily = np.empty_like(time_date_days, dtype=object)
for i in range(0, len(time_date_sam_daily)):
    time_date_sam_daily[i] = datetime.datetime(year = time_date_years[i],
                                               month = time_date_months[i],
                                               day = time_date_days[i])
sam_pd_daily = pd.Series(data=sam_daily, index=time_date_sam_daily)
# resample monthly
sam_pd_monthly = sam_pd_daily.resample('M').mean()
sam_pd_yearly = sam_pd_daily.resample('Y').mean()
sam_pd_yearly = sam_pd_yearly
# calculate 1 year moving mean
sam_monthly_movingmean = sam_pd_monthly.rolling(12).mean().values
monthly_dates = sam_pd_monthly.index
sam_pd_monthly_p10 = sam_pd_daily.resample('M').quantile(.1)
sam_pd_monthly_p90 = sam_pd_daily.resample('M').quantile(.9)
sam_pd_monthly_p10_movingmean = sam_pd_monthly_p10.rolling(12).mean().values
sam_pd_monthly_p90_movingmean = sam_pd_monthly_p90.rolling(12).mean().values
#%%  Calculate SAM per season
# Winter SAM
for i in np.arange(1998, 2022):
    yeartemp_jun = sam_daily[(time_date_years == i) & (time_date_months == 6)]
    yeartemp_jul = sam_daily[(time_date_years == i) & (time_date_months == 7)]
    yeartemp_aug = sam_daily[(time_date_years == i) & (time_date_months == 8)]
    yeartemp_JJA = np.hstack((yeartemp_jun, yeartemp_jul, yeartemp_aug))
    if i == 1998:
        sam_JJA = np.nanmean(yeartemp_JJA)
    else:
        sam_JJA = np.hstack((sam_JJA, np.nanmean(yeartemp_JJA)))
# Spring SAM
for i in np.arange(1998, 2022):
    yeartemp_sep = sam_daily[(time_date_years == i) & (time_date_months == 9)]
    yeartemp_oct = sam_daily[(time_date_years == i) & (time_date_months == 10)]
    yeartemp_nov = sam_daily[(time_date_years == i) & (time_date_months == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        sam_SON = np.nanmean(yeartemp_SON)
    else:
        sam_SON = np.hstack((sam_SON, np.nanmean(yeartemp_SON)))
# Summer SAM
for i in np.arange(1998, 2022):
    yeartemp_dec = sam_daily[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = sam_daily[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = sam_daily[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        sam_DJF = np.nanmean(yeartemp_DJF)
    else:
        sam_DJF = np.hstack((sam_DJF, np.nanmean(yeartemp_DJF)))
# Autumn SAM
for i in np.arange(1998, 2022):
    yeartemp_mar = sam_daily[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = sam_daily[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_may = sam_daily[(time_date_years == i) & (time_date_months == 5)]
    yeartemp_MAM = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        sam_MAM = np.nanmean(yeartemp_MAM)
    else:
        sam_MAM = np.hstack((sam_MAM, np.nanmean(yeartemp_MAM)))     
# Sep-Apr SAM
for i in np.arange(1998, 2022):
    yeartemp_sep = sam_daily[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = sam_daily[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = sam_daily[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = sam_daily[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = sam_daily[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = sam_daily[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = sam_daily[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = sam_daily[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov,
                                 yeartemp_dec, yeartemp_jan, yeartemp_feb,
                                 yeartemp_mar, yeartemp_apr))
    if i == 1998:
        sam_sepapr = np.nanmean(yeartemp_sepapr)
    else:
        sam_sepapr = np.hstack((sam_sepapr, np.nanmean(yeartemp_sepapr)))  
sam_yearly = sam_pd_yearly.values[1:-1]
#%% Load El Niño
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\elnino\\')
elnino_pd = pd.read_csv('meiv2.csv', sep=';', header=None)
for i in range(0, len(elnino_pd)):
    temp_elnino = np.hstack((elnino_pd[1][i], elnino_pd[2][i], elnino_pd[3][i], elnino_pd[4][i],
                             elnino_pd[5][i], elnino_pd[6][i], elnino_pd[7][i], elnino_pd[8][i],
                             elnino_pd[9][i], elnino_pd[10][i], elnino_pd[11][i], elnino_pd[12][i]))
    temp_elnino = temp_elnino.astype(np.float)
    # Join
    if i == 0:
        meiv2 = temp_elnino
        meiv2_months = np.arange(1,13)
        meiv2_years = np.repeat(elnino_pd[0][i], 12)
    else:
        meiv2 = np.hstack((meiv2, temp_elnino))
        meiv2_months = np.hstack((meiv2_months, np.arange(1,13)))
        meiv2_years = np.hstack((meiv2_years, np.repeat(elnino_pd[0][i], 12)))
#%%  Calculate MEI per season
# Yearly MEI
for i in np.arange(1998, 2022):
    yeartemp_meiv2 = meiv2[(meiv2_years == i)]
    if i == 1998:
        mei_yearly = np.nanmean(yeartemp_meiv2)
    else:
        mei_yearly = np.hstack((mei_yearly, np.nanmean(yeartemp_meiv2)))
# Winter MEI
for i in np.arange(1998, 2022):
    yeartemp_jun = meiv2[(meiv2_years == i) & (meiv2_months == 6)]
    yeartemp_jul = meiv2[(meiv2_years == i) & (meiv2_months == 7)]
    yeartemp_aug = meiv2[(meiv2_years == i) & (meiv2_months == 8)]
    yeartemp_JJA = np.hstack((yeartemp_jun, yeartemp_jul, yeartemp_aug))
    if i == 1998:
        mei_JJA = np.nanmean(yeartemp_JJA)
    else:
        mei_JJA = np.hstack((mei_JJA, np.nanmean(yeartemp_JJA)))
# Spring MEI
for i in np.arange(1998, 2022):
    yeartemp_sep = meiv2[(meiv2_years == i) & (meiv2_months == 9)]
    yeartemp_oct = meiv2[(meiv2_years == i) & (meiv2_months == 10)]
    yeartemp_nov = meiv2[(meiv2_years == i) & (meiv2_months == 11)]
    yeartemp_SON = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov))
    if i == 1998:
        mei_SON = np.nanmean(yeartemp_SON)
    else:
        mei_SON = np.hstack((mei_SON, np.nanmean(yeartemp_SON)))
# Summer MEI
for i in np.arange(1998, 2022):
    yeartemp_dec = meiv2[(meiv2_years == i-1) & (meiv2_months == 12)]
    yeartemp_jan = meiv2[(meiv2_years == i) & (meiv2_months == 1)]
    yeartemp_feb = meiv2[(meiv2_years == i) & (meiv2_months == 2)]
    yeartemp_DJF = np.hstack((yeartemp_dec, yeartemp_jan, yeartemp_feb))
    if i == 1998:
        mei_DJF = np.nanmean(yeartemp_DJF)
    else:
        mei_DJF = np.hstack((mei_DJF, np.nanmean(yeartemp_DJF)))
# Autumn MEI
for i in np.arange(1998, 2022):
    yeartemp_mar = meiv2[(meiv2_years == i) & (meiv2_months == 3)]
    yeartemp_apr = meiv2[(meiv2_years == i) & (meiv2_months == 4)]
    yeartemp_may = meiv2[(meiv2_years == i) & (meiv2_months == 5)]
    yeartemp_MAM = np.hstack((yeartemp_mar, yeartemp_apr))
    if i == 1998:
        mei_MAM = np.nanmean(yeartemp_MAM)
    else:
        mei_MAM = np.hstack((mei_MAM, np.nanmean(yeartemp_MAM)))
# Sep-Apr MEI
for i in np.arange(1998, 2022):
    yeartemp_sep = meiv2[(meiv2_years == i-1) & (meiv2_months == 9)]
    yeartemp_oct = meiv2[(meiv2_years == i-1) & (meiv2_months == 10)]
    yeartemp_nov = meiv2[(meiv2_years == i-1) & (meiv2_months == 11)]
    yeartemp_dec = meiv2[(meiv2_years == i-1) & (meiv2_months == 12)]
    yeartemp_jan = meiv2[(meiv2_years == i) & (meiv2_months == 1)]
    yeartemp_feb = meiv2[(meiv2_years == i) & (meiv2_months == 2)]
    yeartemp_mar = meiv2[(meiv2_years == i) & (meiv2_months == 3)]
    yeartemp_apr = meiv2[(meiv2_years == i) & (meiv2_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov,
                                 yeartemp_dec, yeartemp_jan, yeartemp_feb,
                                 yeartemp_mar, yeartemp_apr))
    if i == 1998:
        mei_sepapr = np.nanmean(yeartemp_sepapr)
    else:
        mei_sepapr = np.hstack((mei_sepapr, np.nanmean(yeartemp_sepapr)))  
#%% SST
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
fh = np.load('sst-seaice_19972021_updated.npz', allow_pickle=True)
lat_sst = fh['lat']
lon_sst = fh['lon']
sst = fh['sst']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_sstseaice.npz',allow_pickle = True)
clusters = fh['clusters']
#%% Separar para o cluster 3 (DRA)
dra_cluster = sst[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        sst_dra_sepapr = np.nanmean(yeartemp_sepapr)
        sst_dra_decfeb = np.nanmean(yeartemp_decfeb)
        sst_dra_sepnov = np.nanmean(yeartemp_sepnov)
        sst_dra_marapr = np.nanmean(yeartemp_marapr)
    else:
        sst_dra_sepapr = np.hstack((sst_dra_sepapr, np.nanmean(yeartemp_sepapr)))
        sst_dra_decfeb = np.hstack((sst_dra_decfeb, np.nanmean(yeartemp_decfeb)))
        sst_dra_sepnov = np.hstack((sst_dra_sepnov, np.nanmean(yeartemp_sepnov)))
        sst_dra_marapr = np.hstack((sst_dra_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 4 (BRS)
brs_cluster = sst[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        sst_brs_sepapr = np.nanmean(yeartemp_sepapr)
        sst_brs_decfeb = np.nanmean(yeartemp_decfeb)
        sst_brs_sepnov = np.nanmean(yeartemp_sepnov)
        sst_brs_marapr = np.nanmean(yeartemp_marapr)
    else:
        sst_brs_sepapr = np.hstack((sst_brs_sepapr, np.nanmean(yeartemp_sepapr)))
        sst_brs_decfeb = np.hstack((sst_brs_decfeb, np.nanmean(yeartemp_decfeb)))
        sst_brs_sepnov = np.hstack((sst_brs_sepnov, np.nanmean(yeartemp_sepnov)))
        sst_brs_marapr = np.hstack((sst_brs_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 5 (WEDn)
wedn_cluster = sst[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        sst_wedn_sepapr = np.nanmean(yeartemp_sepapr)
        sst_wedn_decfeb = np.nanmean(yeartemp_decfeb)
        sst_wedn_sepnov = np.nanmean(yeartemp_sepnov)
        sst_wedn_marapr = np.nanmean(yeartemp_marapr)
    else:
        sst_wedn_sepapr = np.hstack((sst_wedn_sepapr, np.nanmean(yeartemp_sepapr)))
        sst_wedn_decfeb = np.hstack((sst_wedn_decfeb, np.nanmean(yeartemp_decfeb)))
        sst_wedn_sepnov = np.hstack((sst_wedn_sepnov, np.nanmean(yeartemp_sepnov)))
        sst_wedn_marapr = np.hstack((sst_wedn_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 2 (GES)
ges_cluster = sst[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        sst_ges_sepapr = np.nanmean(yeartemp_sepapr)
        sst_ges_decfeb = np.nanmean(yeartemp_decfeb)
        sst_ges_sepnov = np.nanmean(yeartemp_sepnov)
        sst_ges_marapr = np.nanmean(yeartemp_marapr)
    else:
        sst_ges_sepapr = np.hstack((sst_ges_sepapr, np.nanmean(yeartemp_sepapr)))
        sst_ges_decfeb = np.hstack((sst_ges_decfeb, np.nanmean(yeartemp_decfeb)))
        sst_ges_sepnov = np.hstack((sst_ges_sepnov, np.nanmean(yeartemp_sepnov)))
        sst_ges_marapr = np.hstack((sst_ges_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 1 (WEDsouth)
weds_cluster = sst[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))    
    if i == 1998:
        sst_weds_sepapr = np.nanmean(yeartemp_sepapr)
        sst_weds_decfeb = np.nanmean(yeartemp_decfeb)
        sst_weds_sepnov = np.nanmean(yeartemp_sepnov)
        sst_weds_marapr = np.nanmean(yeartemp_marapr)
    else:
        sst_weds_sepapr = np.hstack((sst_weds_sepapr, np.nanmean(yeartemp_sepapr)))
        sst_weds_decfeb = np.hstack((sst_weds_decfeb, np.nanmean(yeartemp_decfeb)))
        sst_weds_sepnov = np.hstack((sst_weds_sepnov, np.nanmean(yeartemp_sepnov)))
        sst_weds_marapr = np.hstack((sst_weds_marapr, np.nanmean(yeartemp_marapr)))
#%% Sea Ice
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
fh = np.load('sst-seaice_19972021_updated.npz', allow_pickle=True)
seaice = fh['seaice']
seaice = seaice*100
#%% Separar para o cluster 3 (DRA)
dra_cluster = seaice[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        seaice_dra_sepapr = np.nanmean(yeartemp_sepapr)
        seaice_dra_decfeb = np.nanmean(yeartemp_decfeb)
        seaice_dra_sepnov = np.nanmean(yeartemp_sepnov)
        seaice_dra_marapr = np.nanmean(yeartemp_marapr)
    else:
        seaice_dra_sepapr = np.hstack((seaice_dra_sepapr, np.nanmean(yeartemp_sepapr)))
        seaice_dra_decfeb = np.hstack((seaice_dra_decfeb, np.nanmean(yeartemp_decfeb)))
        seaice_dra_sepnov = np.hstack((seaice_dra_sepnov, np.nanmean(yeartemp_sepnov)))
        seaice_dra_marapr = np.hstack((seaice_dra_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 4 (BRS)
brs_cluster = seaice[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        seaice_brs_sepapr = np.nanmean(yeartemp_sepapr)
        seaice_brs_decfeb = np.nanmean(yeartemp_decfeb)
        seaice_brs_sepnov = np.nanmean(yeartemp_sepnov)
        seaice_brs_marapr = np.nanmean(yeartemp_marapr)
    else:
        seaice_brs_sepapr = np.hstack((seaice_brs_sepapr, np.nanmean(yeartemp_sepapr)))
        seaice_brs_decfeb = np.hstack((seaice_brs_decfeb, np.nanmean(yeartemp_decfeb)))
        seaice_brs_sepnov = np.hstack((seaice_brs_sepnov, np.nanmean(yeartemp_sepnov)))
        seaice_brs_marapr = np.hstack((seaice_brs_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 5 (WEDn)
wedn_cluster = seaice[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        seaice_wedn_sepapr = np.nanmean(yeartemp_sepapr)
        seaice_wedn_decfeb = np.nanmean(yeartemp_decfeb)
        seaice_wedn_sepnov = np.nanmean(yeartemp_sepnov)
        seaice_wedn_marapr = np.nanmean(yeartemp_marapr)
    else:
        seaice_wedn_sepapr = np.hstack((seaice_wedn_sepapr, np.nanmean(yeartemp_sepapr)))
        seaice_wedn_decfeb = np.hstack((seaice_wedn_decfeb, np.nanmean(yeartemp_decfeb)))
        seaice_wedn_sepnov = np.hstack((seaice_wedn_sepnov, np.nanmean(yeartemp_sepnov)))
        seaice_wedn_marapr = np.hstack((seaice_wedn_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 2 (GES)
ges_cluster = seaice[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        seaice_ges_sepapr = np.nanmean(yeartemp_sepapr)
        seaice_ges_decfeb = np.nanmean(yeartemp_decfeb)
        seaice_ges_sepnov = np.nanmean(yeartemp_sepnov)
        seaice_ges_marapr = np.nanmean(yeartemp_marapr)
    else:
        seaice_ges_sepapr = np.hstack((seaice_ges_sepapr, np.nanmean(yeartemp_sepapr)))
        seaice_ges_decfeb = np.hstack((seaice_ges_decfeb, np.nanmean(yeartemp_decfeb)))
        seaice_ges_sepnov = np.hstack((seaice_ges_sepnov, np.nanmean(yeartemp_sepnov)))
        seaice_ges_marapr = np.hstack((seaice_ges_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 1 (WEDsouth)
weds_cluster = seaice[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))    
    if i == 1998:
        seaice_weds_sepapr = np.nanmean(yeartemp_sepapr)
        seaice_weds_decfeb = np.nanmean(yeartemp_decfeb)
        seaice_weds_sepnov = np.nanmean(yeartemp_sepnov)
        seaice_weds_marapr = np.nanmean(yeartemp_marapr)
    else:
        seaice_weds_sepapr = np.hstack((seaice_weds_sepapr, np.nanmean(yeartemp_sepapr)))
        seaice_weds_decfeb = np.hstack((seaice_weds_decfeb, np.nanmean(yeartemp_decfeb)))
        seaice_weds_sepnov = np.hstack((seaice_weds_sepnov, np.nanmean(yeartemp_sepnov)))
        seaice_weds_marapr = np.hstack((seaice_weds_marapr, np.nanmean(yeartemp_marapr)))
#%% PAR
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\par\\')
fh = np.load('par_19972021_new.npz', allow_pickle=True)
lat_par = fh['lat']
lon_par = fh['lon']
par = fh['par']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_par.npz',allow_pickle = True)
clusters = fh['clusters']
#%% Separar para o cluster 3 (DRA)
dra_cluster = par[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0, dtype=np.float64)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        par_dra_sepapr = np.nanmean(yeartemp_sepapr)
        par_dra_decfeb = np.nanmean(yeartemp_decfeb)
        par_dra_sepnov = np.nanmean(yeartemp_sepnov)
        par_dra_marapr = np.nanmean(yeartemp_marapr)
    else:
        par_dra_sepapr = np.hstack((par_dra_sepapr, np.nanmean(yeartemp_sepapr)))
        par_dra_decfeb = np.hstack((par_dra_decfeb, np.nanmean(yeartemp_decfeb)))
        par_dra_sepnov = np.hstack((par_dra_sepnov, np.nanmean(yeartemp_sepnov)))
        par_dra_marapr = np.hstack((par_dra_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 4 (BRS)
brs_cluster = par[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0, dtype=np.float64)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        par_brs_sepapr = np.nanmean(yeartemp_sepapr)
        par_brs_decfeb = np.nanmean(yeartemp_decfeb)
        par_brs_sepnov = np.nanmean(yeartemp_sepnov)
        par_brs_marapr = np.nanmean(yeartemp_marapr)
    else:
        par_brs_sepapr = np.hstack((par_brs_sepapr, np.nanmean(yeartemp_sepapr)))
        par_brs_decfeb = np.hstack((par_brs_decfeb, np.nanmean(yeartemp_decfeb)))
        par_brs_sepnov = np.hstack((par_brs_sepnov, np.nanmean(yeartemp_sepnov)))
        par_brs_marapr = np.hstack((par_brs_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 5 (WEDn)
wedn_cluster = par[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0, dtype=np.float64)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        par_wedn_sepapr = np.nanmean(yeartemp_sepapr)
        par_wedn_decfeb = np.nanmean(yeartemp_decfeb)
        par_wedn_sepnov = np.nanmean(yeartemp_sepnov)
        par_wedn_marapr = np.nanmean(yeartemp_marapr)
    else:
        par_wedn_sepapr = np.hstack((par_wedn_sepapr, np.nanmean(yeartemp_sepapr)))
        par_wedn_decfeb = np.hstack((par_wedn_decfeb, np.nanmean(yeartemp_decfeb)))
        par_wedn_sepnov = np.hstack((par_wedn_sepnov, np.nanmean(yeartemp_sepnov)))
        par_wedn_marapr = np.hstack((par_wedn_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 2 (GES)
ges_cluster = par[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0, dtype=np.float64)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        par_ges_sepapr = np.nanmean(yeartemp_sepapr)
        par_ges_decfeb = np.nanmean(yeartemp_decfeb)
        par_ges_sepnov = np.nanmean(yeartemp_sepnov)
        par_ges_marapr = np.nanmean(yeartemp_marapr)
    else:
        par_ges_sepapr = np.hstack((par_ges_sepapr, np.nanmean(yeartemp_sepapr)))
        par_ges_decfeb = np.hstack((par_ges_decfeb, np.nanmean(yeartemp_decfeb)))
        par_ges_sepnov = np.hstack((par_ges_sepnov, np.nanmean(yeartemp_sepnov)))
        par_ges_marapr = np.hstack((par_ges_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 1 (WEDsouth)
weds_cluster = par[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0, dtype=np.float64)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))    
    if i == 1998:
        par_weds_sepapr = np.nanmean(yeartemp_sepapr)
        par_weds_decfeb = np.nanmean(yeartemp_decfeb)
        par_weds_sepnov = np.nanmean(yeartemp_sepnov)
        par_weds_marapr = np.nanmean(yeartemp_marapr)
    else:
        par_weds_sepapr = np.hstack((par_weds_sepapr, np.nanmean(yeartemp_sepapr)))
        par_weds_decfeb = np.hstack((par_weds_decfeb, np.nanmean(yeartemp_decfeb)))
        par_weds_sepnov = np.hstack((par_weds_sepnov, np.nanmean(yeartemp_sepnov)))
        par_weds_marapr = np.hstack((par_weds_marapr, np.nanmean(yeartemp_marapr)))
#%% Winds
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\winds\\')
fh = np.load('winds_19972022_era5.npz', allow_pickle=True)
lat_winds = fh['lat']
lon_winds = fh['lon']
winds_u = fh['wind_u']
winds_v = fh['wind_v']
time_date = fh['time_date']
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_winds.npz',allow_pickle = True)
clusters = fh['clusters']
#%% Separar para o cluster 3 (DRA)
dra_windsu = winds_u[clusters == 3,:]
dra_windsu = np.nanmean(dra_windsu,0)
dra_windsv = winds_v[clusters == 3,:]
dra_windsv = np.nanmean(dra_windsv,0)
dra_windsu_df = pd.Series(dra_windsu, index=time_date, name='winds_u')
dra_windsu_df_daily = dra_windsu_df.resample('D').mean()
dra_windsu = dra_windsu_df_daily.values
dra_windsv_df = pd.Series(dra_windsv, index=time_date, name='winds_v')
dra_windsv_df_daily = dra_windsv_df.resample('D').mean()
dra_windsv = dra_windsv_df_daily.values
time_date_daily = dra_windsv_df_daily.index
time_date_years = time_date_daily.year
time_date_months = time_date_daily.month
dra_windsspeed = np.sqrt(dra_windsu**2 + dra_windsv**2)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = dra_windsspeed[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_windsspeed[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_windsspeed[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_windsspeed[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_windsspeed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_windsspeed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_windsspeed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_windsspeed[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        windspeed_dra_sepapr = np.nanmean(yeartemp_sepapr)
        windspeed_dra_decfeb = np.nanmean(yeartemp_decfeb)
        windspeed_dra_sepnov = np.nanmean(yeartemp_sepnov)
        windspeed_dra_marapr = np.nanmean(yeartemp_marapr)
    else:
        windspeed_dra_sepapr = np.hstack((windspeed_dra_sepapr, np.nanmean(yeartemp_sepapr)))
        windspeed_dra_decfeb = np.hstack((windspeed_dra_decfeb, np.nanmean(yeartemp_decfeb)))
        windspeed_dra_sepnov = np.hstack((windspeed_dra_sepnov, np.nanmean(yeartemp_sepnov)))
        windspeed_dra_marapr = np.hstack((windspeed_dra_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 4 (BRS)
brs_windsu = winds_u[clusters == 4,:]
brs_windsu = np.nanmean(brs_windsu,0)
brs_windsv = winds_v[clusters == 4,:]
brs_windsv = np.nanmean(brs_windsv,0)
brs_windsu_df = pd.Series(brs_windsu, index=time_date, name='winds_u')
brs_windsu_df_daily = brs_windsu_df.resample('D').mean()
brs_windsu = brs_windsu_df_daily.values
brs_windsv_df = pd.Series(brs_windsv, index=time_date, name='winds_v')
brs_windsv_df_daily = brs_windsv_df.resample('D').mean()
brs_windsv = brs_windsv_df_daily.values
time_date_daily = brs_windsv_df_daily.index
time_date_years = time_date_daily.year
time_date_months = time_date_daily.month
brs_windsspeed = np.sqrt(brs_windsu**2 + brs_windsv**2)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = brs_windsspeed[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_windsspeed[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_windsspeed[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_windsspeed[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_windsspeed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_windsspeed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_windsspeed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_windsspeed[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        windspeed_brs_sepapr = np.nanmean(yeartemp_sepapr)
        windspeed_brs_decfeb = np.nanmean(yeartemp_decfeb)
        windspeed_brs_sepnov = np.nanmean(yeartemp_sepnov)
        windspeed_brs_marapr = np.nanmean(yeartemp_marapr)
    else:
        windspeed_brs_sepapr = np.hstack((windspeed_brs_sepapr, np.nanmean(yeartemp_sepapr)))
        windspeed_brs_decfeb = np.hstack((windspeed_brs_decfeb, np.nanmean(yeartemp_decfeb)))
        windspeed_brs_sepnov = np.hstack((windspeed_brs_sepnov, np.nanmean(yeartemp_sepnov)))
        windspeed_brs_marapr = np.hstack((windspeed_brs_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 5 (WEDn)
wedn_windsu = winds_u[clusters == 5,:]
wedn_windsu = np.nanmean(wedn_windsu,0)
wedn_windsv = winds_v[clusters == 5,:]
wedn_windsv = np.nanmean(wedn_windsv,0)
wedn_windsu_df = pd.Series(wedn_windsu, index=time_date, name='winds_u')
wedn_windsu_df_daily = wedn_windsu_df.resample('D').mean()
wedn_windsu = wedn_windsu_df_daily.values
wedn_windsv_df = pd.Series(wedn_windsv, index=time_date, name='winds_v')
wedn_windsv_df_daily = wedn_windsv_df.resample('D').mean()
wedn_windsv = wedn_windsv_df_daily.values
time_date_daily = wedn_windsv_df_daily.index
time_date_years = time_date_daily.year
time_date_months = time_date_daily.month
wedn_windsspeed = np.sqrt(wedn_windsu**2 + wedn_windsv**2)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = wedn_windsspeed[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_windsspeed[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_windsspeed[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_windsspeed[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_windsspeed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_windsspeed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_windsspeed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_windsspeed[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        windspeed_wedn_sepapr = np.nanmean(yeartemp_sepapr)
        windspeed_wedn_decfeb = np.nanmean(yeartemp_decfeb)
        windspeed_wedn_sepnov = np.nanmean(yeartemp_sepnov)
        windspeed_wedn_marapr = np.nanmean(yeartemp_marapr)
    else:
        windspeed_wedn_sepapr = np.hstack((windspeed_wedn_sepapr, np.nanmean(yeartemp_sepapr)))
        windspeed_wedn_decfeb = np.hstack((windspeed_wedn_decfeb, np.nanmean(yeartemp_decfeb)))
        windspeed_wedn_sepnov = np.hstack((windspeed_wedn_sepnov, np.nanmean(yeartemp_sepnov)))
        windspeed_wedn_marapr = np.hstack((windspeed_wedn_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 2 (GES)
ges_windsu = winds_u[clusters == 2,:]
ges_windsu = np.nanmean(ges_windsu,0)
ges_windsv = winds_v[clusters == 2,:]
ges_windsv = np.nanmean(ges_windsv,0)
ges_windsu_df = pd.Series(ges_windsu, index=time_date, name='winds_u')
ges_windsu_df_daily = ges_windsu_df.resample('D').mean()
ges_windsu = ges_windsu_df_daily.values
ges_windsv_df = pd.Series(ges_windsv, index=time_date, name='winds_v')
ges_windsv_df_daily = ges_windsv_df.resample('D').mean()
ges_windsv = ges_windsv_df_daily.values
time_date_daily = ges_windsv_df_daily.index
time_date_years = time_date_daily.year
time_date_months = time_date_daily.month
ges_windsspeed = np.sqrt(ges_windsu**2 + ges_windsv**2)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = ges_windsspeed[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_windsspeed[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_windsspeed[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_windsspeed[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_windsspeed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_windsspeed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_windsspeed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_windsspeed[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        windspeed_ges_sepapr = np.nanmean(yeartemp_sepapr)
        windspeed_ges_decfeb = np.nanmean(yeartemp_decfeb)
        windspeed_ges_sepnov = np.nanmean(yeartemp_sepnov)
        windspeed_ges_marapr = np.nanmean(yeartemp_marapr)
    else:
        windspeed_ges_sepapr = np.hstack((windspeed_ges_sepapr, np.nanmean(yeartemp_sepapr)))
        windspeed_ges_decfeb = np.hstack((windspeed_ges_decfeb, np.nanmean(yeartemp_decfeb)))
        windspeed_ges_sepnov = np.hstack((windspeed_ges_sepnov, np.nanmean(yeartemp_sepnov)))
        windspeed_ges_marapr = np.hstack((windspeed_ges_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 1 (WEDsouth)
weds_windsu = winds_u[clusters == 1,:]
weds_windsu = np.nanmean(weds_windsu,0)
weds_windsv = winds_v[clusters == 1,:]
weds_windsv = np.nanmean(weds_windsv,0)
weds_windsu_df = pd.Series(weds_windsu, index=time_date, name='winds_u')
weds_windsu_df_daily = weds_windsu_df.resample('D').mean()
weds_windsu = weds_windsu_df_daily.values
weds_windsv_df = pd.Series(weds_windsv, index=time_date, name='winds_v')
weds_windsv_df_daily = weds_windsv_df.resample('D').mean()
weds_windsv = weds_windsv_df_daily.values
time_date_daily = weds_windsv_df_daily.index
time_date_years = time_date_daily.year
time_date_months = time_date_daily.month
weds_windsspeed = np.sqrt(weds_windsu**2 + weds_windsv**2)
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = weds_windsspeed[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_windsspeed[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_windsspeed[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_windsspeed[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_windsspeed[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_windsspeed[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_windsspeed[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_windsspeed[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))    
    if i == 1998:
        windspeed_weds_sepapr = np.nanmean(yeartemp_sepapr)
        windspeed_weds_decfeb = np.nanmean(yeartemp_decfeb)
        windspeed_weds_sepnov = np.nanmean(yeartemp_sepnov)
        windspeed_weds_marapr = np.nanmean(yeartemp_marapr)
    else:
        windspeed_weds_sepapr = np.hstack((windspeed_weds_sepapr, np.nanmean(yeartemp_sepapr)))
        windspeed_weds_decfeb = np.hstack((windspeed_weds_decfeb, np.nanmean(yeartemp_decfeb)))
        windspeed_weds_sepnov = np.hstack((windspeed_weds_sepnov, np.nanmean(yeartemp_sepnov)))
        windspeed_weds_marapr = np.hstack((windspeed_weds_marapr, np.nanmean(yeartemp_marapr)))
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
#%% Separar para o cluster 1 (WEDsouth)
weds_cluster = chl[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
np.nanmedian(weds_cluster)
np.nanmax(weds_cluster)
np.nanmin(weds_cluster)
np.nanstd(weds_cluster)*3
weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
weds_cluster = weds_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))

    if i == 1998:
        chl_weds_sepapr = np.nanmean(yeartemp_sepapr)
        chl_weds_decfeb = np.nanmean(yeartemp_decfeb)
        chl_weds_sepnov = np.nanmean(yeartemp_sepnov)
        chl_weds_marapr = np.nanmean(yeartemp_marapr)   
    else:
        chl_weds_sepapr = np.hstack((chl_weds_sepapr, np.nanmean(yeartemp_sepapr)))
        chl_weds_decfeb = np.hstack((chl_weds_decfeb, np.nanmean(yeartemp_decfeb)))
        chl_weds_sepnov = np.hstack((chl_weds_sepnov, np.nanmean(yeartemp_sepnov)))
        chl_weds_marapr = np.hstack((chl_weds_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 2 (GES)
ges_cluster = chl[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
np.nanmedian(ges_cluster)
np.nanmax(ges_cluster)
np.nanmin(ges_cluster)
np.nanstd(ges_cluster)*3
ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
ges_cluster = ges_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        chl_ges_sepapr = np.nanmean(yeartemp_sepapr)
        chl_ges_decfeb = np.nanmean(yeartemp_decfeb)
        chl_ges_sepnov = np.nanmean(yeartemp_sepnov)
        chl_ges_marapr = np.nanmean(yeartemp_marapr)  
    else:
        chl_ges_sepapr = np.hstack((chl_ges_sepapr, np.nanmean(yeartemp_sepapr)))
        chl_ges_decfeb = np.hstack((chl_ges_decfeb, np.nanmean(yeartemp_decfeb)))
        chl_ges_sepnov = np.hstack((chl_ges_sepnov, np.nanmean(yeartemp_sepnov)))
        chl_ges_marapr = np.hstack((chl_ges_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 4 (BRS)
brs_cluster = chl[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
np.nanmedian(brs_cluster)
np.nanmax(brs_cluster)
np.nanmin(brs_cluster)
np.nanstd(brs_cluster)*3
brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
brs_cluster = brs_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        chl_brs_sepapr = np.nanmean(yeartemp_sepapr)
        chl_brs_decfeb = np.nanmean(yeartemp_decfeb)
        chl_brs_sepnov = np.nanmean(yeartemp_sepnov)
        chl_brs_marapr = np.nanmean(yeartemp_marapr)  
    else:
        chl_brs_sepapr = np.hstack((chl_brs_sepapr, np.nanmean(yeartemp_sepapr)))
        chl_brs_decfeb = np.hstack((chl_brs_decfeb, np.nanmean(yeartemp_decfeb)))
        chl_brs_sepnov = np.hstack((chl_brs_sepnov, np.nanmean(yeartemp_sepnov)))
        chl_brs_marapr = np.hstack((chl_brs_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 5 (WEDn)
wedn_cluster = chl[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
np.nanmedian(wedn_cluster)
np.nanmax(wedn_cluster)
np.nanmin(wedn_cluster)
np.nanstd(wedn_cluster)*3
wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
wedn_cluster = wedn_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        chl_wedn_sepapr = np.nanmean(yeartemp_sepapr)
        chl_wedn_decfeb = np.nanmean(yeartemp_decfeb)
        chl_wedn_sepnov = np.nanmean(yeartemp_sepnov)
        chl_wedn_marapr = np.nanmean(yeartemp_marapr)  
    else:
        chl_wedn_sepapr = np.hstack((chl_wedn_sepapr, np.nanmean(yeartemp_sepapr)))
        chl_wedn_decfeb = np.hstack((chl_wedn_decfeb, np.nanmean(yeartemp_decfeb)))
        chl_wedn_sepnov = np.hstack((chl_wedn_sepnov, np.nanmean(yeartemp_sepnov)))
        chl_wedn_marapr = np.hstack((chl_wedn_marapr, np.nanmean(yeartemp_marapr)))
#%% Separar para o cluster 3 (DRA)
dra_cluster = chl[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
np.nanmedian(dra_cluster)
np.nanmax(dra_cluster)
np.nanmin(dra_cluster)
np.nanstd(dra_cluster)*3
dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
dra_cluster = dra_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))
    if i == 1998:
        chl_dra_sepapr = np.nanmean(yeartemp_sepapr)
        chl_dra_decfeb = np.nanmean(yeartemp_decfeb)
        chl_dra_sepnov = np.nanmean(yeartemp_sepnov)
        chl_dra_marapr = np.nanmean(yeartemp_marapr)  
    else:
        chl_dra_sepapr = np.hstack((chl_dra_sepapr, np.nanmean(yeartemp_sepapr)))
        chl_dra_decfeb = np.hstack((chl_dra_decfeb, np.nanmean(yeartemp_decfeb)))
        chl_dra_sepnov = np.hstack((chl_dra_sepnov, np.nanmean(yeartemp_sepnov)))
        chl_dra_marapr = np.hstack((chl_dra_marapr, np.nanmean(yeartemp_marapr)))
#%% Make dataframes for each season DRA
# Spring
spring_pd = pd.DataFrame(data=chl_dra_sepnov, index=np.arange(1998, 2022), columns = ['Chla'])
spring_pd['SST'] = sst_dra_sepnov
spring_pd['Sea Ice'] = seaice_dra_sepnov
spring_pd['Winds'] = windspeed_dra_sepnov
spring_pd['PAR'] = par_dra_sepnov
spring_pd['SAM'] = sam_SON
spring_pd['MEI'] = mei_SON
sns.heatmap(spring_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_DRA_Spring.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(spring_pd['Chla'], spring_pd['Winds']) # ***
# Summer
summer_pd = pd.DataFrame(data=chl_dra_decfeb, index=np.arange(1998, 2022), columns = ['Chla'])
summer_pd['SST'] = sst_dra_decfeb
summer_pd['Sea Ice'] = seaice_dra_decfeb
summer_pd['Winds'] = windspeed_dra_decfeb
summer_pd['PAR'] = par_dra_decfeb
summer_pd['SAM'] = sam_DJF
summer_pd['MEI'] = mei_DJF
sns.heatmap(summer_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_DRA_summer.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(spring_pd['Chla'], spring_pd['Winds']) # ***
# Autumn
autumn_pd = pd.DataFrame(data=chl_dra_marapr, index=np.arange(1998, 2022), columns = ['Chla'])
autumn_pd['SST'] = sst_dra_marapr
autumn_pd['Sea Ice'] = seaice_dra_marapr
autumn_pd['Winds'] = windspeed_dra_marapr
autumn_pd['PAR'] = par_dra_marapr
autumn_pd['SAM'] = sam_MAM
autumn_pd['MEI'] = mei_MAM
sns.heatmap(autumn_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_DRA_autumn.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(autumn_pd['Chla'], autumn_pd['SST']) # ***
# Sep-Apr
sepapr_pd = pd.DataFrame(data=chl_dra_sepapr, index=np.arange(1998, 2022), columns = ['Chla'])
sepapr_pd['SST'] = sst_dra_sepapr
sepapr_pd['Sea Ice'] = seaice_dra_sepapr
sepapr_pd['Winds'] = windspeed_dra_sepapr
sepapr_pd['PAR'] = par_dra_sepapr
sepapr_pd['SAM'] = sam_sepapr
sepapr_pd['MEI'] = mei_sepapr
sns.heatmap(sepapr_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_DRA_sepapr.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(sepapr_pd['Chla'], sepapr_pd['SAM']) # ***
#%% Make dataframes for each season BRS
# Spring
spring_pd = pd.DataFrame(data=chl_brs_sepnov, index=np.arange(1998, 2022), columns = ['Chla'])
spring_pd['SST'] = sst_brs_sepnov
spring_pd['Sea Ice'] = seaice_brs_sepnov
spring_pd['Winds'] = windspeed_brs_sepnov
spring_pd['PAR'] = par_brs_sepnov
spring_pd['SAM'] = sam_SON
spring_pd['MEI'] = mei_SON
sns.heatmap(spring_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_brs_Spring.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(spring_pd['Chla'], spring_pd['PAR']) # ***
# Summer
summer_pd = pd.DataFrame(data=chl_brs_decfeb, index=np.arange(1998, 2022), columns = ['Chla'])
summer_pd['SST'] = sst_brs_decfeb
summer_pd['Sea Ice'] = seaice_brs_decfeb
summer_pd['Winds'] = windspeed_brs_decfeb
summer_pd['PAR'] = par_brs_decfeb
summer_pd['SAM'] = sam_DJF
summer_pd['MEI'] = mei_DJF
sns.heatmap(summer_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_brs_summer.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(spring_pd['Chla'], spring_pd['PAR']) # ***
# Autumn
autumn_pd = pd.DataFrame(data=chl_brs_marapr, index=np.arange(1998, 2022), columns = ['Chla'])
autumn_pd['SST'] = sst_brs_marapr
autumn_pd['Sea Ice'] = seaice_brs_marapr
autumn_pd['Winds'] = windspeed_brs_marapr
autumn_pd['PAR'] = par_brs_marapr
autumn_pd['SAM'] = sam_MAM
autumn_pd['MEI'] = mei_MAM
sns.heatmap(autumn_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_brs_autumn.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(autumn_pd['Chla'], autumn_pd['PAR']) # ***
# Sep-Apr
sepapr_pd = pd.DataFrame(data=chl_brs_sepapr, index=np.arange(1998, 2022), columns = ['Chla'])
sepapr_pd['SST'] = sst_brs_sepapr
sepapr_pd['Sea Ice'] = seaice_brs_sepapr
sepapr_pd['Winds'] = windspeed_brs_sepapr
sepapr_pd['PAR'] = par_brs_sepapr
sepapr_pd['SAM'] = sam_sepapr
sepapr_pd['MEI'] = mei_sepapr
sns.heatmap(sepapr_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_brs_sepapr.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(sepapr_pd['Chla'], sepapr_pd['SAM']) # ***
#%% Make dataframes for each season WEDN
# Spring
spring_pd = pd.DataFrame(data=chl_wedn_sepnov, index=np.arange(1998, 2022), columns = ['Chla'])
spring_pd['SST'] = sst_wedn_sepnov
spring_pd['Sea Ice'] = seaice_wedn_sepnov
spring_pd['Winds'] = windspeed_wedn_sepnov
spring_pd['PAR'] = par_wedn_sepnov
spring_pd['SAM'] = sam_SON
spring_pd['MEI'] = mei_SON
sns.heatmap(spring_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_wedn_Spring.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(spring_pd['Chla'], spring_pd['Winds']) # ***
# Summer
summer_pd = pd.DataFrame(data=chl_wedn_decfeb, index=np.arange(1998, 2022), columns = ['Chla'])
summer_pd['SST'] = sst_wedn_decfeb
summer_pd['Sea Ice'] = seaice_wedn_decfeb
summer_pd['Winds'] = windspeed_wedn_decfeb
summer_pd['PAR'] = par_wedn_decfeb
summer_pd['SAM'] = sam_DJF
summer_pd['MEI'] = mei_DJF
sns.heatmap(summer_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_wedn_summer.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(summer_pd['Chla'], summer_pd['Sea Ice']) # ***
# Autumn
autumn_pd = pd.DataFrame(data=chl_wedn_marapr, index=np.arange(1998, 2022), columns = ['Chla'])
autumn_pd['SST'] = sst_wedn_marapr
autumn_pd['Sea Ice'] = seaice_wedn_marapr
autumn_pd['Winds'] = windspeed_wedn_marapr
autumn_pd['PAR'] = par_wedn_marapr
autumn_pd['SAM'] = sam_MAM
autumn_pd['MEI'] = mei_MAM
sns.heatmap(autumn_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_wedn_autumn.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Sep-Apr
sepapr_pd = pd.DataFrame(data=chl_wedn_sepapr, index=np.arange(1998, 2022), columns = ['Chla'])
sepapr_pd['SST'] = sst_wedn_sepapr
sepapr_pd['Sea Ice'] = seaice_wedn_sepapr
sepapr_pd['Winds'] = windspeed_wedn_sepapr
sepapr_pd['PAR'] = par_wedn_sepapr
sepapr_pd['SAM'] = sam_sepapr
sepapr_pd['MEI'] = mei_sepapr
sns.heatmap(sepapr_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_wedn_sepapr.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(sepapr_pd['Chla'], sepapr_pd['PAR']) # ***
#%% Make dataframes for each season GES
# Spring
spring_pd = pd.DataFrame(data=chl_ges_sepnov, index=np.arange(1998, 2022), columns = ['Chla'])
spring_pd['SST'] = sst_ges_sepnov
spring_pd['Sea Ice'] = seaice_ges_sepnov
spring_pd['Winds'] = windspeed_ges_sepnov
spring_pd['PAR'] = par_ges_sepnov
spring_pd['SAM'] = sam_SON
spring_pd['MEI'] = mei_SON
sns.heatmap(spring_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_ges_Spring.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(sepapr_pd['Chla'], sepapr_pd['PAR']) # ***
# Summer
summer_pd = pd.DataFrame(data=chl_ges_decfeb, index=np.arange(1998, 2022), columns = ['Chla'])
summer_pd['SST'] = sst_ges_decfeb
summer_pd['Sea Ice'] = seaice_ges_decfeb
summer_pd['Winds'] = windspeed_ges_decfeb
summer_pd['PAR'] = par_ges_decfeb
summer_pd['SAM'] = sam_DJF
summer_pd['MEI'] = mei_DJF
sns.heatmap(summer_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_ges_summer.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Autumn
autumn_pd = pd.DataFrame(data=chl_ges_marapr, index=np.arange(1998, 2022), columns = ['Chla'])
autumn_pd['SST'] = sst_ges_marapr
autumn_pd['Sea Ice'] = seaice_ges_marapr
autumn_pd['Winds'] = windspeed_ges_marapr
autumn_pd['PAR'] = par_ges_marapr
autumn_pd['SAM'] = sam_MAM
autumn_pd['MEI'] = mei_MAM
sns.heatmap(autumn_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_ges_autumn.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(autumn_pd['Chla'], autumn_pd['SAM']) # ***
# Sep-Apr
sepapr_pd = pd.DataFrame(data=chl_ges_sepapr, index=np.arange(1998, 2022), columns = ['Chla'])
sepapr_pd['SST'] = sst_ges_sepapr
sepapr_pd['Sea Ice'] = seaice_ges_sepapr
sepapr_pd['Winds'] = windspeed_ges_sepapr
sepapr_pd['PAR'] = par_ges_sepapr
sepapr_pd['SAM'] = sam_sepapr
sepapr_pd['MEI'] = mei_sepapr
sns.heatmap(sepapr_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_ges_sepapr.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Make dataframes for each season WEDS
# Spring
spring_pd = pd.DataFrame(data=chl_weds_sepnov, index=np.arange(1998, 2022), columns = ['Chla'])
spring_pd['SST'] = sst_weds_sepnov
spring_pd['Sea Ice'] = seaice_weds_sepnov
spring_pd['Winds'] = windspeed_weds_sepnov
spring_pd['PAR'] = par_weds_sepnov
spring_pd['SAM'] = sam_SON
spring_pd['MEI'] = mei_SON
sns.heatmap(spring_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_weds_Spring.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(spring_pd['Chla'], spring_pd['PAR']) # ***
# Summer
summer_pd = pd.DataFrame(data=chl_weds_decfeb, index=np.arange(1998, 2022), columns = ['Chla'])
summer_pd['SST'] = sst_weds_decfeb
summer_pd['Sea Ice'] = seaice_weds_decfeb
summer_pd['Winds'] = windspeed_weds_decfeb
summer_pd['PAR'] = par_weds_decfeb
summer_pd['SAM'] = sam_DJF
summer_pd['MEI'] = mei_DJF
sns.heatmap(summer_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_weds_summer.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Autumn
autumn_pd = pd.DataFrame(data=chl_weds_marapr, index=np.arange(1998, 2022), columns = ['Chla'])
autumn_pd['SST'] = sst_weds_marapr
autumn_pd['Sea Ice'] = seaice_weds_marapr
autumn_pd['Winds'] = windspeed_weds_marapr
autumn_pd['PAR'] = par_weds_marapr
autumn_pd['SAM'] = sam_MAM
autumn_pd['MEI'] = mei_MAM
sns.heatmap(autumn_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_weds_autumn.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(autumn_pd['Chla'], autumn_pd['PAR']) # ***
# Sep-Apr
sepapr_pd = pd.DataFrame(data=chl_weds_sepapr, index=np.arange(1998, 2022), columns = ['Chla'])
sepapr_pd['SST'] = sst_weds_sepapr
sepapr_pd['Sea Ice'] = seaice_weds_sepapr
sepapr_pd['Winds'] = windspeed_weds_sepapr
sepapr_pd['PAR'] = par_weds_sepapr
sepapr_pd['SAM'] = sam_sepapr
sepapr_pd['MEI'] = mei_sepapr
sns.heatmap(sepapr_pd.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\27072023\\correlationmatrix_weds_sepapr.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
stats.pearsonr(sepapr_pd['Chla'], sepapr_pd['Sea Ice']) # ***
#%% DRA check significants
stats.pearsonr(Chla, spring_pd)











#%%
# SAM vs Chl-a Sep-Apr
stats.spearmanr(sam_sepapr[~np.isnan(chl_dra_sepapr)], chl_dra_sepapr[~np.isnan(chl_dra_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(chl_brs_sepapr)], chl_brs_sepapr[~np.isnan(chl_brs_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(chl_wedn_sepapr)], chl_wedn_sepapr[~np.isnan(chl_wedn_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(chl_ges_sepapr)], chl_ges_sepapr[~np.isnan(chl_ges_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(chl_weds_sepapr)], chl_weds_sepapr[~np.isnan(chl_weds_sepapr)], nan_policy='omit')
# MEI vs Chl-a Sep-Apr
stats.spearmanr(mei_sepapr[~np.isnan(chl_dra_sepapr)], chl_dra_sepapr[~np.isnan(chl_dra_sepapr)], nan_policy='omit')
stats.spearmanr(mei_sepapr[~np.isnan(chl_brs_sepapr)], chl_brs_sepapr[~np.isnan(chl_brs_sepapr)], nan_policy='omit')
stats.spearmanr(mei_sepapr[~np.isnan(chl_wedn_sepapr)], chl_wedn_sepapr[~np.isnan(chl_wedn_sepapr)], nan_policy='omit')
stats.spearmanr(mei_sepapr[~np.isnan(chl_ges_sepapr)], chl_ges_sepapr[~np.isnan(chl_ges_sepapr)], nan_policy='omit')
stats.spearmanr(mei_sepapr[~np.isnan(chl_weds_sepapr)], chl_weds_sepapr[~np.isnan(chl_weds_sepapr)], nan_policy='omit')






### SAM
## CHL
# Spring
stats.spearmanr(sam_SON[~np.isnan(chl_dra_sepnov)], chl_dra_sepnov[~np.isnan(chl_dra_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(chl_brs_sepnov)], chl_brs_sepnov[~np.isnan(chl_brs_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(chl_wedn_sepnov)], chl_wedn_sepnov[~np.isnan(chl_wedn_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(chl_ges_sepnov)], chl_ges_sepnov[~np.isnan(chl_ges_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(chl_weds_sepnov)], chl_weds_sepnov[~np.isnan(chl_weds_sepnov)], nan_policy='omit')
# Summer
stats.spearmanr(sam_DJF[~np.isnan(chl_dra_decfeb)], chl_dra_decfeb[~np.isnan(chl_dra_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(chl_brs_decfeb)], chl_brs_decfeb[~np.isnan(chl_brs_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(chl_wedn_decfeb)], chl_wedn_decfeb[~np.isnan(chl_wedn_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(chl_ges_decfeb)], chl_ges_decfeb[~np.isnan(chl_ges_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(chl_weds_decfeb)], chl_weds_decfeb[~np.isnan(chl_weds_decfeb)], nan_policy='omit')
# Autumn
stats.spearmanr(sam_MAM[~np.isnan(chl_dra_marapr)], chl_dra_marapr[~np.isnan(chl_dra_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(chl_brs_marapr)], chl_brs_marapr[~np.isnan(chl_brs_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(chl_wedn_marapr)], chl_wedn_marapr[~np.isnan(chl_wedn_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(chl_ges_marapr)], chl_ges_marapr[~np.isnan(chl_ges_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(chl_weds_marapr)], chl_weds_marapr[~np.isnan(chl_weds_marapr)], nan_policy='omit')
# Sep-Apr
stats.spearmanr(sam_sepapr[~np.isnan(chl_dra_sepapr)], chl_dra_sepapr[~np.isnan(chl_dra_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(chl_brs_sepapr)], chl_brs_sepapr[~np.isnan(chl_brs_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(chl_wedn_sepapr)], chl_wedn_sepapr[~np.isnan(chl_wedn_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(chl_ges_sepapr)], chl_ges_sepapr[~np.isnan(chl_ges_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(chl_weds_sepapr)], chl_weds_sepapr[~np.isnan(chl_weds_sepapr)], nan_policy='omit')
## SST
# Spring
stats.spearmanr(sam_SON[~np.isnan(sst_dra_sepnov)], sst_dra_sepnov[~np.isnan(sst_dra_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(sst_brs_sepnov)], sst_brs_sepnov[~np.isnan(sst_brs_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(sst_wedn_sepnov)], sst_wedn_sepnov[~np.isnan(sst_wedn_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(sst_ges_sepnov)], sst_ges_sepnov[~np.isnan(sst_ges_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(sst_weds_sepnov)], sst_weds_sepnov[~np.isnan(sst_weds_sepnov)], nan_policy='omit')
# Summer
stats.spearmanr(sam_DJF[~np.isnan(sst_dra_decfeb)], sst_dra_decfeb[~np.isnan(sst_dra_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(sst_brs_decfeb)], sst_brs_decfeb[~np.isnan(sst_brs_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(sst_wedn_decfeb)], sst_wedn_decfeb[~np.isnan(sst_wedn_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(sst_ges_decfeb)], sst_ges_decfeb[~np.isnan(sst_ges_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(sst_weds_decfeb)], sst_weds_decfeb[~np.isnan(sst_weds_decfeb)], nan_policy='omit')
# Autumn
stats.spearmanr(sam_MAM[~np.isnan(sst_dra_marapr)], sst_dra_marapr[~np.isnan(sst_dra_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(sst_brs_marapr)], sst_brs_marapr[~np.isnan(sst_brs_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(sst_wedn_marapr)], sst_wedn_marapr[~np.isnan(sst_wedn_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(sst_ges_marapr)], sst_ges_marapr[~np.isnan(sst_ges_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(sst_weds_marapr)], sst_weds_marapr[~np.isnan(sst_weds_marapr)], nan_policy='omit')
# Sep-Apr
stats.spearmanr(sam_sepapr[~np.isnan(sst_dra_sepapr)], sst_dra_sepapr[~np.isnan(sst_dra_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(sst_brs_sepapr)], sst_brs_sepapr[~np.isnan(sst_brs_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(sst_wedn_sepapr)], sst_wedn_sepapr[~np.isnan(sst_wedn_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(sst_ges_sepapr)], sst_ges_sepapr[~np.isnan(sst_ges_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(sst_weds_sepapr)], sst_weds_sepapr[~np.isnan(sst_weds_sepapr)], nan_policy='omit')
## Sea Ice
# Spring
stats.spearmanr(sam_SON[~np.isnan(seaice_dra_sepnov)], seaice_dra_sepnov[~np.isnan(seaice_dra_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(seaice_brs_sepnov)], seaice_brs_sepnov[~np.isnan(seaice_brs_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(seaice_wedn_sepnov)], seaice_wedn_sepnov[~np.isnan(seaice_wedn_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(seaice_ges_sepnov)], seaice_ges_sepnov[~np.isnan(seaice_ges_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(seaice_weds_sepnov)], seaice_weds_sepnov[~np.isnan(seaice_weds_sepnov)], nan_policy='omit')
# Summer
stats.spearmanr(sam_DJF[~np.isnan(seaice_dra_decfeb)], seaice_dra_decfeb[~np.isnan(seaice_dra_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(seaice_brs_decfeb)], seaice_brs_decfeb[~np.isnan(seaice_brs_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(seaice_wedn_decfeb)], seaice_wedn_decfeb[~np.isnan(seaice_wedn_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(seaice_ges_decfeb)], seaice_ges_decfeb[~np.isnan(seaice_ges_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(seaice_weds_decfeb)], seaice_weds_decfeb[~np.isnan(seaice_weds_decfeb)], nan_policy='omit')
# Autumn
stats.spearmanr(sam_MAM[~np.isnan(seaice_dra_marapr)], seaice_dra_marapr[~np.isnan(seaice_dra_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(seaice_brs_marapr)], seaice_brs_marapr[~np.isnan(seaice_brs_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(seaice_wedn_marapr)], seaice_wedn_marapr[~np.isnan(seaice_wedn_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(seaice_ges_marapr)], seaice_ges_marapr[~np.isnan(seaice_ges_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(seaice_weds_marapr)], seaice_weds_marapr[~np.isnan(seaice_weds_marapr)], nan_policy='omit')
# Sep-Apr
stats.spearmanr(sam_sepapr[~np.isnan(seaice_dra_sepapr)], seaice_dra_sepapr[~np.isnan(seaice_dra_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(seaice_brs_sepapr)], seaice_brs_sepapr[~np.isnan(seaice_brs_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(seaice_wedn_sepapr)], seaice_wedn_sepapr[~np.isnan(seaice_wedn_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(seaice_ges_sepapr)], seaice_ges_sepapr[~np.isnan(seaice_ges_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(seaice_weds_sepapr)], seaice_weds_sepapr[~np.isnan(seaice_weds_sepapr)], nan_policy='omit')
## PAR
# Spring
stats.spearmanr(sam_SON[~np.isnan(par_dra_sepnov)], par_dra_sepnov[~np.isnan(par_dra_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(par_brs_sepnov)], par_brs_sepnov[~np.isnan(par_brs_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(par_wedn_sepnov)], par_wedn_sepnov[~np.isnan(par_wedn_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(par_ges_sepnov)], par_ges_sepnov[~np.isnan(par_ges_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(par_weds_sepnov)], par_weds_sepnov[~np.isnan(par_weds_sepnov)], nan_policy='omit')
# Summer
stats.spearmanr(sam_DJF[~np.isnan(par_dra_decfeb)], par_dra_decfeb[~np.isnan(par_dra_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(par_brs_decfeb)], par_brs_decfeb[~np.isnan(par_brs_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(par_wedn_decfeb)], par_wedn_decfeb[~np.isnan(par_wedn_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(par_ges_decfeb)], par_ges_decfeb[~np.isnan(par_ges_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(par_weds_decfeb)], par_weds_decfeb[~np.isnan(par_weds_decfeb)], nan_policy='omit')
# Autumn
stats.spearmanr(sam_MAM[~np.isnan(par_dra_marapr)], par_dra_marapr[~np.isnan(par_dra_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(par_brs_marapr)], par_brs_marapr[~np.isnan(par_brs_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(par_wedn_marapr)], par_wedn_marapr[~np.isnan(par_wedn_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(par_ges_marapr)], par_ges_marapr[~np.isnan(par_ges_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(par_weds_marapr)], par_weds_marapr[~np.isnan(par_weds_marapr)], nan_policy='omit')
# Sep-Apr
stats.spearmanr(sam_sepapr[~np.isnan(par_dra_sepapr)], par_dra_sepapr[~np.isnan(par_dra_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(par_brs_sepapr)], par_brs_sepapr[~np.isnan(par_brs_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(par_wedn_sepapr)], par_wedn_sepapr[~np.isnan(par_wedn_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(par_ges_sepapr)], par_ges_sepapr[~np.isnan(par_ges_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(par_weds_sepapr)], par_weds_sepapr[~np.isnan(par_weds_sepapr)], nan_policy='omit')
## Wind speed
# Spring
stats.spearmanr(sam_SON[~np.isnan(windspeed_dra_sepnov)], windspeed_dra_sepnov[~np.isnan(windspeed_dra_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(windspeed_brs_sepnov)], windspeed_brs_sepnov[~np.isnan(windspeed_brs_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(windspeed_wedn_sepnov)], windspeed_wedn_sepnov[~np.isnan(windspeed_wedn_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(windspeed_ges_sepnov)], windspeed_ges_sepnov[~np.isnan(windspeed_ges_sepnov)], nan_policy='omit')
stats.spearmanr(sam_SON[~np.isnan(windspeed_weds_sepnov)], windspeed_weds_sepnov[~np.isnan(windspeed_weds_sepnov)], nan_policy='omit')
# Summer
stats.spearmanr(sam_DJF[~np.isnan(windspeed_dra_decfeb)], windspeed_dra_decfeb[~np.isnan(windspeed_dra_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(windspeed_brs_decfeb)], windspeed_brs_decfeb[~np.isnan(windspeed_brs_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(windspeed_wedn_decfeb)], windspeed_wedn_decfeb[~np.isnan(windspeed_wedn_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(windspeed_ges_decfeb)], windspeed_ges_decfeb[~np.isnan(windspeed_ges_decfeb)], nan_policy='omit')
stats.spearmanr(sam_DJF[~np.isnan(windspeed_weds_decfeb)], windspeed_weds_decfeb[~np.isnan(windspeed_weds_decfeb)], nan_policy='omit')
# Autumn
stats.spearmanr(sam_MAM[~np.isnan(windspeed_dra_marapr)], windspeed_dra_marapr[~np.isnan(windspeed_dra_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(windspeed_brs_marapr)], windspeed_brs_marapr[~np.isnan(windspeed_brs_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(windspeed_wedn_marapr)], windspeed_wedn_marapr[~np.isnan(windspeed_wedn_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(windspeed_ges_marapr)], windspeed_ges_marapr[~np.isnan(windspeed_ges_marapr)], nan_policy='omit')
stats.spearmanr(sam_MAM[~np.isnan(windspeed_weds_marapr)], windspeed_weds_marapr[~np.isnan(windspeed_weds_marapr)], nan_policy='omit')
# Sep-Apr
stats.spearmanr(sam_sepapr[~np.isnan(windspeed_dra_sepapr)], windspeed_dra_sepapr[~np.isnan(windspeed_dra_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(windspeed_brs_sepapr)], windspeed_brs_sepapr[~np.isnan(windspeed_brs_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(windspeed_wedn_sepapr)], windspeed_wedn_sepapr[~np.isnan(windspeed_wedn_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(windspeed_ges_sepapr)], windspeed_ges_sepapr[~np.isnan(windspeed_ges_sepapr)], nan_policy='omit')
stats.spearmanr(sam_sepapr[~np.isnan(windspeed_weds_sepapr)], windspeed_weds_sepapr[~np.isnan(windspeed_weds_sepapr)], nan_policy='omit')
























