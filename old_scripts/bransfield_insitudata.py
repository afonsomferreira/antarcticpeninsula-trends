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
#%% Load In-situ chl data
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\dados_insitu\\')
chl_df = pd.read_csv('chl_hplc_ready.csv', sep=';')
lat_insitu = chl_df['lat'].values
lon_insitu = chl_df['lon'].values
bransfield_verts = [(-65, -64.5),
             (-65, -63),
             (-52.5, -60),
             (-52.5, -61.6)]
chl_insitu = chl_df['Chl'].values
date_insitu = chl_df['date'].values
datetime_insitu = np.empty(len(date_insitu), dtype=object)
for i in range(0, len(datetime_insitu)):
    datetime_insitu[i] = datetime.datetime(year=int(date_insitu[i][6:10]),
                                           month=int(date_insitu[i][3:5]),
                                           day=int(date_insitu[i][:2]))
# Find which points are located within the Bransfield Area
points = np.vstack((lon_insitu, lat_insitu)).T
p = Path(bransfield_verts) # make a polygon
inside2 = p.contains_points(points) 
# Keep points only from within the region
chl_insitu_bransfield = chl_insitu[inside2]
datetime_insitu_bransfield = datetime_insitu[inside2]
# Average for each Sep-April
time_date_years = np.empty_like(datetime_insitu_bransfield)
time_date_months = np.empty_like(datetime_insitu_bransfield)
for i in range(0, len(datetime_insitu_bransfield)):
    time_date_years[i] = datetime_insitu_bransfield[i].year
    time_date_months[i] = datetime_insitu_bransfield[i].month
# Count number of data per month
# SEPTEMBER
for i in np.arange(1997, 2019):
    sep_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 9)])
    if i == 1997:
        sep_count_19812021 = sep_count
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        sep_count_19812021 = np.hstack((sep_count_19812021, sep_count))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# OCTOBER
for i in np.arange(1997, 2019):
    oct_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 10)])
    if i == 1997:
        oct_count_19812021 = oct_count
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        oct_count_19812021 = np.hstack((oct_count_19812021, oct_count))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# NOVEMBER
for i in np.arange(1997, 2019):
    nov_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 11)])
    if i == 1997:
        nov_count_19812021 = nov_count
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        nov_count_19812021 = np.hstack((nov_count_19812021, nov_count))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# DECEMBER
for i in np.arange(1997, 2019):
    dec_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 12)])
    if i == 1997:
        dec_count_19812021 = dec_count
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        dec_count_19812021 = np.hstack((dec_count_19812021, dec_count))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# JANUARY
for i in np.arange(1997, 2019):
    jan_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 1)])
    if i == 1997:
        jan_count_19812021 = jan_count
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        jan_count_19812021 = np.hstack((jan_count_19812021, jan_count))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# FEBRUARY
for i in np.arange(1997, 2019):
    feb_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 2)])
    if i == 1997:
        feb_count_19812021 = feb_count
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        feb_count_19812021 = np.hstack((feb_count_19812021, feb_count))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# MARCH
for i in np.arange(1997, 2019):
    mar_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 3)])
    if i == 1997:
        mar_count_19812021 = mar_count
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        mar_count_19812021 = np.hstack((mar_count_19812021, mar_count))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
## 
# JAN-FEBRUARY ONLY
for i in np.arange(1997, 2019):
    jan_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 1)])
    feb_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 2)])
    if i == 1997:
        janfeb_count_19812021 = jan_count+feb_count
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        janfeb_count_19812021 = np.hstack((janfeb_count_19812021, jan_count+feb_count))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
# JAN-FEB-MARCH
for i in np.arange(1997, 2019):
    jan_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 1)])
    feb_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 2)])
    mar_count = np.size(chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 3)])
    if i == 1997:
        janmar_count_19812021 = jan_count+feb_count+mar_count
#        sst_summerstds19982021 = yeartemp_summerstd
    else:
        janmar_count_19812021 = np.hstack((janmar_count_19812021, jan_count+feb_count+mar_count))
#        sst_summerstds19982021 = np.hstack((sst_summerstds19982021, yeartemp_summerstd))
## Just Jan-Feb Means
for i in np.arange(1997, 2019):
    yeartemp_jan = chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 2)]
#    yeartemp_mar = chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_jan, yeartemp_feb
                                                )))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_jan, yeartemp_feb
                                                )))    
    if i == 1997:
        chlinsitu_janfeb19812021 = yeartemp_summermean
        chlinsitustd_janfeb19982021 = yeartemp_summerstd
    else:
        chlinsitu_janfeb19812021 = np.hstack((chlinsitu_janfeb19812021, yeartemp_summermean))
        chlinsitustd_janfeb19982021 = np.hstack((chlinsitustd_janfeb19982021, yeartemp_summerstd))
# Remove years with fewer than 10 samples
chlinsitu_janfeb19812021[janfeb_count_19812021<10] = np.nan
chlinsitustd_janfeb19982021[janfeb_count_19812021<10] = np.nan
## Just Jan-Mar Means
for i in np.arange(1997, 2019):
    yeartemp_jan = chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = chl_insitu_bransfield[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_summermean = np.nanmean(np.hstack((yeartemp_jan, yeartemp_feb, yeartemp_mar
                                                )))
    yeartemp_summerstd = np.nanstd(np.hstack((yeartemp_jan, yeartemp_feb, yeartemp_mar
                                                )))    
    if i == 1997:
        chlinsitu_janmar19812021 = yeartemp_summermean
        chlinsitustd_janmar19982021 = yeartemp_summerstd
    else:
        chlinsitu_janmar19812021 = np.hstack((chlinsitu_janmar19812021, yeartemp_summermean))
        chlinsitustd_janmar19982021 = np.hstack((chlinsitustd_janmar19982021, yeartemp_summerstd))
# Remove years with fewer than 10 samples
chlinsitu_janfeb19812021[janfeb_count_19812021<10] = np.nan
chlinsitustd_janfeb19982021[janfeb_count_19812021<10] = np.nan
# Remove years with fewer than 10 samples
chlinsitu_janmar19812021[janmar_count_19812021<10] = np.nan
chlinsitustd_janmar19982021[janmar_count_19812021<10] = np.nan
#%% Plot Jan-Feb
(_, caps, _) = plt.errorbar(np.arange(1997, 2019), chlinsitu_janfeb19812021, yerr=chlinsitustd_janfeb19982021, capsize=3, elinewidth=1, linestyle='None', marker='o', c='k')
[slope,intercept,r,pval,_] = stats.linregress(np.arange(1997, 2019)[~np.isnan(chlinsitu_janfeb19812021)], chlinsitu_janfeb19812021[~np.isnan(chlinsitu_janfeb19812021)])
plt.plot(np.arange(1997,2019), np.arange(1997,2019)*slope+intercept, linestyle='--', c='b', alpha=0.5)
plt.text(1998, 6, f"R={r:.2f}", fontsize=14)
plt.text(1998, 5.5, f"p-val={pval:.2f}", fontsize=14)
plt.xlabel('Years', fontsize=14)
plt.ylabel('Chla', fontsize=14)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\')
graphs_dir = 'bransfield_insituchlhplc_janfebmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%%
(_, caps, _) = plt.errorbar(np.arange(1997, 2019), chlinsitu_janmar19812021, yerr=chlinsitustd_janmar19982021, capsize=3, elinewidth=1, linestyle='None', marker='o', c='k')
[slope,intercept,r,pval,_] = stats.linregress(np.arange(1997, 2019)[~np.isnan(chlinsitu_janmar19812021)], chlinsitu_janmar19812021[~np.isnan(chlinsitu_janmar19812021)])
plt.plot(np.arange(1997,2019), np.arange(1997,2019)*slope+intercept, linestyle='--', c='b', alpha=0.5)
plt.text(1998, 6, f"R={r:.2f}", fontsize=14)
plt.text(1998, 5.5, f"p-val={pval:.2f}", fontsize=14)
plt.xlabel('Years', fontsize=14)
plt.ylabel('Chla', fontsize=14)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\')
graphs_dir = 'bransfield_insituchlhplc_janmarmean.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%%
plt.bar(np.arange(1997,2019), janfeb_count_19812021)
plt.xlabel('Years', fontsize=14)
plt.ylabel('N', fontsize=14)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\')
graphs_dir = 'bransfield_insituchlhplc_janfeb_counts.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
plt.bar(np.arange(1997,2019), janmar_count_19812021)
plt.xlabel('Years', fontsize=14)
plt.ylabel('N', fontsize=14)
plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\')
graphs_dir = 'bransfield_insituchlhplc_janmar_counts.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%%


[slope,intercept,r,pval,_] = stats.linregress(np.arange(2006, 2019)[~np.isnan(chlinsitu_sepapr19812021[9:])], chlinsitu_sepapr19812021[9:][~np.isnan(chlinsitu_sepapr19812021[9:])])

