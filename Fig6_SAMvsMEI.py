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
#%% Load MEI
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
#%% Load SAM
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sam\\')
sam_pd = pd.read_csv('norm.daily.aao.cdas.z700.19790101_current.csv', sep=',')
sam_daily = sam_pd['aao_index_cdas'].values
time_date_years = sam_pd['year'].values
time_date_months = sam_pd['month'].values
time_date_days = sam_pd['day'].values
#%% Load sea ice duration
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaicetimings\\')
seaicetiming = pd.read_csv('ice_timing.csv', sep=';')
seaicetiming_years = np.arange(1980, 2022)
seaiceduration = seaicetiming['Pori dur'].values
seaiceduration = seaiceduration[18:]
normalized_duration = (seaiceduration-min(seaiceduration))/(max(seaiceduration)-min(seaiceduration))
#%% Load sea ice extent
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaiceextent\\')
seaiceextentdata = pd.read_csv('seaiceextent.csv', sep=';')
seaiceextent_years = np.arange(1980, 2022)
seaiceextent = seaiceextentdata['WAP_Ext'].values
seaiceextent = seaiceextent[18:]
normalized_extent = (seaiceextent-min(seaiceextent))/(max(seaiceextent)-min(seaiceextent))
#%% Average monthly
time_date_sam_daily = np.empty_like(time_date_days, dtype=object)
for i in range(0, len(time_date_sam_daily)):
    time_date_sam_daily[i] = datetime.datetime(year = time_date_years[i],
                                               month = time_date_months[i],
                                               day = time_date_days[i])
sam_pd_daily = pd.Series(data=sam_daily, index=time_date_sam_daily)
# resample monthly
sam_pd_monthly = sam_pd_daily.resample('M').mean()
# calculate 1 year moving mean
sam_monthly_movingmean = sam_pd_monthly.rolling(12).mean().values
monthly_dates = sam_pd_monthly.index
sam_pd_monthly_cumsum = sam_pd_monthly.cumsum()
#%%
# SEP-APR
for i in np.arange(1998, 2022):
    yeartemp_sepdec = sam_daily[(time_date_years == i-1) & ((time_date_months == 9) | (time_date_months == 10)
                                                                | (time_date_months == 11) | (time_date_months == 12))]
    yeartemp_janapr = sam_daily[(time_date_years == i) & ((time_date_months == 1) | (time_date_months == 2)
                                                                | (time_date_months == 3) | (time_date_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        sam_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        sam_SEPAPR = np.hstack((sam_SEPAPR, np.nanmean(yeartemp_SEPAPR)))

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
#%% Original
plt.scatter(sam_SEPAPR, mei_sepapr)
plt.xlabel('SAM SEP-APR', fontsize=12)
plt.ylabel('MEI SEP-APR', fontsize=12)
plt.axhline(c='k', alpha=0.2)
plt.axvline(c='k', alpha=0.2)
slope, intercept, rvalue, pvalue , _ = stats.linregress(sam_SEPAPR, mei_sepapr)
plt.plot(sam_SEPAPR, sam_SEPAPR*slope + intercept, c='r')
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\SAMvsMEI.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Without El Niños (MEI > 0.5)
plt.scatter(sam_SEPAPR[mei_sepapr<0.5], mei_sepapr[mei_sepapr<0.5], s=50, c='k', label='Non El-Niño')
plt.scatter(sam_SEPAPR[mei_sepapr>0.5], mei_sepapr[mei_sepapr>0.5], s=50, c='w', edgecolor='k', label='El-Niño')
plt.xlabel('SAM SEP-APR', fontsize=12)
plt.ylabel('MEI SEP-APR', fontsize=12)
plt.axhline(c='k', alpha=0.2)
plt.axvline(c='k', alpha=0.2)
slope, intercept, rvalue, pvalue , _ = stats.linregress(sam_SEPAPR[mei_sepapr<0.5], mei_sepapr[mei_sepapr<0.5])
plt.plot(sam_SEPAPR, sam_SEPAPR*slope + intercept, c='k')
plt.legend(loc=0, fontsize=12)
slope, intercept, rvalue, pvalue , _ = stats.linregress(sam_SEPAPR[mei_sepapr>0.5], mei_sepapr[mei_sepapr>0.5])
plt.plot(sam_SEPAPR, sam_SEPAPR*slope + intercept, c='k', linestyle='--', alpha=0.2)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\SAMvsMEI_SupFig4.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
plt.scatter(sam_SEPAPR[mei_sepapr<0.5], mei_sepapr[mei_sepapr<0.5], s=normalized_extent[mei_sepapr<0.5]*150+1)
plt.scatter(sam_SEPAPR[mei_sepapr>0.5], mei_sepapr[mei_sepapr>0.5], s=normalized_extent[mei_sepapr>0.5]*150+1)
plt.xlabel('SAM SEP-APR', fontsize=12)
plt.ylabel('MEI SEP-APR', fontsize=12)
plt.axhline(c='k', alpha=0.2)
plt.axvline(c='k', alpha=0.2)
slope, intercept, rvalue, pvalue , _ = stats.linregress(sam_SEPAPR[mei_sepapr>0.5], mei_sepapr[mei_sepapr>0.5])
slope, intercept, rvalue, pvalue , _ = stats.linregress(sam_SEPAPR[mei_sepapr<0.5], mei_sepapr[mei_sepapr<0.5])
plt.plot(sam_SEPAPR, sam_SEPAPR*slope + intercept, c='orange')
plt.plot(sam_SEPAPR, sam_SEPAPR*slope + intercept, c='b')
plt.tight_layout()