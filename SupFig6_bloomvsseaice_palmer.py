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
#%% Load sea ice duration
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaicetimings\\')
seaicetiming = pd.read_csv('ice_timing.csv', sep=';')
seaicetiming_years = np.arange(1980, 2022)
seaiceadvance = seaicetiming['Pori adv'].values
seaiceretreat = seaicetiming['Pori ret'].values
#%% Load bloom phenology metrics
### Load data 1998-2022
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
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = weds_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = weds_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
#    b_magnitude_temp = np.nansum(yeartemp_sepapr_pd_8day.values[b_init_temp:b_term_temp+1])/(len(yeartemp_sepapr_pd_8day.values[b_init_temp:b_term_temp+1]) - np.sum(np.isnan(yeartemp_sepapr_pd_8day.values[b_init_temp:b_term_temp+1])))
    # Average for 8 days
    if i == 1998:
        weds_b_init = b_init_temp
        weds_b_term = b_term_temp
    else:
        weds_b_init = np.hstack((weds_b_init, b_init_temp))    
        weds_b_term = np.hstack((weds_b_term, b_term_temp))     
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
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = ges_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = ges_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
    # Average for 8 days
    if i == 1998:
        ges_b_init = b_init_temp
        ges_b_term = b_term_temp
        
    else:
        ges_b_init = np.hstack((ges_b_init, b_init_temp))    
        ges_b_term = np.hstack((ges_b_term, b_term_temp))     
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
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = dra_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = dra_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
    # Average for 8 days
    if i == 1998:
        dra_b_init = b_init_temp
        dra_b_term = b_term_temp
        
    else:
        dra_b_init = np.hstack((dra_b_init, b_init_temp))    
        dra_b_term = np.hstack((dra_b_term, b_term_temp))    
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
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = brs_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = brs_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
    # Average for 8 days
    if i == 1998:
        brs_b_init = b_init_temp
        brs_b_term = b_term_temp
        
    else:
        brs_b_init = np.hstack((brs_b_init, b_init_temp))    
        brs_b_term = np.hstack((brs_b_term, b_term_temp))    
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
for i in np.arange(1998, 2023):
    ix = pd.date_range(start=datetime.date(i-1, 9, 1), end=datetime.date(i, 4, 30), freq='D')
    # Extract august to may
    if i == 1998:
        yeartemp_sep = 0
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 4))[-1][-1]
        yeartemp_sepapr = wedn_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
    else:
        yeartemp_sep = np.where((time_date_years == i-1) & (time_date_months == 8))[0][0]
        yeartemp_apr = np.where((time_date_years == i) & (time_date_months == 5))[-1][-1]
        yeartemp_sepapr = wedn_cluster[yeartemp_sep:yeartemp_apr+1]
        yeartemp_sepapr_pd = pd.Series(yeartemp_sepapr, index=time_date[yeartemp_sep:yeartemp_apr+1])
        yeartemp_sepapr_pd = yeartemp_sepapr_pd.reindex(ix)
        if len(yeartemp_sepapr_pd) > 242:
          yeartemp_sepapr_pd = yeartemp_sepapr_pd.drop([yeartemp_sepapr_pd.index[181]])
    # Convert to 8D weeks
    yeartemp_sepapr_pd_8day = yeartemp_sepapr_pd.resample('8D').mean()
    # Calculate phenology metrics
    chl_median = np.nanmedian(yeartemp_sepapr_pd_8day.values)
    # Check which weeks are above 5% median
    chl_weeksabovemedian5 = yeartemp_sepapr_pd_8day > chl_median*1.05
    # Calculate metrics
    b_init_temp = np.argmax(check_for_bloominit(chl_weeksabovemedian5))
    b_term_temp = len(chl_weeksabovemedian5) - np.argmax(check_for_bloominit(chl_weeksabovemedian5[::-1])) - 1
    # Average for 8 days
    if i == 1998:
        wedn_b_init = b_init_temp
        wedn_b_term = b_term_temp
        
    else:
        wedn_b_init = np.hstack((wedn_b_init, b_init_temp))    
        wedn_b_term = np.hstack((wedn_b_term, b_term_temp))    
#%% Create phenology metrics arrays (only DRA, BRS and GES)
binit_19982021 = np.hstack((dra_b_init[:-1], brs_b_init[:-1], ges_b_init[:-1]))
bterm_19982021 = np.hstack((dra_b_term[:-1], brs_b_term[:-1], ges_b_term[:-1]))
#%% Sea Ice
seaiceretreat_19982021 = seaiceretreat[18:]
seaiceadvance_19982021 = seaiceadvance[18:]
#%% Plot
# Sea Ice retreat vs Bloom Init
# DRA
plt.scatter(seaiceretreat_19982021, dra_b_init[:-1], c='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaiceretreat_19982021, dra_b_init[:-1])
plt.plot(seaiceretreat_19982021, seaiceretreat_19982021*slope + intercept, c='k')
# BRS
plt.scatter(seaiceretreat_19982021, brs_b_init[:-1], c='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaiceretreat_19982021, brs_b_init[:-1])
plt.plot(seaiceretreat_19982021, seaiceretreat_19982021*slope + intercept, c='k')
# GES
plt.scatter(seaiceretreat_19982021, ges_b_init[:-1], c='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaiceretreat_19982021, ges_b_init[:-1])
plt.plot(seaiceretreat_19982021, seaiceretreat_19982021*slope + intercept, c='k')
# Sea Ice Advance vs Bloom Term
# DRA
plt.scatter(seaiceadvance_19982021, dra_b_term[:-1], c='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaiceadvance_19982021, dra_b_term[:-1])
plt.plot(seaiceadvance_19982021, seaiceadvance_19982021*slope + intercept, c='k')
# BRS
plt.scatter(seaiceadvance_19982021, brs_b_term[:-1], c='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaiceadvance_19982021, brs_b_term[:-1])
plt.plot(seaiceadvance_19982021, seaiceadvance_19982021*slope + intercept, c='k')
# GES
plt.scatter(seaiceadvance_19982021, ges_b_term[:-1], c='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaiceadvance_19982021, ges_b_term[:-1])
plt.plot(seaiceadvance_19982021, seaiceadvance_19982021*slope + intercept, c='k')

















#%%
plt.scatter(seaicetiming_years[seaiceretreat>200], seaiceretreat[seaiceretreat>200], c='k')
plt.scatter(seaicetiming_years[seaiceretreat<200], seaiceretreat[seaiceretreat<200], c='w', edgecolor='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years[seaiceretreat>200], seaiceretreat[seaiceretreat>200])
plt.plot(seaicetiming_years, seaicetiming_years*slope + intercept, c='k')
plt.xlabel('Years', fontsize=12)
plt.ylabel('Sea Ice Retreat date (Day of Year)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\SupFig5_A.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
plt.scatter(seaicetiming_years, seaiceadvance, c='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceadvance)
plt.plot(seaicetiming_years, seaicetiming_years*slope + intercept, c='k')
plt.xlabel('Years', fontsize=12)
plt.ylabel('Sea Ice Advance date (Day of Year)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\SupFig5_B.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
plt.scatter(seaicetiming_years, seaiceduration, c='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceduration)
plt.plot(seaicetiming_years, seaicetiming_years*slope + intercept, c='k')
plt.xlabel('Years', fontsize=12)
plt.ylabel('Sea Ice Duration (Days)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\SupFig5_C.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%



