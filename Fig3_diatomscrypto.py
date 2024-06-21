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
#%% Load community data
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-oscar-2023\\resources\\')
matchups_fullhplc = pd.read_csv('hplc_fulldataframe_10May2023.csv', sep=',', index_col=0)
matchups_lon = matchups_fullhplc['Longitude']
matchups_lat = matchups_fullhplc['Latitude']
#%%
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_4km.npz',allow_pickle = True)
clusters = fh['clusters']
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\cciv6data\\')
fh = np.load('chloc4so_19972022.npz', allow_pickle=True)
lat_seaice = fh['lat'][144:]
lon_seaice = fh['lon']
#%% Find clusters each insitu data belongs to
for i in range(0, len(matchups_lon)):
    # Find which pixel
    matchups_lat_closest = np.where(lat_seaice == min(lat_seaice, key=lambda x:abs(x-matchups_lat[i])))[0][0]
    matchups_lon_closest = np.where(lon_seaice == min(lon_seaice, key=lambda x:abs(x-matchups_lon[i])))[0][0]
    # Find which cluster each point belongs to
    cluster_temp = clusters[matchups_lat_closest, matchups_lon_closest]
    if i == 0:
        matchups_cluster = cluster_temp
    else:
        matchups_cluster = np.hstack((matchups_cluster, cluster_temp))    
#%% Separate by cluster
matchups_BRA = matchups_fullhplc[matchups_cluster == 4]
matchups_GES = matchups_fullhplc[matchups_cluster == 2]
#%% Extract datetime
datetime_matchups_GES = np.empty_like(matchups_GES.index)
for i in range(0, len(matchups_GES)):
    datetime_matchups_GES[i] = datetime.datetime(year=int(matchups_GES.index[i][:4]),
                                                 month=int(matchups_GES.index[i][5:7]),
                                                 day=int(matchups_GES.index[i][8:10]),
                                                 hour=int(matchups_GES.index[i][11:13]),
                                                 minute=int(matchups_GES.index[i][14:16]),
                                                 second=int(matchups_GES.index[i][17:19]))
matchups_GES.index = datetime_matchups_GES    
#%% Average yearly
matchups_GES_yearly = matchups_GES.resample('Y').mean()
# Calculate proportion between diatoms and cryptophytes
proportion_diatomcrypto = matchups_GES_yearly['Cryptophytes'].values/matchups_GES_yearly['Diatoms'].values 
proportion_diatomcrypto1 = (proportion_diatomcrypto - min(proportion_diatomcrypto)) / ( max(proportion_diatomcrypto) - min(proportion_diatomcrypto) )
proportion_diatomcrypto1
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(matchups_GES_yearly.index.year, matchups_GES_yearly['Diatoms'].values)
ax1.bar(matchups_GES_yearly.index.year, -matchups_GES_yearly['Cryptophytes'].values)
ax2.plot(matchups_GES_yearly.index.year, proportion_diatomcrypto)





plt.bar(matchups_GES_yearly.index.year, matchups_GES_yearly['Diatoms'].values)
plt.bar(matchups_GES_yearly.index.year, matchups_GES_yearly['Cryptophytes'].values)

#%% Clean
#seaiceretreat = seaiceretreat.astype(float)
#seaiceadvance = seaiceadvance.astype(float)
#seaiceretreat[10] = np.nan
#seaiceadvance[10] = np.nan
seaiceadvance = seaiceadvance+365

slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceretreat_without1990)

#%% Plot
#plt.fill_between(x=seaicetiming_years, y1=seaiceretreat, y2=seaiceadvance, facecolor='grey', alpha=0.1)
plt.scatter(seaicetiming_years, seaiceretreat, c='#2D2926FF', s=65, marker='s')
plt.scatter(seaicetiming_years, seaiceadvance, c='#2D2926FF', s=65)
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceadvance)
plt.plot(seaicetiming_years, seaicetiming_years*slope + intercept, c='#E94B3CFF', linewidth=2, alpha=0.5)
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceretreat)
plt.plot(seaicetiming_years, seaicetiming_years*slope + intercept, c='#E94B3CFF', linewidth=2, alpha=0.5)
plt.yticks(ticks=[244, 274, 305, 335, 366, 397, 425, 456, 486, 517, 547, 578],
           labels= ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
           fontsize=14)
plt.xticks(fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\Seaiceretreatadvance_plot.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

#%%














# seaice without 1990
seaicetiming_years
seaiceretreat_without1990 = seaiceretreat
seaiceretreat_without1990 = seaiceretreat_without1990.astype(float)
seaiceretreat_without1990[10] = np.nan
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceretreat_without1990)




plt.scatter(seaicetiming_years, seaiceretreat, c='k', s=50)
plt.scatter(seaicetiming_years[10], seaiceretreat[10], c='w', edgecolor='k', s=50)

plt.plot(seaicetiming_years, seaicetiming_years*slope + intercept, c='k')







#%%
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



