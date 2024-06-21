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
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\seaice_palmer\\palmer_updated\\')
seaicetiming = pd.read_csv('151_ltr3_indices_1979_2023.csv', sep=';')
seaicetiming_years = np.arange(1980, 2023)
seaiceadvance = seaicetiming['Pori_adv'].values[:-1]
seaiceretreat = seaicetiming['Pori_ret'].values[:-1]
seaicedur = seaicetiming['Pori_dur'].values[:-1]
seaiceadvance = np.delete(seaiceadvance, 10)
seaiceretreat = np.delete(seaiceretreat, 10)
seaicetiming_years = np.delete(seaicetiming_years, 10)
seaiceadvance = seaiceadvance+365
# full timeseries
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceadvance)
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceretreat)
# 1998-2022
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years[17:], seaiceadvance[17:])
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years[17:], seaiceretreat[17:])
#%% Plot
#plt.fill_between(x=seaicetiming_years, y1=seaiceretreat, y2=seaiceadvance, facecolor='grey', alpha=0.1)
plt.scatter(seaicetiming_years, seaiceretreat, c='#d2dfe4', s=65, edgecolor='k')
plt.scatter(seaicetiming_years, seaiceadvance, c='#007d71', s=65, edgecolor='k')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceadvance)
plt.plot(seaicetiming_years, seaicetiming_years*slope + intercept, c='k', linewidth=2, alpha=0.5)
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years[17:], seaiceadvance[17:])
plt.plot(seaicetiming_years[17:], seaicetiming_years[17:]*slope + intercept, c='k', linewidth=2, alpha=0.5, linestyle='--')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years, seaiceretreat)
plt.plot(seaicetiming_years, seaicetiming_years*slope + intercept, c='k', linewidth=2, alpha=0.5, linestyle='-')
slope, intercept, rvalue, pvalue , _ = stats.linregress(seaicetiming_years[17:], seaiceretreat[17:])
plt.plot(seaicetiming_years[17:], seaicetiming_years[17:]*slope + intercept, c='k', linewidth=2, alpha=0.5, linestyle='--')

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



