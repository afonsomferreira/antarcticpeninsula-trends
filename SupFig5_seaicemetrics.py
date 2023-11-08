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
#%% Load sea ice duration
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaicetimings\\')
seaicetiming = pd.read_csv('ice_timing.csv', sep=';')
seaicetiming_years = np.arange(1980, 2022)
seaiceduration = seaicetiming['Pori dur'].values
seaiceadvance = seaicetiming['Pori adv'].values
seaiceretreat = seaicetiming['Pori ret'].values
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



