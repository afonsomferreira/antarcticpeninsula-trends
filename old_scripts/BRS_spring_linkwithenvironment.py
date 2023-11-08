# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:00:16 2022

@author: afons
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
## Load clusters
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('antarcticpeninsula_newclusters_seaicebelow15.npz',allow_pickle = True)
clusters = fh['clusters']
#%% Plot Chl-a Monthly vs. other factors
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
## Load Chl-a
fh = np.load('chloc4so_19972021_10km.npz', allow_pickle=True)
lat = fh['lat'][100:]
lon = fh['lon'][30:250]
chl = fh['chl'][100:, 30:250, :]
time_date = fh['time_date']
chl[chl > 50] = 50
chl_brs = chl[clusters == 4,:]
chl_brs = np.nanmean(chl_brs,0)
chl_brs = np.where(chl_brs > np.nanmedian(chl_brs)-np.nanstd(chl_brs)*3, chl_brs, np.nan)
chl_brs = np.where(chl_brs < np.nanmedian(chl_brs)+np.nanstd(chl_brs)*3, chl_brs, np.nan)
chl_brs_df = pd.Series(data=chl_brs, index=time_date)
chl_brs_df_monthly = chl_brs_df.resample('M').mean()
chl_brs_monthly = chl_brs_df_monthly.values
chl_brs_monthly_timedate = chl_brs_df_monthly.index
## Load SST
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia')
fh = np.load('sst_19972021_10km.npz', allow_pickle=True)
lat_sst = fh['lat'][100:]
lon_sst = fh['lon'][30:250]
sst = fh['sst'][100:, 30:250, :]
time_date_sst = fh['time_date']
sst_brs = sst[clusters == 4,:]
sst_brs = np.nanmean(sst_brs,0)
sst_brs_df = pd.Series(data=sst_brs, index=time_date_sst)
sst_brs_df_monthly = sst_brs_df.resample('M').mean()
sst_brs_monthly = sst_brs_df_monthly.values
sst_brs_monthly_timedate = sst_brs_df_monthly.index
## Load Sea Ice
fh = np.load('seaice_19972021_10km.npz', allow_pickle=True)
lat_seaice = fh['lat'][100:]
lon_seaice = fh['lon'][30:250]
seaice = fh['seaice'][100:, 30:250, :]
seaice = seaice*100
time_date_seaice = fh['time_date']
seaice_brs = seaice[clusters == 4,:]
seaice_brs = np.nanmean(seaice_brs,0)
seaice_brs_df = pd.Series(data=seaice_brs, index=time_date_seaice)
seaice_brs_df_monthly = seaice_brs_df.resample('M').mean()
seaice_brs_monthly = seaice_brs_df_monthly.values
seaice_brs_monthly_timedate = seaice_brs_df_monthly.index
## Load PAR
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\par\\')
fh = np.load('par_19972022_10km.npz', allow_pickle=True)
lat_par = fh['lat'][100:]
lon_par = fh['lon'][30:250]
par = fh['par'][100:, 30:250, :]
time_date_par = fh['time_date']
par_brs = par[clusters == 4,:]
par_brs = np.nanmean(par_brs,0)
par_brs_df = pd.Series(data=par_brs, index=time_date_par)
par_brs_df_monthly = par_brs_df.resample('M').mean()
par_brs_monthly = par_brs_df_monthly.values
par_brs_monthly_timedate = par_brs_df_monthly.index
## Load MEI
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\elnino\\')
mei_pd = pd.read_csv('meiv2.csv', sep=';')
mei_monthly = mei_pd['MEI2'][12:].values
## Load SAM
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sam\\')
sam_pd = pd.read_csv('sam.csv', sep=';')
sam_monthly = sam_pd['SAM'][228:].values
## Load Sunspots Number
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sunspots\\')
sunspots_pd = pd.read_csv('SN_d_tot_V2.0.csv', sep=';', header=None)
sunspots_pd.columns = ['year', 'month', 'day', 'decimal year', 'SNvalue' , 'SNerror', 'Nb', 'observations']
sunspots = sunspots_pd['SNvalue'].values
sunspots_year = sunspots_pd['year'].values
sunspots_month = sunspots_pd['month'].values
sunspots_day = sunspots_pd['day'].values
time_date_sunspots = np.empty_like(sunspots, dtype=object)
for i in range(0, len(time_date_sunspots)):
    time_date_sunspots[i] = datetime.datetime(year=sunspots_year[i],
                                                  month=sunspots_month[i],
                                                  day=sunspots_day[i])
    
sunspots = sunspots.astype(float)
sunspots[sunspots==-1] = np.nan
sunspots_df = pd.Series(data=sunspots, index=time_date_sunspots)
sunspots_df_monthly = sunspots_df.resample('M').mean()
sunspots_monthly = sunspots_df_monthly.values[2160:2448]
sunspots_monthly_timedate = sunspots_df_monthly.index[2160:2448]
# Sea Ice duration and timings from PALMER LTER
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaicetimings')
ice_timing = pd.read_csv('ice_timing.csv', sep=';')
ice_timing_years = ice_timing['Ice Year'].values[18:]
seaice_advanceday = ice_timing['Pori adv'].values[18:]
seaice_retreatday = ice_timing['Pori ret'].values[18:]
seaice_duration = ice_timing['Pori dur'].values[18:]
# Sea Ice Extent from PALMER LTER
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\seaice_palmer\\palmer_seaiceextent')
seaice_extent_df = pd.read_csv('seaiceextent.csv', sep=';')
seaice_extent_years = seaice_extent_df['Year'].values[18:]
seaice_extent = seaice_extent_df['Ori_Ext'].values[18:]
#%% Calculate March Chl average vs. factors
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
##  Chl-a
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(chl_brs[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        chl_mar = yeartemp_mar
    else:
        chl_mar = np.hstack((chl_mar, yeartemp_mar))
##  SST
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(sst_brs[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        sst_mar = yeartemp_mar
    else:
        sst_mar = np.hstack((sst_mar, yeartemp_mar))
## Sea Ice
time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(seaice_brs[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        seaice_mar = yeartemp_mar
    else:
        seaice_mar = np.hstack((seaice_mar, yeartemp_mar))
## PAR
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(par_brs[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        par_mar = yeartemp_mar
    else:
        par_mar = np.hstack((par_mar, yeartemp_mar))
## MEI
mei = mei_pd['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(mei[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        mei_mar = yeartemp_mar
    else:
        mei_mar = np.hstack((mei_mar, yeartemp_mar))
## MEI Lag 3
mei_3lag = mei_pd.shift(3)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(mei_3lag[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        mei_lag3_mar = yeartemp_mar
    else:
        mei_lag3_mar = np.hstack((mei_lag3_mar, yeartemp_mar))
## MEI Lag 6
mei_6lag = mei_pd.shift(6)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(mei_6lag[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        mei_lag6_mar = yeartemp_mar
    else:
        mei_lag6_mar = np.hstack((mei_lag6_mar, yeartemp_mar))
## SAM
sam = sam_pd['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(sam[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        sam_mar = yeartemp_mar
    else:
        sam_mar = np.hstack((sam_mar, yeartemp_mar))
## SAM Lag 3
sam_lag3 = sam_pd.shift(3)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(sam_lag3[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        sam_lag3_mar = yeartemp_mar
    else:
        sam_lag3_mar = np.hstack((sam_lag3_mar, yeartemp_mar))
## SAM Lag 6
sam_lag6 = sam_pd.shift(6)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(sam_lag6[(time_date_years == i) & (time_date_months == 3)])
    if i == 1998:
        sam_lag6_mar = yeartemp_mar
    else:
        sam_lag6_mar = np.hstack((sam_lag6_mar, yeartemp_mar))
## Sunspots
for i in np.arange(1998, 2022):
    yeartemp_mar = np.nanmean(sunspots[(sunspots_year == i) & (sunspots_month == 3)])
    if i == 1998:
        sunspots_mar = yeartemp_mar
    else:
        sunspots_mar = np.hstack((sunspots_mar, yeartemp_mar))
#%% Chl-a vs SST (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sst_mar, c='r', label='SST')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sst_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, sst_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sst_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], sst_mar[~np.isnan(chl_mar)])
#%% Chl-a vs Sea Ice (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_mar, c='r', label='Sea Ice')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaice_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, seaice_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaice_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], seaice_mar[~np.isnan(chl_mar)])
#%% Chl-a vs PAR (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, par_mar, c='r', label='PAR')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_par_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, par_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_par_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], par_mar[~np.isnan(chl_mar)])
#%% Chl-a vs SAM (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_mar, c='r', label='SAM')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sam_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, sam_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sam_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], sam_mar[~np.isnan(chl_mar)])
#%% Chl-a vs SAM Lag 3 (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag3_mar, c='r', label='SAM LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sam_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, sam_lag3_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sam_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], sam_lag3_mar[~np.isnan(chl_mar)])
#%% Chl-a vs SAM Lag 6 (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag6_mar, c='r', label='SAM LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sam_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, sam_lag6_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sam_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], sam_lag6_mar[~np.isnan(chl_mar)])
#%% Chl-a vs MEI (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_mar, c='r', label='MEI')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_mei_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, mei_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_mei_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], mei_mar[~np.isnan(chl_mar)])
#%% Chl-a vs MEI Lag 3 (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag3_mar, c='r', label='MEI LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_mei_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, mei_lag3_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_mei_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], mei_lag3_mar[~np.isnan(chl_mar)])
#%% Chl-a vs MEI Lag 6 (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag6_mar, c='r', label='MEI LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_mei_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, mei_lag6_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_mei_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], mei_lag6_mar[~np.isnan(chl_mar)])
#%% Chl-a vs Sea Ice Advance (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_advanceday, c='r', label='Sea Ice Adv')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaiceadv_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, seaice_advanceday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaiceadv_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], seaice_advanceday[~np.isnan(chl_mar)])
#%% Chl-a vs Sea Ice Retreat (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_retreatday, c='r', label='Sea Ice Ret')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaiceret_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, seaice_retreatday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaiceret_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], seaice_retreatday[~np.isnan(chl_mar)])
#%% Chl-a vs Sea Ice Duration (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_duration, c='r', label='Sea Ice Dur')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaicedur_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, seaice_duration)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaicedur_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], seaice_duration[~np.isnan(chl_mar)])
#%% Chl-a vs Sea Ice Extent (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_extent, c='r', label='Sea Ice Ext')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaiceextent_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, seaice_extent)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_seaiceextent_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], seaice_extent[~np.isnan(chl_mar)])
#%% Chl-a vs Sunspots (March)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_mar, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sunspots_mar, c='r', label='Sunspots')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sunspots_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_mar, sunspots_mar)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_mar\\chla_sunspots_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_mar[~np.isnan(chl_mar)], sunspots_mar[~np.isnan(chl_mar)])








































































#%% September-October
##  Chl-a
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
for i in np.arange(1998, 2022):
#    yeartemp_aug = chl_brs[(time_date_years == i-1) & (time_date_months == 8)]
    yeartemp_sep = chl_brs[(time_date_years == i-1) & (time_date_months == 9)]
#    yeartemp_oct = chl_brs[(time_date_years == i-1) & (time_date_months == 10)]
    nonnantemp_augoct = np.count_nonzero(~np.isnan(yeartemp_sep))
    yeartemp_sep = np.nanmean((yeartemp_sep))
    if i == 1998:
        chl_sep = yeartemp_sep
        chl_augoct_avail = nonnantemp_augoct
    else:
        chl_sep = np.hstack((chl_sep, yeartemp_sep))
        chl_augoct_avail = np.hstack((chl_augoct_avail, nonnantemp_augoct))

##  SST
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(sst_brs[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        sst_sep = yeartemp_sep
    else:
        sst_sep = np.hstack((sst_sep, yeartemp_sep))
## Sea Ice
time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(seaice_brs[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        seaice_sep = yeartemp_sep
    else:
        seaice_sep = np.hstack((seaice_sep, yeartemp_sep))
## PAR
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(par_brs[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        par_sep = yeartemp_sep
    else:
        par_sep = np.hstack((par_sep, yeartemp_sep))



























#%% Calculate September Chl average vs. factors
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
##  Chl-a
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(chl_brs[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        chl_sep = yeartemp_sep
    else:
        chl_sep = np.hstack((chl_sep, yeartemp_sep))
##  SST
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(sst_brs[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        sst_sep = yeartemp_sep
    else:
        sst_sep = np.hstack((sst_sep, yeartemp_sep))
## Sea Ice
time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(seaice_brs[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        seaice_sep = yeartemp_sep
    else:
        seaice_sep = np.hstack((seaice_sep, yeartemp_sep))
## PAR
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(par_brs[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        par_sep = yeartemp_sep
    else:
        par_sep = np.hstack((par_sep, yeartemp_sep))
## MEI
mei = mei_pd['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(mei[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        mei_sep = yeartemp_sep
    else:
        mei_sep = np.hstack((mei_sep, yeartemp_sep))
## MEI Lag 3
mei_3lag = mei_pd.shift(3)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(mei_3lag[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        mei_lag3_sep = yeartemp_sep
    else:
        mei_lag3_sep = np.hstack((mei_lag3_sep, yeartemp_sep))
## MEI Lag 6
mei_6lag = mei_pd.shift(6)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(mei_6lag[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        mei_lag6_sep = yeartemp_sep
    else:
        mei_lag6_sep = np.hstack((mei_lag6_sep, yeartemp_sep))
## SAM
sam = sam_pd['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(sam[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        sam_sep = yeartemp_sep
    else:
        sam_sep = np.hstack((sam_sep, yeartemp_sep))
## SAM Lag 3
sam_lag3 = sam_pd.shift(3)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(sam_lag3[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        sam_lag3_sep = yeartemp_sep
    else:
        sam_lag3_sep = np.hstack((sam_lag3_sep, yeartemp_sep))
## SAM Lag 6
sam_lag6 = sam_pd.shift(6)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(sam_lag6[(time_date_years == i-1) & (time_date_months == 9)])
    if i == 1998:
        sam_lag6_sep = yeartemp_sep
    else:
        sam_lag6_sep = np.hstack((sam_lag6_sep, yeartemp_sep))
## Sunspots
for i in np.arange(1998, 2022):
    yeartemp_sep = np.nanmean(sunspots[(sunspots_year == i-1) & (sunspots_month == 9)])
    if i == 1998:
        sunspots_sep = yeartemp_sep
    else:
        sunspots_sep = np.hstack((sunspots_sep, yeartemp_sep))
#%% Chl-a vs SST (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sst_sep, c='r', label='SST')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sst_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, sst_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sst_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], sst_sep[~np.isnan(chl_sep)])
#%% Chl-a vs Sea Ice (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_sep, c='r', label='Sea Ice')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaice_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, seaice_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaice_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], seaice_sep[~np.isnan(chl_sep)])
#%% Chl-a vs PAR (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, par_sep, c='r', label='PAR')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_par_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, par_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_par_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(par_sep)], par_sep[~np.isnan(par_sep)])
#%% Chl-a vs SAM (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_sep, c='r', label='SAM')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sam_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, sam_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sam_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], sam_sep[~np.isnan(chl_sep)])
#%% Chl-a vs SAM Lag 3 (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag3_sep, c='r', label='SAM LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sam_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, sam_lag3_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sam_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], sam_lag3_sep[~np.isnan(chl_sep)])
#%% Chl-a vs SAM Lag 6 (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag6_sep, c='r', label='SAM LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sam_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, sam_lag6_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sam_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], sam_lag6_sep[~np.isnan(chl_sep)])
#%% Chl-a vs MEI (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_sep, c='r', label='MEI')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_mei_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, mei_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_mei_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], mei_sep[~np.isnan(chl_sep)])
#%% Chl-a vs MEI Lag 3 (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag3_sep, c='r', label='MEI LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_mei_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, mei_lag3_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_mei_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], mei_lag3_sep[~np.isnan(chl_sep)])
#%% Chl-a vs MEI Lag 6 (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag6_sep, c='r', label='MEI LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_mei_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, mei_lag6_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_mei_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], mei_lag6_sep[~np.isnan(chl_sep)])
#%% Chl-a vs Sea Ice Advance (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_advanceday, c='r', label='Sea Ice Adv')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaiceadv_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, seaice_advanceday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaiceadv_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], seaice_advanceday[~np.isnan(chl_sep)])
#%% Chl-a vs Sea Ice Retreat (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_retreatday, c='r', label='Sea Ice Ret')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaiceret_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, seaice_retreatday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaiceret_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], seaice_retreatday[~np.isnan(chl_sep)])
#%% Chl-a vs Sea Ice Duration (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_duration, c='r', label='Sea Ice Dur')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaicedur_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, seaice_duration)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaicedur_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], seaice_duration[~np.isnan(chl_sep)])
#%% Chl-a vs Sea Ice Extent (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_extent, c='r', label='Sea Ice Ext')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaiceextent_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, seaice_extent)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_seaiceextent_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], seaice_extent[~np.isnan(chl_sep)])
#%% Chl-a vs Sunspots (September)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_sep, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sunspots_sep, c='r', label='Sunspots')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sunspots_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_sep, sunspots_sep)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_sep\\chla_sunspots_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_sep[~np.isnan(chl_sep)], sunspots_sep[~np.isnan(chl_sep)])
#%% Calculate October Chl average vs. factors
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
##  Chl-a
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(chl_brs[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        chl_oct = yeartemp_oct
    else:
        chl_oct = np.hstack((chl_oct, yeartemp_oct))
##  SST
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(sst_brs[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        sst_oct = yeartemp_oct
    else:
        sst_oct = np.hstack((sst_oct, yeartemp_oct))
## Sea Ice
time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(seaice_brs[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        seaice_oct = yeartemp_oct
    else:
        seaice_oct = np.hstack((seaice_oct, yeartemp_oct))
## PAR
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(par_brs[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        par_oct = yeartemp_oct
    else:
        par_oct = np.hstack((par_oct, yeartemp_oct))
## MEI
mei = mei_pd['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(mei[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        mei_oct = yeartemp_oct
    else:
        mei_oct = np.hstack((mei_oct, yeartemp_oct))
## MEI Lag 3
mei_3lag = mei_pd.shift(3)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(mei_3lag[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        mei_lag3_oct = yeartemp_oct
    else:
        mei_lag3_oct = np.hstack((mei_lag3_oct, yeartemp_oct))
## MEI Lag 6
mei_6lag = mei_pd.shift(6)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(mei_6lag[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        mei_lag6_oct = yeartemp_oct
    else:
        mei_lag6_oct = np.hstack((mei_lag6_oct, yeartemp_oct))
## SAM
sam = sam_pd['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(sam[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        sam_oct = yeartemp_oct
    else:
        sam_oct = np.hstack((sam_oct, yeartemp_oct))
## SAM Lag 3
sam_lag3 = sam_pd.shift(3)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(sam_lag3[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        sam_lag3_oct = yeartemp_oct
    else:
        sam_lag3_oct = np.hstack((sam_lag3_oct, yeartemp_oct))
## SAM Lag 6
sam_lag6 = sam_pd.shift(6)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(sam_lag6[(time_date_years == i-1) & (time_date_months == 10)])
    if i == 1998:
        sam_lag6_oct = yeartemp_oct
    else:
        sam_lag6_oct = np.hstack((sam_lag6_oct, yeartemp_oct))
## Sunspots
for i in np.arange(1998, 2022):
    yeartemp_oct = np.nanmean(sunspots[(sunspots_year == i-1) & (sunspots_month == 10)])
    if i == 1998:
        sunspots_oct = yeartemp_oct
    else:
        sunspots_oct = np.hstack((sunspots_oct, yeartemp_oct))
#%% Chl-a vs SST (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sst_oct, c='r', label='SST')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sst_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, sst_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sst_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], sst_oct[~np.isnan(chl_oct)])
#%% Chl-a vs Sea Ice (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_oct, c='r', label='Sea Ice')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaice_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, seaice_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaice_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], seaice_oct[~np.isnan(chl_oct)])
#%% Chl-a vs PAR (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, par_oct, c='r', label='PAR')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_par_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, par_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_par_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], par_oct[~np.isnan(chl_oct)])
#%% Chl-a vs SAM (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_oct, c='r', label='SAM')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sam_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, sam_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sam_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], sam_oct[~np.isnan(chl_oct)])
#%% Chl-a vs SAM Lag 3 (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag3_oct, c='r', label='SAM LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sam_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, sam_lag3_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sam_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], sam_lag3_oct[~np.isnan(chl_oct)])
#%% Chl-a vs SAM Lag 6 (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag6_oct, c='r', label='SAM LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sam_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, sam_lag6_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sam_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], sam_lag6_oct[~np.isnan(chl_oct)])
#%% Chl-a vs MEI (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_oct, c='r', label='MEI')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_mei_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, mei_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_mei_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], mei_oct[~np.isnan(chl_oct)])
#%% Chl-a vs MEI Lag 3 (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag3_oct, c='r', label='MEI LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_mei_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, mei_lag3_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_mei_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], mei_lag3_oct[~np.isnan(chl_oct)])
#%% Chl-a vs MEI Lag 6 (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag6_oct, c='r', label='MEI LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_mei_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, mei_lag6_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_mei_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], mei_lag6_oct[~np.isnan(chl_oct)])
#%% Chl-a vs Sea Ice Advance (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_advanceday, c='r', label='Sea Ice Adv')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaiceadv_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, seaice_advanceday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaiceadv_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], seaice_advanceday[~np.isnan(chl_oct)])
#%% Chl-a vs Sea Ice Retreat (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_retreatday, c='r', label='Sea Ice Ret')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaiceret_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, seaice_retreatday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaiceret_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], seaice_retreatday[~np.isnan(chl_oct)])
#%% Chl-a vs Sea Ice Duration (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_duration, c='r', label='Sea Ice Dur')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaicedur_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, seaice_duration)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaicedur_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], seaice_duration[~np.isnan(chl_oct)])
#%% Chl-a vs Sea Ice Extent (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_extent, c='r', label='Sea Ice Ext')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaiceextent_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, seaice_extent)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_seaiceextent_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], seaice_extent[~np.isnan(chl_oct)])
#%% Chl-a vs Sunspots (October)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_oct, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sunspots_oct, c='r', label='Sunspots')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sunspots_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_oct, sunspots_oct)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_oct\\chla_sunspots_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_oct[~np.isnan(chl_oct)], sunspots_oct[~np.isnan(chl_oct)])
#%% Calculate November Chl average vs. factors
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
##  Chl-a
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(chl_brs[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        chl_nov = yeartemp_nov
    else:
        chl_nov = np.hstack((chl_nov, yeartemp_nov))
##  SST
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(sst_brs[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        sst_nov = yeartemp_nov
    else:
        sst_nov = np.hstack((sst_nov, yeartemp_nov))
## Sea Ice
time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(seaice_brs[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        seaice_nov = yeartemp_nov
    else:
        seaice_nov = np.hstack((seaice_nov, yeartemp_nov))
## PAR
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(par_brs[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        par_nov = yeartemp_nov
    else:
        par_nov = np.hstack((par_nov, yeartemp_nov))
## MEI
mei = mei_pd['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(mei[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        mei_nov = yeartemp_nov
    else:
        mei_nov = np.hstack((mei_nov, yeartemp_nov))
## MEI Lag 3
mei_3lag = mei_pd.shift(3)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(mei_3lag[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        mei_lag3_nov = yeartemp_nov
    else:
        mei_lag3_nov = np.hstack((mei_lag3_nov, yeartemp_nov))
## MEI Lag 6
mei_6lag = mei_pd.shift(6)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(mei_6lag[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        mei_lag6_nov = yeartemp_nov
    else:
        mei_lag6_nov = np.hstack((mei_lag6_nov, yeartemp_nov))
## SAM
sam = sam_pd['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(sam[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        sam_nov = yeartemp_nov
    else:
        sam_nov = np.hstack((sam_nov, yeartemp_nov))
## SAM Lag 3
sam_lag3 = sam_pd.shift(3)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(sam_lag3[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        sam_lag3_nov = yeartemp_nov
    else:
        sam_lag3_nov = np.hstack((sam_lag3_nov, yeartemp_nov))
## SAM Lag 6
sam_lag6 = sam_pd.shift(6)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(sam_lag6[(time_date_years == i-1) & (time_date_months == 11)])
    if i == 1998:
        sam_lag6_nov = yeartemp_nov
    else:
        sam_lag6_nov = np.hstack((sam_lag6_nov, yeartemp_nov))
## Sunspots
for i in np.arange(1998, 2022):
    yeartemp_nov = np.nanmean(sunspots[(sunspots_year == i-1) & (sunspots_month == 11)])
    if i == 1998:
        sunspots_nov = yeartemp_nov
    else:
        sunspots_nov = np.hstack((sunspots_nov, yeartemp_nov))
#%% Chl-a vs SST (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sst_nov, c='r', label='SST')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sst_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, sst_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sst_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], sst_nov[~np.isnan(chl_nov)])
#%% Chl-a vs Sea Ice (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_nov, c='r', label='Sea Ice')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaice_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, seaice_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaice_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], seaice_nov[~np.isnan(chl_nov)])
#%% Chl-a vs PAR (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, par_nov, c='r', label='PAR')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_par_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, par_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_par_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], par_nov[~np.isnan(chl_nov)])
#%% Chl-a vs SAM (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_nov, c='r', label='SAM')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sam_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, sam_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sam_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], sam_nov[~np.isnan(chl_nov)])
#%% Chl-a vs SAM Lag 3 (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag3_nov, c='r', label='SAM LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sam_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, sam_lag3_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sam_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], sam_lag3_nov[~np.isnan(chl_nov)])
#%% Chl-a vs SAM Lag 6 (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag6_nov, c='r', label='SAM LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sam_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, sam_lag6_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sam_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], sam_lag6_nov[~np.isnan(chl_nov)])
#%% Chl-a vs MEI (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_nov, c='r', label='MEI')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_mei_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, mei_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_mei_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], mei_nov[~np.isnan(chl_nov)])
#%% Chl-a vs MEI Lag 3 (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag3_nov, c='r', label='MEI LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_mei_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, mei_lag3_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_mei_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], mei_lag3_nov[~np.isnan(chl_nov)])
#%% Chl-a vs MEI Lag 6 (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag6_nov, c='r', label='MEI LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_mei_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, mei_lag6_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_mei_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], mei_lag6_nov[~np.isnan(chl_nov)])
#%% Chl-a vs Sea Ice Advance (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_advanceday, c='r', label='Sea Ice Adv')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaiceadv_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, seaice_advanceday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaiceadv_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], seaice_advanceday[~np.isnan(chl_nov)])
#%% Chl-a vs Sea Ice Retreat (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_retreatday, c='r', label='Sea Ice Ret')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaiceret_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, seaice_retreatday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaiceret_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], seaice_retreatday[~np.isnan(chl_nov)])
#%% Chl-a vs Sea Ice Duration (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_duration, c='r', label='Sea Ice Dur')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaicedur_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, seaice_duration)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaicedur_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], seaice_duration[~np.isnan(chl_nov)])
#%% Chl-a vs Sea Ice Extent (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_extent, c='r', label='Sea Ice Ext')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaiceextent_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, seaice_extent)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_seaiceextent_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], seaice_extent[~np.isnan(chl_nov)])
#%% Chl-a vs Sunspots (November)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_nov, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sunspots_nov, c='r', label='Sunspots')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sunspots_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_nov, sunspots_nov)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_nov\\chla_sunspots_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_nov[~np.isnan(chl_nov)], sunspots_nov[~np.isnan(chl_nov)])

#%% Calculate December Chl average vs. factors
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
##  Chl-a
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(chl_brs[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        chl_dec = yeartemp_dec
    else:
        chl_dec = np.hstack((chl_dec, yeartemp_dec))
##  SST
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(sst_brs[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        sst_dec = yeartemp_dec
    else:
        sst_dec = np.hstack((sst_dec, yeartemp_dec))
## Sea Ice
time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(seaice_brs[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        seaice_dec = yeartemp_dec
    else:
        seaice_dec = np.hstack((seaice_dec, yeartemp_dec))
## PAR
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(par_brs[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        par_dec = yeartemp_dec
    else:
        par_dec = np.hstack((par_dec, yeartemp_dec))
## MEI
mei = mei_pd['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(mei[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        mei_dec = yeartemp_dec
    else:
        mei_dec = np.hstack((mei_dec, yeartemp_dec))
## MEI Lag 3
mei_3lag = mei_pd.shift(3)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(mei_3lag[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        mei_lag3_dec = yeartemp_dec
    else:
        mei_lag3_dec = np.hstack((mei_lag3_dec, yeartemp_dec))
## MEI Lag 6
mei_6lag = mei_pd.shift(6)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(mei_6lag[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        mei_lag6_dec = yeartemp_dec
    else:
        mei_lag6_dec = np.hstack((mei_lag6_dec, yeartemp_dec))
## SAM
sam = sam_pd['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(sam[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        sam_dec = yeartemp_dec
    else:
        sam_dec = np.hstack((sam_dec, yeartemp_dec))
## SAM Lag 3
sam_lag3 = sam_pd.shift(3)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(sam_lag3[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        sam_lag3_dec = yeartemp_dec
    else:
        sam_lag3_dec = np.hstack((sam_lag3_dec, yeartemp_dec))
## SAM Lag 6
sam_lag6 = sam_pd.shift(6)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(sam_lag6[(time_date_years == i-1) & (time_date_months == 12)])
    if i == 1998:
        sam_lag6_dec = yeartemp_dec
    else:
        sam_lag6_dec = np.hstack((sam_lag6_dec, yeartemp_dec))
## Sunspots
for i in np.arange(1998, 2022):
    yeartemp_dec = np.nanmean(sunspots[(sunspots_year == i-1) & (sunspots_month == 12)])
    if i == 1998:
        sunspots_dec = yeartemp_dec
    else:
        sunspots_dec = np.hstack((sunspots_dec, yeartemp_dec))
#%% Chl-a vs SST (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sst_dec, c='r', label='SST')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sst_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, sst_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sst_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], sst_dec[~np.isnan(chl_dec)])
#%% Chl-a vs Sea Ice (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_dec, c='r', label='Sea Ice')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaice_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, seaice_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaice_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], seaice_dec[~np.isnan(chl_dec)])
#%% Chl-a vs PAR (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, par_dec, c='r', label='PAR')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_par_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, par_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_par_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], par_dec[~np.isnan(chl_dec)])
#%% Chl-a vs SAM (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_dec, c='r', label='SAM')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sam_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, sam_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sam_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], sam_dec[~np.isnan(chl_dec)])
#%% Chl-a vs SAM Lag 3 (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag3_dec, c='r', label='SAM LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sam_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, sam_lag3_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sam_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], sam_lag3_dec[~np.isnan(chl_dec)])
#%% Chl-a vs SAM Lag 6 (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag6_dec, c='r', label='SAM LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sam_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, sam_lag6_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sam_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], sam_lag6_dec[~np.isnan(chl_dec)])
#%% Chl-a vs MEI (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_dec, c='r', label='MEI')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_mei_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, mei_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_mei_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], mei_dec[~np.isnan(chl_dec)])
#%% Chl-a vs MEI Lag 3 (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag3_dec, c='r', label='MEI LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_mei_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, mei_lag3_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_mei_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], mei_lag3_dec[~np.isnan(chl_dec)])
#%% Chl-a vs MEI Lag 6 (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag6_dec, c='r', label='MEI LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_mei_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, mei_lag6_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_mei_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], mei_lag6_dec[~np.isnan(chl_dec)])
#%% Chl-a vs Sea Ice Advance (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_advanceday, c='r', label='Sea Ice Adv')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaiceadv_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, seaice_advanceday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaiceadv_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], seaice_advanceday[~np.isnan(chl_dec)])
#%% Chl-a vs Sea Ice Retreat (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_retreatday, c='r', label='Sea Ice Ret')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaiceret_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, seaice_retreatday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaiceret_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], seaice_retreatday[~np.isnan(chl_dec)])
#%% Chl-a vs Sea Ice Duration (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_duration, c='r', label='Sea Ice Dur')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaicedur_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, seaice_duration)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaicedur_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], seaice_duration[~np.isnan(chl_dec)])
#%% Chl-a vs Sea Ice Extent (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_extent, c='r', label='Sea Ice Ext')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaiceextent_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, seaice_extent)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_seaiceextent_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], seaice_extent[~np.isnan(chl_dec)])
#%% Chl-a vs Sunspots (December)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_dec, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sunspots_dec, c='r', label='Sunspots')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sunspots_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_dec, sunspots_dec)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_dec\\chla_sunspots_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_dec[~np.isnan(chl_dec)], sunspots_dec[~np.isnan(chl_dec)])
#%% Calculate January Chl average vs. factors
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
##  Chl-a
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(chl_brs[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        chl_jan = yeartemp_jan
    else:
        chl_jan = np.hstack((chl_jan, yeartemp_jan))
##  SST
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(sst_brs[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        sst_jan = yeartemp_jan
    else:
        sst_jan = np.hstack((sst_jan, yeartemp_jan))
## Sea Ice
time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(seaice_brs[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        seaice_jan = yeartemp_jan
    else:
        seaice_jan = np.hstack((seaice_jan, yeartemp_jan))
## PAR
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(par_brs[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        par_jan = yeartemp_jan
    else:
        par_jan = np.hstack((par_jan, yeartemp_jan))
## MEI
mei = mei_pd['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(mei[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        mei_jan = yeartemp_jan
    else:
        mei_jan = np.hstack((mei_jan, yeartemp_jan))
## MEI Lag 3
mei_3lag = mei_pd.shift(3)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(mei_3lag[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        mei_lag3_jan = yeartemp_jan
    else:
        mei_lag3_jan = np.hstack((mei_lag3_jan, yeartemp_jan))
## MEI Lag 6
mei_6lag = mei_pd.shift(6)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(mei_6lag[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        mei_lag6_jan = yeartemp_jan
    else:
        mei_lag6_jan = np.hstack((mei_lag6_jan, yeartemp_jan))
## SAM
sam = sam_pd['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(sam[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        sam_jan = yeartemp_jan
    else:
        sam_jan = np.hstack((sam_jan, yeartemp_jan))
## SAM Lag 3
sam_lag3 = sam_pd.shift(3)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(sam_lag3[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        sam_lag3_jan = yeartemp_jan
    else:
        sam_lag3_jan = np.hstack((sam_lag3_jan, yeartemp_jan))
## SAM Lag 6
sam_lag6 = sam_pd.shift(6)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(sam_lag6[(time_date_years == i) & (time_date_months == 1)])
    if i == 1998:
        sam_lag6_jan = yeartemp_jan
    else:
        sam_lag6_jan = np.hstack((sam_lag6_jan, yeartemp_jan))
## Sunspots
for i in np.arange(1998, 2022):
    yeartemp_jan = np.nanmean(sunspots[(sunspots_year == i) & (sunspots_month == 1)])
    if i == 1998:
        sunspots_jan = yeartemp_jan
    else:
        sunspots_jan = np.hstack((sunspots_jan, yeartemp_jan))
#%% Chl-a vs SST (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sst_jan, c='r', label='SST')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sst_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, sst_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sst_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], sst_jan[~np.isnan(chl_jan)])
#%% Chl-a vs Sea Ice (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_jan, c='r', label='Sea Ice')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaice_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, seaice_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaice_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], seaice_jan[~np.isnan(chl_jan)])
#%% Chl-a vs PAR (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, par_jan, c='r', label='PAR')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_par_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, par_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_par_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], par_jan[~np.isnan(chl_jan)])
#%% Chl-a vs SAM (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_jan, c='r', label='SAM')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sam_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, sam_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sam_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], sam_jan[~np.isnan(chl_jan)])
#%% Chl-a vs SAM Lag 3 (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag3_jan, c='r', label='SAM LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sam_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, sam_lag3_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sam_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], sam_lag3_jan[~np.isnan(chl_jan)])
#%% Chl-a vs SAM Lag 6 (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag6_jan, c='r', label='SAM LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sam_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, sam_lag6_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sam_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], sam_lag6_jan[~np.isnan(chl_jan)])
#%% Chl-a vs MEI (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_jan, c='r', label='MEI')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_mei_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, mei_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_mei_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], mei_jan[~np.isnan(chl_jan)])
#%% Chl-a vs MEI Lag 3 (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag3_jan, c='r', label='MEI LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_mei_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, mei_lag3_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_mei_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], mei_lag3_jan[~np.isnan(chl_jan)])
#%% Chl-a vs MEI Lag 6 (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag6_jan, c='r', label='MEI LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_mei_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, mei_lag6_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_mei_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], mei_lag6_jan[~np.isnan(chl_jan)])
#%% Chl-a vs Sea Ice Advance (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_advanceday, c='r', label='Sea Ice Adv')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaiceadv_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, seaice_advanceday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaiceadv_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], seaice_advanceday[~np.isnan(chl_jan)])
#%% Chl-a vs Sea Ice Retreat (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_retreatday, c='r', label='Sea Ice Ret')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaiceret_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, seaice_retreatday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaiceret_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], seaice_retreatday[~np.isnan(chl_jan)])
#%% Chl-a vs Sea Ice Duration (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_duration, c='r', label='Sea Ice Dur')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaicedur_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, seaice_duration)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaicedur_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], seaice_duration[~np.isnan(chl_jan)])
#%% Chl-a vs Sea Ice Extent (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_extent, c='r', label='Sea Ice Ext')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaiceextent_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, seaice_extent)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_seaiceextent_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], seaice_extent[~np.isnan(chl_jan)])
#%% Chl-a vs Sunspots (January)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_jan, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sunspots_jan, c='r', label='Sunspots')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sunspots_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_jan, sunspots_jan)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_jan\\chla_sunspots_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_jan[~np.isnan(chl_jan)], sunspots_jan[~np.isnan(chl_jan)])
#%% Calculate February Chl average vs. factors
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
##  Chl-a
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(chl_brs[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        chl_feb = yeartemp_feb
    else:
        chl_feb = np.hstack((chl_feb, yeartemp_feb))
##  SST
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(sst_brs[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        sst_feb = yeartemp_feb
    else:
        sst_feb = np.hstack((sst_feb, yeartemp_feb))
## Sea Ice
time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(seaice_brs[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        seaice_feb = yeartemp_feb
    else:
        seaice_feb = np.hstack((seaice_feb, yeartemp_feb))
## PAR
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(par_brs[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        par_feb = yeartemp_feb
    else:
        par_feb = np.hstack((par_feb, yeartemp_feb))
## MEI
mei = mei_pd['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(mei[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        mei_feb = yeartemp_feb
    else:
        mei_feb = np.hstack((mei_feb, yeartemp_feb))
## MEI Lag 3
mei_3lag = mei_pd.shift(3)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(mei_3lag[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        mei_lag3_feb = yeartemp_feb
    else:
        mei_lag3_feb = np.hstack((mei_lag3_feb, yeartemp_feb))
## MEI Lag 6
mei_6lag = mei_pd.shift(6)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(mei_6lag[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        mei_lag6_feb = yeartemp_feb
    else:
        mei_lag6_feb = np.hstack((mei_lag6_feb, yeartemp_feb))
## SAM
sam = sam_pd['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(sam[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        sam_feb = yeartemp_feb
    else:
        sam_feb = np.hstack((sam_feb, yeartemp_feb))
## SAM Lag 3
sam_lag3 = sam_pd.shift(3)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(sam_lag3[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        sam_lag3_feb = yeartemp_feb
    else:
        sam_lag3_feb = np.hstack((sam_lag3_feb, yeartemp_feb))
## SAM Lag 6
sam_lag6 = sam_pd.shift(6)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(sam_lag6[(time_date_years == i) & (time_date_months == 2)])
    if i == 1998:
        sam_lag6_feb = yeartemp_feb
    else:
        sam_lag6_feb = np.hstack((sam_lag6_feb, yeartemp_feb))
## Sunspots
for i in np.arange(1998, 2022):
    yeartemp_feb = np.nanmean(sunspots[(sunspots_year == i) & (sunspots_month == 2)])
    if i == 1998:
        sunspots_feb = yeartemp_feb
    else:
        sunspots_feb = np.hstack((sunspots_feb, yeartemp_feb))
#%% Chl-a vs SST (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sst_feb, c='r', label='SST')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sst_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, sst_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sst_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], sst_feb[~np.isnan(chl_feb)])
#%% Chl-a vs Sea Ice (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_feb, c='r', label='Sea Ice')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaice_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, seaice_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaice_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], seaice_feb[~np.isnan(chl_feb)])
#%% Chl-a vs PAR (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, par_feb, c='r', label='PAR')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_par_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, par_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_par_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], par_feb[~np.isnan(chl_feb)])
#%% Chl-a vs SAM (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_feb, c='r', label='SAM')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sam_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, sam_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sam_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], sam_feb[~np.isnan(chl_feb)])
#%% Chl-a vs SAM Lag 3 (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag3_feb, c='r', label='SAM LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sam_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, sam_lag3_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sam_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], sam_lag3_feb[~np.isnan(chl_feb)])
#%% Chl-a vs SAM Lag 6 (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag6_feb, c='r', label='SAM LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sam_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, sam_lag6_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sam_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], sam_lag6_feb[~np.isnan(chl_feb)])
#%% Chl-a vs MEI (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_feb, c='r', label='MEI')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_mei_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, mei_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_mei_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], mei_feb[~np.isnan(chl_feb)])
#%% Chl-a vs MEI Lag 3 (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag3_feb, c='r', label='MEI LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_mei_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, mei_lag3_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_mei_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], mei_lag3_feb[~np.isnan(chl_feb)])
#%% Chl-a vs MEI Lag 6 (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag6_feb, c='r', label='MEI LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_mei_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, mei_lag6_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_mei_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], mei_lag6_feb[~np.isnan(chl_feb)])
#%% Chl-a vs Sea Ice Advance (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_advanceday, c='r', label='Sea Ice Adv')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaiceadv_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, seaice_advanceday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaiceadv_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], seaice_advanceday[~np.isnan(chl_feb)])
#%% Chl-a vs Sea Ice Retreat (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_retreatday, c='r', label='Sea Ice Ret')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaiceret_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, seaice_retreatday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaiceret_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], seaice_retreatday[~np.isnan(chl_feb)])
#%% Chl-a vs Sea Ice Duration (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_duration, c='r', label='Sea Ice Dur')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaicedur_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, seaice_duration)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaicedur_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], seaice_duration[~np.isnan(chl_feb)])
#%% Chl-a vs Sea Ice Extent (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_extent, c='r', label='Sea Ice Ext')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaiceextent_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, seaice_extent)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_seaiceextent_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], seaice_extent[~np.isnan(chl_feb)])
#%% Chl-a vs Sunspots (February)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_feb, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sunspots_feb, c='r', label='Sunspots')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sunspots_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_feb, sunspots_feb)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_feb\\chla_sunspots_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_feb[~np.isnan(chl_feb)], sunspots_feb[~np.isnan(chl_feb)])

#%% Calculate April Chl average vs. factors
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
##  Chl-a
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(chl_brs[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        chl_apr = yeartemp_apr
    else:
        chl_apr = np.hstack((chl_apr, yeartemp_apr))
##  SST
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(sst_brs[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        sst_apr = yeartemp_apr
    else:
        sst_apr = np.hstack((sst_apr, yeartemp_apr))
## Sea Ice
time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(seaice_brs[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        seaice_apr = yeartemp_apr
    else:
        seaice_apr = np.hstack((seaice_apr, yeartemp_apr))
## PAR
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(par_brs[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        par_apr = yeartemp_apr
    else:
        par_apr = np.hstack((par_apr, yeartemp_apr))
## MEI
mei = mei_pd['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(mei[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        mei_apr = yeartemp_apr
    else:
        mei_apr = np.hstack((mei_apr, yeartemp_apr))
## MEI Lag 3
mei_3lag = mei_pd.shift(3)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(mei_3lag[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        mei_lag3_apr = yeartemp_apr
    else:
        mei_lag3_apr = np.hstack((mei_lag3_apr, yeartemp_apr))
## MEI Lag 6
mei_6lag = mei_pd.shift(6)['MEI2'].values
time_date_years = mei_pd['Year'].values
time_date_months = mei_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(mei_6lag[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        mei_lag6_apr = yeartemp_apr
    else:
        mei_lag6_apr = np.hstack((mei_lag6_apr, yeartemp_apr))
## SAM
sam = sam_pd['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(sam[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        sam_apr = yeartemp_apr
    else:
        sam_apr = np.hstack((sam_apr, yeartemp_apr))
## SAM Lag 3
sam_lag3 = sam_pd.shift(3)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(sam_lag3[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        sam_lag3_apr = yeartemp_apr
    else:
        sam_lag3_apr = np.hstack((sam_lag3_apr, yeartemp_apr))
## SAM Lag 6
sam_lag6 = sam_pd.shift(6)['SAM'].values
time_date_years = sam_pd['Year'].values
time_date_months = sam_pd['Month'].values
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(sam_lag6[(time_date_years == i) & (time_date_months == 4)])
    if i == 1998:
        sam_lag6_apr = yeartemp_apr
    else:
        sam_lag6_apr = np.hstack((sam_lag6_apr, yeartemp_apr))
## Sunspots
for i in np.arange(1998, 2022):
    yeartemp_apr = np.nanmean(sunspots[(sunspots_year == i) & (sunspots_month == 4)])
    if i == 1998:
        sunspots_apr = yeartemp_apr
    else:
        sunspots_apr = np.hstack((sunspots_apr, yeartemp_apr))
#%% Chl-a vs SST (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sst_apr, c='r', label='SST')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sst_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, sst_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SST (°C)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sst_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], sst_apr[~np.isnan(chl_apr)])
#%% Chl-a vs Sea Ice (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_apr, c='r', label='Sea Ice')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaice_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, seaice_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice (%)', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaice_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], seaice_apr[~np.isnan(chl_apr)])
#%% Chl-a vs PAR (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, par_apr, c='r', label='PAR')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_par_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, par_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('PAR', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_par_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], par_apr[~np.isnan(chl_apr)])
#%% Chl-a vs SAM (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_apr, c='r', label='SAM')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sam_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, sam_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sam_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], sam_apr[~np.isnan(chl_apr)])
#%% Chl-a vs SAM Lag 3 (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag3_apr, c='r', label='SAM LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sam_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, sam_lag3_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sam_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], sam_lag3_apr[~np.isnan(chl_apr)])
#%% Chl-a vs SAM Lag 6 (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sam_lag6_apr, c='r', label='SAM LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sam_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, sam_lag6_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('SAM LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sam_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], sam_lag6_apr[~np.isnan(chl_apr)])
#%% Chl-a vs MEI (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_apr, c='r', label='MEI')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_mei_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, mei_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_mei_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], mei_apr[~np.isnan(chl_apr)])
#%% Chl-a vs MEI Lag 3 (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag3_apr, c='r', label='MEI LAG3')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_mei_lag3_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, mei_lag3_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG3', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_mei_lag3_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], mei_lag3_apr[~np.isnan(chl_apr)])
#%% Chl-a vs MEI Lag 6 (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, mei_lag6_apr, c='r', label='MEI LAG6')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_mei_lag6_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, mei_lag6_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('MEI LAG6', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_mei_lag6_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], mei_lag6_apr[~np.isnan(chl_apr)])
#%% Chl-a vs Sea Ice Advance (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_advanceday, c='r', label='Sea Ice Adv')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaiceadv_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, seaice_advanceday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Adv', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaiceadv_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], seaice_advanceday[~np.isnan(chl_apr)])
#%% Chl-a vs Sea Ice Retreat (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_retreatday, c='r', label='Sea Ice Ret')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaiceret_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, seaice_retreatday)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ret', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaiceret_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], seaice_retreatday[~np.isnan(chl_apr)])
#%% Chl-a vs Sea Ice Duration (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_duration, c='r', label='Sea Ice Dur')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaicedur_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, seaice_duration)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Dur', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaicedur_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], seaice_duration[~np.isnan(chl_apr)])
#%% Chl-a vs Sea Ice Extent (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, seaice_extent, c='r', label='Sea Ice Ext')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaiceextent_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, seaice_extent)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sea Ice Ext', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_seaiceextent_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], seaice_extent[~np.isnan(chl_apr)])
#%% Chl-a vs Sunspots (April)
years_chl = np.arange(1998, 2022)
# Plot side by side
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.plot(years_chl, chl_apr, c='g', label='Chla')
axs2 = axs.twinx()
axs2.plot(years_chl, sunspots_apr, c='r', label='Sunspots')
axs.set_xlim(years_chl[0],years_chl[-1])
axs.yaxis.label.set_color('k')
axs.tick_params(axis='y', colors='k')
axs.spines['left'].set_color('k')
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs2.set_ylabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sunspots_line.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot Correlation
fig, axs = plt.subplots(1, 1, figsize=(12,6))
axs.scatter(chl_apr, sunspots_apr)
axs.set_ylabel('Chla (mg/m3)', fontsize=12)
axs.set_xlabel('Sunspots', fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\timeseries_comparisons\\chla_apr\\chla_sunspots_scatter.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
slope, intercept, rval, pval, _ = stats.linregress(chl_apr[~np.isnan(chl_apr)], sunspots_apr[~np.isnan(chl_apr)])











#