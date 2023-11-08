# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:33:06 2020

@author: Afonso
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from netCDF4 import Dataset
import datetime
import cmocean
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
#%%
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-oscar-2023\\resources\\')
matchups_fullhplc = pd.read_csv('hplc_fulldataframe_10May2023.csv', sep=',', index_col=0)
matchups_lat = matchups_fullhplc['Latitude'].values
matchups_lon = matchups_fullhplc['Longitude'].values
matchups_chl = matchups_fullhplc['chla'].values
matchups_fullhplc.index = pd.to_datetime(matchups_fullhplc.index)
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
#%% Test
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60]) 
f1 = map.pcolormesh(lon_seaice, lat_seaice, clusters[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.Set1)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1,
                    fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=14)
#cbar.set_label('Chl-a Max Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
f1 = map.scatter(matchups_lon[matchups_cluster == 2], matchups_lat[matchups_cluster == 2], s=15, c='k',  transform=ccrs.PlateCarree(),
                 edgecolor='k', linewidth=.3)
plt.tight_layout()
#%% Separate by cluster
matchups_BRA = matchups_fullhplc[matchups_cluster == 4]
matchups_GES = matchups_fullhplc[matchups_cluster == 2]
#%% Try averaging each September-April
for i in np.arange(1992, 2023):
    ## BRA
    matchups_BRA_sep = matchups_BRA[(matchups_BRA.index.year == i-1) & (matchups_BRA.index.month == 9)]
    matchups_BRA_oct = matchups_BRA[(matchups_BRA.index.year == i-1) & (matchups_BRA.index.month == 10)]
    matchups_BRA_nov = matchups_BRA[(matchups_BRA.index.year == i-1) & (matchups_BRA.index.month == 11)]
    matchups_BRA_dec = matchups_BRA[(matchups_BRA.index.year == i-1) & (matchups_BRA.index.month == 12)]
    matchups_BRA_jan = matchups_BRA[(matchups_BRA.index.year == i) & (matchups_BRA.index.month == 1)]
    matchups_BRA_feb = matchups_BRA[(matchups_BRA.index.year == i) & (matchups_BRA.index.month == 2)]
    matchups_BRA_mar = matchups_BRA[(matchups_BRA.index.year == i) & (matchups_BRA.index.month == 3)]
    matchups_BRA_apr = matchups_BRA[(matchups_BRA.index.year == i) & (matchups_BRA.index.month == 4)]
    # Join all
    matchups_BRA_yeartemp = pd.concat([matchups_BRA_sep, matchups_BRA_oct, matchups_BRA_nov, matchups_BRA_dec,
              matchups_BRA_jan, matchups_BRA_feb, matchups_BRA_mar, matchups_BRA_apr])
    # Average chla and count number of data
    matchups_BRA_chltemp = np.nanmean(matchups_BRA_yeartemp['chla'].values)
    matchups_BRA_chltemp_std = np.nanstd(matchups_BRA_yeartemp['chla'].values)
#    matchups_BRA_chlnumber_temp = np.size(matchups_BRA_chltemp)
    if np.isnan(matchups_BRA_chltemp):
        matchups_BRA_chlnumber_temp = 0
    else:
        matchups_BRA_chlnumber_temp = len(matchups_BRA_yeartemp)        
    # Join results
    if i == 1992:
        matchups_BRA_chlnumber = matchups_BRA_chlnumber_temp
        matchups_BRA_chl = matchups_BRA_chltemp
        matchups_BRA_chl_std = matchups_BRA_chltemp_std
    else:
        matchups_BRA_chlnumber = np.hstack((matchups_BRA_chlnumber, matchups_BRA_chlnumber_temp))
        matchups_BRA_chl = np.hstack((matchups_BRA_chl, matchups_BRA_chltemp))  
        matchups_BRA_chl_std = np.hstack((matchups_BRA_chl_std, matchups_BRA_chltemp_std))        
    ## GES
    matchups_GES_sep = matchups_GES[(matchups_GES.index.year == i-1) & (matchups_GES.index.month == 9)]
    matchups_GES_oct = matchups_GES[(matchups_GES.index.year == i-1) & (matchups_GES.index.month == 10)]
    matchups_GES_nov = matchups_GES[(matchups_GES.index.year == i-1) & (matchups_GES.index.month == 11)]
    matchups_GES_dec = matchups_GES[(matchups_GES.index.year == i-1) & (matchups_GES.index.month == 12)]
    matchups_GES_jan = matchups_GES[(matchups_GES.index.year == i) & (matchups_GES.index.month == 1)]
    matchups_GES_feb = matchups_GES[(matchups_GES.index.year == i) & (matchups_GES.index.month == 2)]
    matchups_GES_mar = matchups_GES[(matchups_GES.index.year == i) & (matchups_GES.index.month == 3)]
    matchups_GES_apr = matchups_GES[(matchups_GES.index.year == i) & (matchups_GES.index.month == 4)]
    # Join all
    matchups_GES_yeartemp = pd.concat([matchups_GES_sep, matchups_GES_oct, matchups_GES_nov, matchups_GES_dec,
              matchups_GES_jan, matchups_GES_feb, matchups_GES_mar, matchups_GES_apr])
    # Average chla and count number of data
    matchups_GES_chltemp = np.nanmean(matchups_GES_yeartemp['chla'].values)
    matchups_GES_chltemp_std = np.nanstd(matchups_GES_yeartemp['chla'].values)
#    matchups_GES_chlnumber_temp = np.size(matchups_GES_chltemp)
    if np.isnan(matchups_GES_chltemp):
        matchups_GES_chlnumber_temp = 0
    else:
        matchups_GES_chlnumber_temp = len(matchups_GES_yeartemp)
    # Join results
    if i == 1992:
        matchups_GES_chlnumber = matchups_GES_chlnumber_temp
        matchups_GES_chl = matchups_GES_chltemp
        matchups_GES_chl_std = matchups_GES_chltemp_std

    else:
        matchups_GES_chlnumber = np.hstack((matchups_GES_chlnumber, matchups_GES_chlnumber_temp))
        matchups_GES_chl = np.hstack((matchups_GES_chl, matchups_GES_chltemp))     
        matchups_GES_chl_std = np.hstack((matchups_GES_chl_std, matchups_GES_chltemp_std))        
#%% GES September to April # Significant
plt.plot(np.arange(1992, 2023), matchups_GES_chl)
plt.plot(np.arange(1992, 2023), matchups_GES_chlnumber)
# Remove years with less than 10 data points
#matchups_GES_chl[matchups_GES_chlnumber<10] = np.nan
good_years_GES = ~np.logical_or(np.isnan(np.arange(1992, 2023)), np.isnan(matchups_GES_chl))
stats.spearmanr(np.arange(1992, 2023)[good_years_GES], matchups_GES_chl[good_years_GES])
#%% BRA September to April # Not Significant
plt.plot(np.arange(1992, 2023), matchups_BRA_chl)
plt.plot(np.arange(1992, 2023), matchups_BRA_chlnumber)
#matchups_BRA_chl[matchups_BRA_chlnumber<10] = np.nan
good_years_BRA = ~np.logical_or(np.isnan(np.arange(1992, 2023)), np.isnan(matchups_BRA_chl))
stats.spearmanr(np.arange(1992, 2023)[good_years_BRA], matchups_BRA_chl[good_years_BRA])
#%%
plt.scatter(np.arange(1992, 2023), matchups_BRA_chl, c='#6a984e', linewidth=0.5, label='In-situ BRA', edgecolor=[0,0,0,0.8])
#plt.fill_between(np.arange(1992, 2023), matchups_BRA_chl, matchups_BRA_chl+matchups_BRA_chl_std, facecolor='#6a984e', alpha=0.3)
#plt.fill_between(np.arange(1992, 2023), matchups_BRA_chl, matchups_BRA_chl-matchups_BRA_chl_std, facecolor='#6a984e', alpha=0.3)
plt.scatter(np.arange(1992, 2023), matchups_GES_chl, c='#29335d', linewidth=0.5, label='In-situ GES', edgecolor=[0,0,0,0.8])
#plt.fill_between(np.arange(1992, 2023), matchups_GES_chl, matchups_GES_chl+matchups_GES_chl_std, facecolor='#29335d', alpha=0.3)
#plt.fill_between(np.arange(1992, 2023), matchups_GES_chl, matchups_GES_chl-matchups_GES_chl_std, facecolor='#29335d', alpha=0.3)
slope_BRA, intercept_BRA, rval_BRA, pval_BRA, __ = stats.linregress(np.arange(1992, 2023)[good_years_BRA], matchups_BRA_chl[good_years_BRA])
slope_GES, intercept_GES, rval_GES, pval_GES, __ = stats.linregress(np.arange(1992, 2023)[good_years_GES], matchups_GES_chl[good_years_GES])
plt.plot(np.arange(1992, 2023), np.arange(1992, 2023)*slope_BRA+intercept_BRA, c='#6a984e', linestyle='--')
plt.plot(np.arange(1992, 2023), np.arange(1992, 2023)*slope_GES+intercept_GES, c='#29335d', linestyle='--')
plt.xlim(1991.5,2020.5)
plt.ylim(0, 8)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=0, fontsize=14)





