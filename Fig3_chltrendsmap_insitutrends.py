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
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\oc4so-chl\\')
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\cciv6data\\')
### Load data 1998-2020
fh = np.load('chloc4so_19972022.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
chl = fh['chl_oc4so']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
chl[chl > 100] = 100
### Calculate monthly means
for i in np.arange(1998, 2023):
    yeartemp_sep = chl[:,:, (time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = chl[:,:, (time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = chl[:,:, (time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = chl[:,:, (time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = chl[:,:, (time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl[:,:, (time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = chl[:,:, (time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = chl[:,:, (time_date_years == i) & (time_date_months == 4)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr)),2)
    if i == 1998:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
#%% Calculate linear trend for each pixel
yearchar = np.arange(1998, 2023)
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        yearchar = np.arange(1998, 2023)
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if np.isnan(chl_summerpixel).any():
            continue
        # remove nans if there exist any
        #if len(np.isnan(chl_summerpixel)) > 0:
        #    nans_chl_summerpixel = np.isnan(chl_summerpixel)
        #    chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
        #    yearchar = yearchar[~nans_chl_summerpixel]
        # calculate linear regression
        slope, _, rvalue, pvalue, _ = stats.linregress(yearchar, chl_summerpixel)
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020[i,j] = slope
        if pvalue < 0.05:
            chl_summertrend19992020_significant[i,j] = slope
#%%
xx, yy = np.meshgrid(lon, lat)
xx = xx[~np.isnan(chl_summertrend19992020_significant)]
yy = yy[~np.isnan(chl_summertrend19992020_significant)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-.1, vmax=.1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.scatter(xx, yy, 2, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none')
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
#map.scatter(xx, yy, 1, transform=ccrs.PlateCarree(), c='w', marker='^', edgecolors='none')
plt.pcolor(lon, lat, chl_summertrend19992020_significant, transform=ccrs.PlateCarree(), hatch='///////', alpha=0.)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Change in Chl-$\it{a}$ (mgm$^{-3}$/year)', fontsize=12)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels = [0.9, 1.1], colors='k',
#            transform=ccrs.PlateCarree())
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\Fig3A_test.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Gather in-situ data and plot them in same plot
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-oscar-sizeclassespigments-2023\\resources\\')
matchups_fullhplc = pd.read_csv('hplc_fulldataframe_10May2023.csv', sep=',', index_col=0)
matchups_lat = matchups_fullhplc['Latitude'].values
matchups_lon = matchups_fullhplc['Longitude'].values
matchups_chl = matchups_fullhplc['chla'].values
matchups_fullhplc.index = pd.to_datetime(matchups_fullhplc.index)
#%%
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('antarcticpeninsula_newclusters_seaicebelow15.npz',allow_pickle = True)
clusters = fh['clusters']
lat_clusters = fh['lat']
lon_clusters = fh['lon']
#%% Find clusters each insitu data belongs to
for i in range(0, len(matchups_lon)):
    # Find which pixel
    matchups_lat_closest = np.where(lat_clusters == min(lat_clusters, key=lambda x:abs(x-matchups_lat[i])))[0][0]
    matchups_lon_closest = np.where(lon_clusters == min(lon_clusters, key=lambda x:abs(x-matchups_lon[i])))[0][0]
    # Find which cluster each point belongs to
    cluster_temp = clusters[matchups_lat_closest, matchups_lon_closest]
    if i == 0:
        matchups_cluster = cluster_temp
    else:
        matchups_cluster = np.hstack((matchups_cluster, cluster_temp))    
#%% Separate by cluster
matchups_BRA = matchups_fullhplc[matchups_cluster == 4]
matchups_GES = matchups_fullhplc[matchups_cluster == 2]
#%% Try averaging each September-April
for i in np.arange(1998, 2023):
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
    if i == 1998:
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
    if i == 1998:
        matchups_GES_chlnumber = matchups_GES_chlnumber_temp
        matchups_GES_chl = matchups_GES_chltemp
        matchups_GES_chl_std = matchups_GES_chltemp_std

    else:
        matchups_GES_chlnumber = np.hstack((matchups_GES_chlnumber, matchups_GES_chlnumber_temp))
        matchups_GES_chl = np.hstack((matchups_GES_chl, matchups_GES_chltemp))     
        matchups_GES_chl_std = np.hstack((matchups_GES_chl_std, matchups_GES_chltemp_std))        
#%% GES September to April # Significant
plt.plot(np.arange(1998, 2023), matchups_GES_chl)
plt.plot(np.arange(1998, 2023), matchups_GES_chlnumber)
# Remove years with less than 10 data points
#matchups_GES_chl[matchups_GES_chlnumber<10] = np.nan
good_years_GES = ~np.logical_or(np.isnan(np.arange(1998, 2023)), np.isnan(matchups_GES_chl))
stats.spearmanr(np.arange(1998, 2023)[good_years_GES], matchups_GES_chl[good_years_GES])
#%% BRA September to April # Not Significant
plt.plot(np.arange(1998, 2023), matchups_BRA_chl)
plt.plot(np.arange(1998, 2023), matchups_BRA_chlnumber)
#matchups_BRA_chl[matchups_BRA_chlnumber<10] = np.nan
good_years_BRA = ~np.logical_or(np.isnan(np.arange(1998, 2023)), np.isnan(matchups_BRA_chl))
stats.spearmanr(np.arange(1998, 2023)[good_years_BRA], matchups_BRA_chl[good_years_BRA])
#%%
plt.scatter(np.arange(1998, 2023), matchups_BRA_chl, 100, c='#6a984e', linewidth=0.5, label='In-situ BRA', edgecolor=[0,0,0,0.8], marker='s')
#plt.fill_between(np.arange(1992, 2023), matchups_BRA_chl, matchups_BRA_chl+matchups_BRA_chl_std, facecolor='#6a984e', alpha=0.3)
#plt.fill_between(np.arange(1992, 2023), matchups_BRA_chl, matchups_BRA_chl-matchups_BRA_chl_std, facecolor='#6a984e', alpha=0.3)
plt.scatter(np.arange(1998, 2023), matchups_GES_chl, 100, c='#2c4ea3', linewidth=0.5, label='In-situ GES', edgecolor=[0,0,0,0.8])
#plt.fill_between(np.arange(1992, 2023), matchups_GES_chl, matchups_GES_chl+matchups_GES_chl_std, facecolor='#29335d', alpha=0.3)
#plt.fill_between(np.arange(1992, 2023), matchups_GES_chl, matchups_GES_chl-matchups_GES_chl_std, facecolor='#29335d', alpha=0.3)
slope_BRA, intercept_BRA, rval_BRA, pval_BRA, __ = stats.linregress(np.arange(1998, 2023)[good_years_BRA], matchups_BRA_chl[good_years_BRA])
slope_GES, intercept_GES, rval_GES, pval_GES, __ = stats.linregress(np.arange(1998, 2023)[good_years_GES], matchups_GES_chl[good_years_GES])
plt.plot(np.arange(1998, 2023), np.arange(1998, 2023)*slope_BRA+intercept_BRA, color='#6a984e', linestyle='--')
plt.plot(np.arange(1998, 2023), np.arange(1998, 2023)*slope_GES+intercept_GES, color='#2c4ea3', linestyle='--')
plt.xlim(1997.5,2020.5)
plt.ylim(0, 8)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.xlabel('Years', fontsize=14)
plt.ylabel('Chl-$\it{a}$ (mgm$^{-3}$)', fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\chl_trends_insitu.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% Bar plot
bransfield_n = [16, 40, 35, 0, 418, 389, 92, 5]
plt.bar(x = np.arange(1, 9), height=bransfield_n, color='#6a984e')
plt.xticks(ticks = np.arange(1, 9), labels=['Sep', 'Oct', ' Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'], fontsize=14)
plt.ylabel('In-situ Samples', fontsize=14)
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\chl_samplesinsitu_bransfield.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
gerlache_n = [6, 59, 257, 307, 574, 296, 192, 18]
plt.bar(x = np.arange(1, 9), height=gerlache_n, color='#2c4ea3')
plt.xticks(ticks = np.arange(1, 9), labels=['Sep', 'Oct', ' Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'], fontsize=14)
plt.ylabel('In-situ Samples', fontsize=14)
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\chl_samplesinsitu_gerlache.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
# BRA
# Sep 16
# Oct 40
# Nov 25
# Dec 0
# Jan 418
# Feb 389
# Mar 92
# Apr 5 

# GES
# Sep 6
# Oct 59
# Nov 257
# Dec 307
# Jan 574
# Feb 296
# Mar 192
# Apr 18





















#%%
