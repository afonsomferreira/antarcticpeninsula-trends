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
# calculate 1 year moving mean
sam_monthly_movingmean = sam_pd_monthly.rolling(12).mean().values
monthly_dates = sam_pd_monthly.index
sam_pd_monthly_cumsum = sam_pd_monthly.cumsum()
#%%
fig, axs = plt.subplots(1, 1, figsize=(6,4))
lns1 = axs.plot(np.arange(529), sam_pd_monthly.values, 'gray', linewidth=1.5, alpha=0.4, label='SAM')
lns2 = axs.plot(np.arange(529), sam_monthly_movingmean, 'k', linewidth=1.5, alpha=1, label='SAM Mov. Mean')
axs.set_ylim(-3, 3)
axs2 = axs.twinx()
lns3 = axs2.plot(np.arange(529), sam_pd_monthly_cumsum.values, c='#da2b39', linewidth=1.5, alpha=1, label='Cumulative SAM', linestyle='-')
plt.xlim(12,529)
axs2.set_ylim(-50,50)
axs.axhline(c='k', linewidth=1, alpha=1, linestyle='-')
plt.axvline(372,c='k', linewidth=1, alpha=.7, linestyle='--')
#axs.plot(np.arange(529)[336:], np.arange(529)[336:]*slope + intercept, linestyle='--', c='k', label='2007-2022 Trend')
plt.xticks(ticks=[12, 72, 132, 192, 252, 312, 372, 432, 492], labels=['1980', '1985', '1990', '1995', '2000',
                                              '2005', '2010', '2015', '2020'])
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
axs.legend(lns, labs, loc=2, fontsize=12)
axs2.tick_params(axis='y', colors='#da2b39')
#fig.legend(loc="upper left")
#fig.legend(loc="upper right", bbox_to_anchor=(1.1,1.14), bbox_transform=axs.transAxes, ncol=4, fontsize=10)
axs.set_ylabel('SAM index', fontsize=14)
axs2.set_ylabel('Cumulative SAM', fontsize=14, color='#da2b39')
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\Fig6_SAM.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Load CHL data 1998-2022
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\cciv6data\\')
fh = np.load('chloc4so_19972022.npz', allow_pickle=True)
lat = fh['lat'][144:]
lon = fh['lon']
chl = fh['chl_oc4so'][144:,:,:]
time_date_chl = fh['time_date']
time_date_years_chl = np.empty_like(time_date_chl)
time_date_months_chl = np.empty_like(time_date_chl)
for i in range(0, len(time_date_chl)):
    time_date_years_chl[i] = time_date_chl[i].year
    time_date_months_chl[i] = time_date_chl[i].month
# Correct values
chl[chl > 100] = 100
# Load original 10km clusters
#os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
#fh = np.load('antarcticpeninsula_newclusters_seaicebelow15.npz',allow_pickle = True)
#clusters = fh['clusters']
#lat_clusters = fh['lat']
#lon_clusters = fh['lon'] 
# Load upscaled 4km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_4km.npz',allow_pickle = True)
clusters = fh['clusters']
#lat_clusters = fh['lat']
#lon_clusters = fh['lon'] 
#%% Now for SEP-APRIL
for i in np.arange(1998, 2023):
    yeartemp_sep = chl[:,:, (time_date_years_chl == i-1) & (time_date_months_chl == 9)]
    yeartemp_oct = chl[:,:, (time_date_years_chl == i-1) & (time_date_months_chl == 10)]
    yeartemp_nov = chl[:,:, (time_date_years_chl == i-1) & (time_date_months_chl == 11)]
    yeartemp_dec = chl[:,:, (time_date_years_chl == i-1) & (time_date_months_chl == 12)]
    yeartemp_jan = chl[:,:, (time_date_years_chl == i) & (time_date_months_chl == 1)]
    yeartemp_feb = chl[:,:, (time_date_years_chl == i) & (time_date_months_chl == 2)]
    yeartemp_mar = chl[:,:, (time_date_years_chl == i) & (time_date_months_chl == 3)]
    yeartemp_apr = chl[:,:, (time_date_years_chl == i) & (time_date_months_chl == 4)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr)),2)
    if i == 1998:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
# SEP-APR
for i in np.arange(1998, 2023):
    yeartemp_sepdec = sam_daily[(time_date_years == i-1) & ((time_date_months == 9) | (time_date_months == 10)
                                                                | (time_date_months == 11) | (time_date_months == 12))]
    yeartemp_janapr = sam_daily[(time_date_years == i) & ((time_date_months == 1) | (time_date_months == 2)
                                                                | (time_date_months == 3) | (time_date_months == 4))]
    yeartemp_SEPAPR = np.hstack((yeartemp_sepdec, yeartemp_janapr))
    if i == 1998:
        sam_SEPAPR = np.nanmean(yeartemp_SEPAPR)
    else:
        sam_SEPAPR = np.hstack((sam_SEPAPR, np.nanmean(yeartemp_SEPAPR)))
#%% Calculate correlation
chl_summertrend19992020 = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
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
        corr, pvalue = stats.spearmanr(chl_summerpixel, sam_SEPAPR, nan_policy='omit')
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020[i,j] = corr
        if pvalue < 0.05:
            chl_summertrend19992020_significant[i,j] = corr
#%%
xx, yy = np.meshgrid(lon, lat)
xx = xx[~np.isnan(chl_summertrend19992020_significant)]
yy = yy[~np.isnan(chl_summertrend19992020_significant)]
plt.figure(figsize=(6,6))
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
#map.scatter(xx, yy, 2, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none', alpha=0.5)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Correlation', fontsize=14)
map.contour(lon, lat, chl_summertrend19992020, levels = [0], colors='k',
            transform=ccrs.PlateCarree(), linewidths=.5, linestyles='--')
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\Fig6_CHLSAM.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()


