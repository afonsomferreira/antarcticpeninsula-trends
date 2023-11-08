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
# Sep-Apr MEI
for i in np.arange(1998, 2023):
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
#%% Average monthly
# calculate 1 year moving mean
meiv2_df = pd.DataFrame(meiv2)
mei_monthly_movingmean = meiv2_df.rolling(12).mean().values
mei_monthly_cumsum = meiv2.cumsum()
#%%
fig, axs = plt.subplots(1, 1, figsize=(6,4))
lns1 = axs.plot(np.arange(516), meiv2_df.values, 'gray', linewidth=1.5, alpha=0.4, label='MEI')
lns2 = axs.plot(np.arange(516), mei_monthly_movingmean, 'k', linewidth=1.5, alpha=1, label='MEI Mov. Mean')
axs.set_ylim(-3, 3)
axs2 = axs.twinx()
lns3 = axs2.plot(np.arange(516), mei_monthly_cumsum, c='#da2b39', linewidth=1.5, alpha=1, label='Cumulative MEI', linestyle='-')
plt.xlim(0,516)
axs2.set_ylim(-100,100)
axs.axhline(c='k', linewidth=1, alpha=1, linestyle='-')
plt.axvline(360,c='k', linewidth=1, alpha=.7, linestyle='--')
#axs.plot(np.arange(529)[336:], np.arange(529)[336:]*slope + intercept, linestyle='--', c='k', label='2007-2022 Trend')
plt.xticks(ticks=[0, 60, 120, 180, 240, 300, 360, 420, 480], labels=['1980', '1985', '1990', '1995', '2000',
                                              '2005', '2010', '2015', '2020'])
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
axs.legend(lns, labs, loc=3, fontsize=12)
axs2.tick_params(axis='y', colors='#da2b39')
#fig.legend(loc="upper left")
#fig.legend(loc="upper right", bbox_to_anchor=(1.1,1.14), bbox_transform=axs.transAxes, ncol=4, fontsize=10)
axs.set_ylabel('MEI index', fontsize=14)
axs2.set_ylabel('Cumulative MEI', fontsize=14, color='#da2b39')
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\Fig6_MEI.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()


#%% Calculate correlation
chl_summertrend19992020_mei = np.empty((np.size(lat), np.size(lon)))*np.nan
chl_summertrend19992020_significant_mei = np.empty((np.size(lat), np.size(lon)))*np.nan
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
        corr, pvalue = stats.spearmanr(chl_summerpixel, mei_sepapr, nan_policy='omit')
        # add slope value (i.e. change in chl per year) to pixel
        chl_summertrend19992020_mei[i,j] = corr
        if pvalue < 0.05:
            chl_summertrend19992020_significant_mei[i,j] = corr
#%%
xx, yy = np.meshgrid(lon, lat)
#xx = xx[~np.isnan(chl_summertrend19992020_significant)]
#yy = yy[~np.isnan(chl_summertrend19992020_significant)]
plt.figure(figsize=(6,6))
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_summertrend19992020_mei[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
#map.scatter(xx, yy, 2, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none', alpha=0.5)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Correlation', fontsize=14)
map.contour(lon, lat, chl_summertrend19992020_mei, levels = [0], colors='k',
            transform=ccrs.PlateCarree(), linewidths=.5, linestyles='--')
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\Fig6_CHLMEI.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

#%% Create map to see pixels where SAM has a positive correlation and MEI a negative
samvsmei_correlations = np.empty((np.size(lat), np.size(lon)))*np.nan
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        if (chl_summertrend19992020[i,j] > 0) & (chl_summertrend19992020_mei[i,j] < 0):
            samvsmei_correlations[i,j] = 1
        elif ((chl_summertrend19992020[i,j] < 0) & (chl_summertrend19992020_mei[i,j] > 0)):
            samvsmei_correlations[i,j] = 1

plt.figure(figsize=(6,6))
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, samvsmei_correlations[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', vmin=-1, vmax=1,
                    cmap=cmocean.cm.balance, zorder=0)
#map.contour(lon, lat, chl_summertrend19992020_significant, levels=[0, 0.01], colors='black', zorder=1, transform=ccrs.PlateCarree())
#map.plot(xx,yy,'k*', markeredgewidth=1, markersize=.5, transform=ccrs.PlateCarree(), alpha=1)
#map.scatter(xx, yy, 2, transform=ccrs.PlateCarree(), c='k', marker='.', edgecolors='none', alpha=0.5)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#plt.contour(X, Y, Z, levels=[0.5, 0.75], colors=['black','cyan'])
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\samvsmei_correlationcomparison.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()                       








