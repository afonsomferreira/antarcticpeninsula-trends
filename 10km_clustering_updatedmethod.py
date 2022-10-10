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
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from netCDF4 import Dataset
import datetime
import cmocean
import dtw as dtw
from scipy import integrate
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\oc4so-chl\\')
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
### Load data 1998-2020
### Load data 1998-2020
fh = np.load('chloc4so_19972021_10km.npz', allow_pickle=True)
lat = fh['lat'][100:]
lon = fh['lon'][30:250]
chl = fh['chl'][100:, 30:250, :]
time_date = fh['time_date']
# Correct values
chl[chl > 50] = 50
# SST
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia')
### Load data 1998-2020
fh = np.load('sst_19972021_10km.npz', allow_pickle=True)
lat_sst = fh['lat'][100:]
lon_sst = fh['lon'][30:250]
sst = fh['sst'][100:, 30:250, :]
time_date_sst = fh['time_date']
### Load 1981-1996
fh = np.load('sst_19811996_10km.npz', allow_pickle=True)
#lat_ = fh['lat']
#lon = fh['lon']
sst_19811996 = fh['sst'][100:, 30:250, :]
time_date_19811996 = fh['time_date']
# SEA ICE
### Load data 1998-2020
fh = np.load('seaice_19972021_10km.npz', allow_pickle=True)
lat_seaice = fh['lat'][100:]
lon_seaice = fh['lon'][30:250]
seaice = fh['seaice'][100:, 30:250, :]
seaice = seaice*100
time_date_seaice = fh['time_date']
### Load 1981-1996
fh = np.load('seaice_19811996_10km.npz', allow_pickle=True)
#lat_ = fh['lat']
#lon = fh['lon']
seaice_19811996 = fh['seaice'][100:, 30:250, :]
seaice_19811996 = seaice_19811996*100
#time_date_19811996 = fh['time_date']
#PAR
### Load data 1998-2020
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\par\\')
fh = np.load('par_19972022_10km.npz', allow_pickle=True)
lat_par = fh['lat'][100:]
lon_par = fh['lon'][30:250]
par = fh['par'][100:, 30:250, :]
time_date_par = fh['time_date']
#%% Calculate metrics
# Chl-a Mean (Sep-Apr)
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
chl_sepapr19982021 = np.nanmean(chl[:,:, (time_date_months >= 9) | (time_date_months <= 4)],2)
# Bloom Peak
b_init_matrix = np.empty((len(lat), len(lon)))*np.nan
b_term_matrix = np.empty((len(lat), len(lon)))*np.nan
b_peak_matrix = np.empty((len(lat), len(lon)))*np.nan
chl_max_matrix = np.empty((len(lat), len(lon)))*np.nan
#chl_sum_matrix = np.empty((len(lat), len(lon)))*np.nan
#b_dur_matrix = np.empty((len(lat), len(lon)))*np.nan
ix = pd.date_range(start=datetime.date(1997, 8, 1), end=datetime.date(1998, 5, 31), freq='D')
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        pixel_pd = pd.Series(data=chl[i,j,:], index=time_date)
        pixel_averagecycle_pd = pixel_pd.groupby([pixel_pd.index.month, pixel_pd.index.day]).mean()
        pixel_averagecycle_values = pixel_averagecycle_pd.values
        pixel_averagecycle_values = np.delete(pixel_averagecycle_values, 59)
        pixel_averagecycle_augmay = np.hstack((pixel_averagecycle_values[212:],pixel_averagecycle_values[:151]))
        pixel_averagecycle_pd = pd.Series(data=pixel_averagecycle_augmay, index=ix)
        # Average Weekly
        yeartemp_augmay_pd_8day = pixel_averagecycle_pd.resample('8D').mean()
        # Stop if full of Nans
        if np.sum(~np.isnan(yeartemp_augmay_pd_8day.values)) < 20:
            continue        
        # Calculate Median 
        chl_median = np.nanmedian(yeartemp_augmay_pd_8day.values)
        # Check which weeks are above 5% median
        chl_weeksabovemedian5 = yeartemp_augmay_pd_8day > chl_median*1.05
        # Check periods with consecutive weeks above 5% median
        def start_valid_island(a, thresh, window_size):
            m = a>thresh
            me = np.r_[False,m,False]
            idx = np.flatnonzero(me[:-1]!=me[1:])
            lens = idx[1::2]-idx[::2]
            return idx[::2][(lens >= window_size).argmax()]
        b_peak = np.argmax(yeartemp_augmay_pd_8day)
        chl_max = np.nanmax(yeartemp_augmay_pd_8day)
        b_init = start_valid_island(yeartemp_augmay_pd_8day.values, chl_median*1.05, 2)
        b_term = 38 - 1 - start_valid_island(yeartemp_augmay_pd_8day.values[::-1], chl_median*1.05, 2)
#        b_dur = b_term - b_init + 1
#        b_area = np.nansum(yeartemp_augmay_pd_8day.values[b_init:b_term+1])/np.sum(~np.isnan(yeartemp_augmay_pd_8day.values[b_init:b_term+1]))
        # Add to matrices
        b_init_matrix[i,j] = b_init
        b_term_matrix[i,j] = b_term
        b_peak_matrix[i,j] = b_peak
        chl_max_matrix[i,j] = chl_max
#        chl_sum_matrix[i,j] = b_area
#        b_dur_matrix[i,j] =b_dur
# PAR
## First PAR instance
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
par_first_matrix = np.empty((len(lat), len(lon)))*np.nan
par_last_matrix = np.empty((len(lat), len(lon)))*np.nan
ix = pd.date_range(start=datetime.date(1997, 8, 1), end=datetime.date(1998, 5, 31), freq='D')
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        pixel_pd = pd.Series(data=par[i,j,:], index=time_date_par)
        pixel_pd = pixel_pd.resample('D').mean() 
        pixel_averagecycle_pd = pixel_pd.groupby([pixel_pd.index.month, pixel_pd.index.day]).mean()
        pixel_averagecycle_values = pixel_averagecycle_pd.values
        pixel_averagecycle_values = np.delete(pixel_averagecycle_values, 59)
        pixel_averagecycle_augmay = np.hstack((pixel_averagecycle_values[212:],pixel_averagecycle_values[:151]))
        pixel_averagecycle_pd = pd.Series(data=pixel_averagecycle_augmay, index=ix)
        # Average Weekly
        yeartemp_augmay_pd_8day = pixel_averagecycle_pd.resample('8D').mean()
        # Stop if full of Nans
        if np.sum(~np.isnan(yeartemp_augmay_pd_8day.values)) < 20:
            continue        
        # Calculate Median 
        par_median = np.nanmedian(yeartemp_augmay_pd_8day.values)
        # Check which weeks are above 5% median
        par_weeksabovemedian5 = yeartemp_augmay_pd_8day > 0
        # Check periods with consecutive weeks above 5% median
        def start_valid_island(a, thresh, window_size):
            m = a>thresh
            me = np.r_[False,m,False]
            idx = np.flatnonzero(me[:-1]!=me[1:])
            lens = idx[1::2]-idx[::2]
            return idx[::2][(lens >= window_size).argmax()]
#        b_peak = yeartemp_augmay_pd_8day.index[np.argmax(yeartemp_augmay_pd_8day)].dayofyear
#        par_max = np.nanmax(yeartemp_augmay_pd_8day)
        b_init = start_valid_island(yeartemp_augmay_pd_8day.values, 0, 2)
        b_term = 38 - 1 - start_valid_island(yeartemp_augmay_pd_8day.values[::-1], 0, 2)
#        b_dur = b_term - b_init + 1
#        b_area = np.nansum(yeartemp_augmay_pd_8day.values[b_init:b_term+1])/np.sum(~np.isnan(yeartemp_augmay_pd_8day.values[b_init:b_term+1]))
        # Add to matrices
        par_first_matrix[i,j] = b_init
        par_last_matrix[i,j] = b_term
#        b_peak_matrix[i,j] = b_peak
#        par_max_matrix[i,j] = par_max
#        chl_sum_matrix[i,j] = b_area
#        b_dur_matrix[i,j] =b_dur
#%%
map_indexes_number = np.arange(1,26401)
map_indexes = np.array((np.reshape(map_indexes_number,(120,220))),dtype=float)
map_indexes_1D = map_indexes.ravel()
# CHL indices
b_peak_1D = b_peak_matrix.ravel()
chl_max_1D = chl_max_matrix.ravel()
b_init_1D = b_init_matrix.ravel()
b_term_1D = b_term_matrix.ravel()
chl_sepapr_1D = chl_sepapr19982021.ravel()
par_first_1D = par_first_matrix.ravel()
par_last_1D = par_last_matrix.ravel()
map_indexes_1D_nonan = map_indexes_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]
b_peak_1D_nonan = b_peak_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]
chl_max_1D_nonan = chl_max_1D[~np.isnan(chl_max_1D) & ~np.isnan(chl_max_1D)]
b_init_1D_nonan = b_init_1D[~np.isnan(b_init_1D) & ~np.isnan(chl_max_1D)]
b_term_1D_nonan = b_term_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]
chl_sepapr_1D_nonan = chl_sepapr_1D[~np.isnan(chl_sepapr_1D) & ~np.isnan(chl_max_1D)]
par_first_1D_nonan = par_first_1D[~np.isnan(par_first_1D) & ~np.isnan(chl_max_1D) & ~np.isnan(par_first_1D)]
par_last_1D_nonan = par_last_1D[~np.isnan(par_last_1D) & ~np.isnan(chl_max_1D) & ~np.isnan(par_first_1D)]


x = [b_peak_1D_nonan,
     chl_max_1D_nonan,
     b_init_1D_nonan,
     b_term_1D_nonan,
     chl_sepapr_1D_nonan,
#     par_first_1D_nonan,
#     par_last_1D_nonan
     ]
df = pd.DataFrame(x)
df = df.T
df.columns = ['Peak', 'Max', 'Initiation', 'Termination', 'Sep-Apr Mean'],# 'PAR First', 'PAR Last']
import seaborn as sns
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(corr, mask=mask, cmap=plt.cm.seismic, vmax=1,vmin=-1,center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
#%%
# Separating out the features
features = ['Peak', 'Initiation', 'Termination', 'Sep-Apr Mean']
x = df.loc[:, features]
#x = x.dropna().values
from sklearn.preprocessing import normalize
data_scaled = normalize(x)
data_scaled = pd.DataFrame(data_scaled, columns=features)
data_scaled.head()
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Phenoregions Hierarchical Clustering Dendogram")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'), truncate_mode = 'lastp',
                                  color_threshold=10, show_contracted=True)
plt.axhline(y=10, c='k', linewidth=2)
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\hierarchicalclustering_10km.png'
plt.savefig(graphs_dir,format = 'png', dpi = 500)
plt.close()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='manhattan', linkage='average')  
cluster.fit_predict(data_scaled)
y_hdbscan = cluster.labels_

#%%
map_indexes_1D_hdbscan_cluster1 = map_indexes_1D_nonan[y_hdbscan == 0] #cluster1
map_indexes_1D_hdbscan_cluster2 = map_indexes_1D_nonan[y_hdbscan == 1] #cluster2
map_indexes_1D_hdbscan_cluster3 = map_indexes_1D_nonan[y_hdbscan == 2] #cluster3
map_indexes_1D_hdbscan_cluster4 = map_indexes_1D_nonan[y_hdbscan == 3] #cluster4
map_indexes_1D_hdbscan_cluster5 = map_indexes_1D_nonan[y_hdbscan == 4] #cluster5
# match
map_clusters_hdbscan = np.empty((120,220))*np.nan
for k in range(0,120):
    for i in range(0,220):
        if map_indexes[k,i] in map_indexes_1D_hdbscan_cluster1:
            map_clusters_hdbscan[k,i] = 1
        elif map_indexes[k,i] in map_indexes_1D_hdbscan_cluster2:
            map_clusters_hdbscan[k,i] = 2  
        elif map_indexes[k,i] in map_indexes_1D_hdbscan_cluster3:
            map_clusters_hdbscan[k,i] = 3        
        elif map_indexes[k,i] in map_indexes_1D_hdbscan_cluster4:
            map_clusters_hdbscan[k,i] = 4
        elif map_indexes[k,i] in map_indexes_1D_hdbscan_cluster5:
            map_clusters_hdbscan[k,i] = 5

#plot spatial clusters

viridis = plt.cm.get_cmap('viridis', 5)
newcolors = viridis(np.linspace(0, 1, 5))
CoUp = np.array([215/256, 25/256, 28/256, 1]) #215,25,28
CoBa = np.array([153/256, 0, 204/256, 1]) #253,174,97
CoMa = np.array([241/256, 180/256, 47/256, 1]) #255,255,191
OcN = np.array([171/256, 221/256, 164/256, 1]) #171,221,164
OcSW = np.array([43/256, 131/256, 186/256, 1]) #43,131,186
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
newcolors[0, :] = OcSW
newcolors[1, :] = CoUp
newcolors[2, :] = OcN
newcolors[3, :] = CoMa
newcolors[4, :] = CoBa

newcmp = ListedColormap(newcolors)             

plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60]) 
f1 = map.pcolormesh(lon, lat, map_clusters_hdbscan[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=newcmp)
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\hierarchical_5clusters_10km_manhattan_average.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()





















#%%
## SST Mean
time_date_years = np.empty_like(time_date_sst)
time_date_months = np.empty_like(time_date_sst)
for i in range(0, len(time_date_sst)):
    time_date_years[i] = time_date_sst[i].year
    time_date_months[i] = time_date_sst[i].month
sst_sepapr19982021 = np.nanmean(sst[:,:, (time_date_months >= 9) | (time_date_months <= 4)],2)
## First PAR instance
time_date_years = np.empty_like(time_date_par)
time_date_months = np.empty_like(time_date_par)
for i in range(0, len(time_date_par)):
    time_date_years[i] = time_date_par[i].year
    time_date_months[i] = time_date_par[i].month
ix = pd.date_range(start=datetime.date(1997, 8, 1), end=datetime.date(1998, 5, 31), freq='D')
par_first_matrix = np.empty((len(lat), len(lon)))*np.nan
par_last_matrix = np.empty((len(lat), len(lon)))*np.nan
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        pixel_pd = pd.Series(data=par[i,j,:], index=time_date_par)
        pixel_pd = pixel_pd.resample('D').mean()
        pixel_averagecycle_pd = pixel_pd.groupby([pixel_pd.index.month, pixel_pd.index.day]).mean()
        pixel_averagecycle_values = pixel_averagecycle_pd.values
        pixel_averagecycle_values = np.delete(pixel_averagecycle_values, 59)
        pixel_averagecycle_augmay = np.hstack((pixel_averagecycle_values[212:],pixel_averagecycle_values[:151]))
        pixel_averagecycle_pd = pd.Series(data=pixel_averagecycle_augmay, index=ix)
        # Average Weekly
        yeartemp_augmay_pd_8day = pixel_averagecycle_pd.resample('8D').mean()
        # Stop if full of Nans
        if np.sum(~np.isnan(yeartemp_augmay_pd_8day.values)) < 20:
            continue        
        par_first_matrix[i,j] = yeartemp_augmay_pd_8day.index[np.where(yeartemp_augmay_pd_8day.values > 0)[0][0]].dayofyear
        par_last_matrix[i,j] = yeartemp_augmay_pd_8day.index[np.where(yeartemp_augmay_pd_8day.values > 0)[0][-1]].dayofyear
# Sea Ice Retraction, Advance and Mean
seaice_sepapr19982021 = np.nanmean(seaice[:,:, (time_date_months >= 9) | (time_date_months <= 4)],2)

time_date_years = np.empty_like(time_date_seaice)
time_date_months = np.empty_like(time_date_seaice)
for i in range(0, len(time_date_seaice)):
    time_date_years[i] = time_date_seaice[i].year
    time_date_months[i] = time_date_seaice[i].month
seaice_retract_matrix = np.empty((len(lat), len(lon)))*np.nan
seaice_advance_matrix = np.empty((len(lat), len(lon)))*np.nan
ix = pd.date_range(start=datetime.date(1997, 8, 1), end=datetime.date(1998, 5, 31), freq='D')
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        pixel_pd = pd.Series(data=seaice[i,j,:], index=time_date_seaice)
        pixel_pd = pixel_pd.resample('D').mean()
        pixel_averagecycle_pd = pixel_pd.groupby([pixel_pd.index.month, pixel_pd.index.day]).mean()
        pixel_averagecycle_values = pixel_averagecycle_pd.values
        pixel_averagecycle_values = np.delete(pixel_averagecycle_values, 59)
        pixel_averagecycle_augmay = np.hstack((pixel_averagecycle_values[212:],pixel_averagecycle_values[:151]))
        pixel_averagecycle_pd = pd.Series(data=pixel_averagecycle_augmay, index=ix)
        # Average Weekly
        yeartemp_augmay_pd_8day = pixel_averagecycle_pd.resample('8D').mean()
        # Stop if full of Nans
        if np.sum(~np.isnan(yeartemp_augmay_pd_8day.values)) < 20:
            continue        
        if np.sum(yeartemp_augmay_pd_8day.values > 15) == 0:
            continue   
        seaice_retract_matrix[i,j] = yeartemp_augmay_pd_8day.index[np.where(yeartemp_augmay_pd_8day.values > 15)[0][0]].dayofyear
#        par_last_matrix[i,j] = yeartemp_augmay_pd_8day.index[np.where(yeartemp_augmay_pd_8day.values > 0)[0][-1]].dayofyear
ix = pd.date_range(start=datetime.date(1997, 12, 1), end=datetime.date(1998, 8, 31), freq='D')
for i in range(0, len(lat)):
    print(i)
    for j in range(0, len(lon)):
        pixel_pd = pd.Series(data=seaice[i,j,:], index=time_date_seaice)
        pixel_pd = pixel_pd.resample('D').mean()
        pixel_averagecycle_pd = pixel_pd.groupby([pixel_pd.index.month, pixel_pd.index.day]).mean()
        pixel_averagecycle_values = pixel_averagecycle_pd.values
        pixel_averagecycle_values = np.delete(pixel_averagecycle_values, 59)
        pixel_averagecycle_augmay = np.hstack((pixel_averagecycle_values[334:],pixel_averagecycle_values[:243]))
        pixel_averagecycle_pd = pd.Series(data=pixel_averagecycle_augmay, index=ix)
        # Average Weekly
        yeartemp_augmay_pd_8day = pixel_averagecycle_pd.resample('8D').mean()
        # Stop if full of Nans
        if np.sum(~np.isnan(yeartemp_augmay_pd_8day.values)) < 20:
            continue        
        if np.sum(yeartemp_augmay_pd_8day.values > 15) == 0:
            continue   
        seaice_advance_matrix[i,j] = yeartemp_augmay_pd_8day.index[np.where(yeartemp_augmay_pd_8day.values > 15)[0][-1]].dayofyear
#        par_last_matrix[i,j] = yeartemp_augmay_pd_8day.index[np.where(yeartemp_augmay_pd_8day.values > 0)[0][-1]].dayofyear



#%% Plot maps for each
# Chl-a Peak
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, b_peak_matrix[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmap_new,
                    vmin=0, vmax=38)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[0, 4, 8, 12, 16, 20, 24, 27, 31, 35],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'], fontsize=14)
#cbar.set_label('Chl-a Peak Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlpeak_10km_updated.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Chl-a Init Full scale
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, b_init_matrix[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmap_new,
                    vmin=0, vmax=38)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[0, 4, 8, 12, 16, 20, 24, 27, 31, 35],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'], fontsize=14)
#cbar.set_label('Chl-a Peak Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlinit_10km_updated.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Chl-a Init Smaller Scale
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, b_init_matrix[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmap_new,
                    vmin=0, vmax=28)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[0, 4, 8, 12, 16, 20, 24, 27, 31, 35],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'], fontsize=14)
#cbar.set_label('Chl-a Peak Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlinit_10km_updated_shortscale.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Chl-a Term Full Scale
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, b_term_matrix[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmap_new,
                    vmin=0, vmax=38)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[0, 4, 8, 12, 16, 20, 24, 27, 31, 35],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'], fontsize=14)
#cbar.set_label('Chl-a Peak Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlterm_10km_updated.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Chl-a Term Short scale
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, b_term_matrix[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmap_new,
                    vmin=12, vmax=35)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[0, 4, 8, 12, 16, 20, 24, 27, 31, 35],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May'], fontsize=14)
#cbar.set_label('Chl-a Peak Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlterm_10km_updated_shortscale.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Bloom Dur
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, b_dur_matrix[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmap_new,
                    vmin=10, vmax=25)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1,
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(fontsize=14)
#cbar.set_label('Chl-a Peak Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlduration_10km_updated_shortscale.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Chl-a Max
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_max_matrix[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(1),
                    vmax=np.log10(10), cmap=plt.cm.viridis)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(1), np.log10(2), np.log10(3), np.log10(4), np.log10(5), np.log10(6), np.log10(7),
                               np.log10(8), np.log10(9), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['1','2', '3','4','5','6','7','8','9', '10'], fontsize=14)
#cbar.set_label('Chl-a Int Nov-Feb', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlmax_10km_updated.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Chl-a Sum
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_sum_matrix[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(1),
                    vmax=np.log10(10), cmap=plt.cm.viridis)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(1), np.log10(2), np.log10(3), np.log10(4), np.log10(5), np.log10(6), np.log10(7),
                               np.log10(8), np.log10(9), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['1','2', '3','4','5','6','7','8','9', '10'], fontsize=14)
#cbar.set_label('Chl-a Int Nov-Feb', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlsum_10km_updated.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
map_indexes_number = np.arange(1,26401)
map_indexes = np.array((np.reshape(map_indexes_number,(120,220))),dtype=float)
map_indexes_1D = map_indexes.ravel()
b_peak_1D = b_peak_matrix.ravel()
chl_max_1D = chl_max_matrix.ravel()
chl_sum_1D = chl_sum_matrix.ravel()
b_init_1D = b_init_matrix.ravel()
b_term_1D = b_term_matrix.ravel()
b_dur_1D = b_dur_matrix.ravel()
map_indexes_1D_nonan = map_indexes_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]
b_peak_1D_nonan = b_peak_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]
chl_max_1D_nonan = chl_max_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]
b_init_1D_nonan = b_init_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]
b_term_1D_nonan = b_term_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]
b_dur_1D_nonan = b_dur_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]
chl_sum_1D_nonan = chl_sum_1D[~np.isnan(b_peak_1D) & ~np.isnan(chl_max_1D)]





x = [b_peak_1D_nonan,
     chl_max_1D_nonan,
     b_init_1D_nonan,
     b_term_1D_nonan,
     b_dur_1D_nonan,
     chl_sum_1D_nonan
     ]
df = pd.DataFrame(x)
df = df.T
df.columns = ['Peak', 'Max', 'Initiation', 'Termination', 'Duration', 'Sum']
import seaborn as sns
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(corr, mask=mask, cmap=plt.cm.seismic, vmax=1,vmin=-1,center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
#%%
# Separating out the features
features = ['Peak', 'Max', 'Initiation', 'Termination', 'Sum']
x = df.loc[:, features].values
from sklearn.preprocessing import normalize
data_scaled = normalize(x)
data_scaled = pd.DataFrame(data_scaled, columns=features)
data_scaled.head()
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Phenoregions Hierarchical Clustering Dendogram")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'), truncate_mode = 'lastp',
                                  color_threshold=10, show_contracted=True)
plt.axhline(y=10, c='k', linewidth=2)
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\hierarchicalclustering_10km.png'
plt.savefig(graphs_dir,format = 'png', dpi = 500)
plt.close()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='manhattan', linkage='average')  
cluster.fit_predict(data_scaled)
y_hdbscan = cluster.labels_

#%%
map_indexes_1D_hdbscan_cluster1 = map_indexes_1D_nonan[y_hdbscan == 0] #cluster1
map_indexes_1D_hdbscan_cluster2 = map_indexes_1D_nonan[y_hdbscan == 1] #cluster2
map_indexes_1D_hdbscan_cluster3 = map_indexes_1D_nonan[y_hdbscan == 2] #cluster3
map_indexes_1D_hdbscan_cluster4 = map_indexes_1D_nonan[y_hdbscan == 3] #cluster4
map_indexes_1D_hdbscan_cluster5 = map_indexes_1D_nonan[y_hdbscan == 4] #cluster5
# match
map_clusters_hdbscan = np.empty((120,220))*np.nan
for k in range(0,120):
    for i in range(0,220):
        if map_indexes[k,i] in map_indexes_1D_hdbscan_cluster1:
            map_clusters_hdbscan[k,i] = 1
        elif map_indexes[k,i] in map_indexes_1D_hdbscan_cluster2:
            map_clusters_hdbscan[k,i] = 2  
        elif map_indexes[k,i] in map_indexes_1D_hdbscan_cluster3:
            map_clusters_hdbscan[k,i] = 3        
        elif map_indexes[k,i] in map_indexes_1D_hdbscan_cluster4:
            map_clusters_hdbscan[k,i] = 4
        elif map_indexes[k,i] in map_indexes_1D_hdbscan_cluster5:
            map_clusters_hdbscan[k,i] = 5

#plot spatial clusters

viridis = plt.cm.get_cmap('viridis', 5)
newcolors = viridis(np.linspace(0, 1, 5))
CoUp = np.array([215/256, 25/256, 28/256, 1]) #215,25,28
CoBa = np.array([153/256, 0, 204/256, 1]) #253,174,97
CoMa = np.array([241/256, 180/256, 47/256, 1]) #255,255,191
OcN = np.array([171/256, 221/256, 164/256, 1]) #171,221,164
OcSW = np.array([43/256, 131/256, 186/256, 1]) #43,131,186
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
newcolors[0, :] = OcSW
newcolors[1, :] = CoUp
newcolors[2, :] = OcN
newcolors[3, :] = CoMa
newcolors[4, :] = CoBa

newcmp = ListedColormap(newcolors)             

plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60]) 
f1 = map.pcolormesh(lon, lat, map_clusters_hdbscan[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=newcmp)
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\hierarchical_5clusters_10km_manhattan_average.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

#%% Clean Noise
map_clusters_hdbscan_noiseremoved = np.empty_like(map_clusters_hdbscan)
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        pixel_cluster_temp = map_clusters_hdbscan[i, j]
        if i == 0 or i == 1 or i == 119 or i == 120 or j == 0 or j == 220 or j== 1 or j== 219:
            map_clusters_hdbscan_noiseremoved[i, j] = pixel_cluster_temp
            continue
        neighbouring_pixels = map_clusters_hdbscan[i-2:i+3, j-2:j+3]
        if pixel_cluster_temp == stats.mode(neighbouring_pixels.ravel(), nan_policy='omit')[0][0]:
            map_clusters_hdbscan_noiseremoved[i, j] = pixel_cluster_temp
            continue
        else:
            map_clusters_hdbscan_noiseremoved[i, j] = np.nan

# Remove extra bits of noise
map_clusters_hdbscan_noiseremoved[44:47, 90:96] = np.nan
map_clusters_hdbscan_noiseremoved[52:56, 94:98] = np.nan
map_clusters_hdbscan_noiseremoved[52:55, 98] = np.nan
map_clusters_hdbscan_noiseremoved[53:56, 143:148] = np.nan
map_clusters_hdbscan_noiseremoved[54, 142] = np.nan
map_clusters_hdbscan_noiseremoved[56, 143] = np.nan
map_clusters_hdbscan_noiseremoved[55, 148] = np.nan
map_clusters_hdbscan_noiseremoved[62:65, 129:135] = np.nan
map_clusters_hdbscan_noiseremoved[61, 130:134] = np.nan
map_clusters_hdbscan_noiseremoved[63, 126:127] = np.nan
map_clusters_hdbscan_noiseremoved[76:79, 120:123] = np.nan
map_clusters_hdbscan_noiseremoved[74:77, 126:135] = np.nan
map_clusters_hdbscan_noiseremoved[76:78, 139:144] = np.nan
map_clusters_hdbscan_noiseremoved[76:78, 146:158] = np.nan
map_clusters_hdbscan_noiseremoved[74:76, 152:160] = np.nan
map_clusters_hdbscan_noiseremoved[73, 156] = np.nan
map_clusters_hdbscan_noiseremoved[74:79, 163:173] = np.nan
map_clusters_hdbscan_noiseremoved[72, 168] = np.nan
map_clusters_hdbscan_noiseremoved[76, 174] = np.nan
map_clusters_hdbscan_noiseremoved[78:84, 174:195] = np.nan
map_clusters_hdbscan_noiseremoved[77, 178] = np.nan
map_clusters_hdbscan_noiseremoved[86:94, 108:120] = np.nan
map_clusters_hdbscan_noiseremoved[88, 107] = np.nan
map_clusters_hdbscan_noiseremoved[90:99, 97:108] = np.nan
map_clusters_hdbscan_noiseremoved[89, 105:108] = np.nan
map_clusters_hdbscan_noiseremoved[99:105, 98:107] = np.nan
map_clusters_hdbscan_noiseremoved[89:95, 90:100] = np.nan
map_clusters_hdbscan_noiseremoved[76, 49:51] = np.nan
map_clusters_hdbscan_noiseremoved[78, 49] = np.nan
map_clusters_hdbscan_noiseremoved[90:, :] = np.nan #ok
map_clusters_hdbscan_noiseremoved[:20, :] = np.nan #ok
map_clusters_hdbscan_noiseremoved[:, :10] = np.nan #ok
map_clusters_hdbscan_noiseremoved[:, 190:] = np.nan
map_clusters_hdbscan_noiseremoved[57, 118] = np.nan
map_clusters_hdbscan_noiseremoved[58, 118] = np.nan
map_clusters_hdbscan_noiseremoved[58, 116] = np.nan
#%%
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60]) 
f1 = map.pcolormesh(lon, lat, map_clusters_hdbscan_noiseremoved[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=newcmp,
                    vmin=1, vmax=4)
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\hierarchical_5clusters_10km_manhattan_average_cleaned.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()


#%%
#### SAVE ####
np.savez_compressed('antarcticpeninsula_cluster', lat = lat, lon = lon, clusters = map_clusters_hdbscan_noiseremoved)

