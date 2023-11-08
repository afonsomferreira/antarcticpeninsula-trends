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
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from netCDF4 import Dataset
import datetime
import cmocean
import dtw as dtw
from scipy import integrate
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\oc4so-chl\\')
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
### Load data 1998-2020
fh = np.load('chloc4so_19972021_yearlyaveragecycles_month_10km.npz', allow_pickle=True)
lat = fh['lat'][100:]
lon = fh['lon'][30:250]
chl_averagecycle = fh['chl_averagecycle'][100:, 30:250, :]
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
### Take away pixels with very low frequency
### Load data 1998-2020
fh = np.load('chloc4so_19972021_10km.npz', allow_pickle=True)
chl = fh['chl'][100:, 30:250, :]
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        if np.count_nonzero(np.isnan(chl[i, j, :])) > 8500:
            chl[i, j, :] = np.nan
#chl_1D = np.nanmean(chl, 2).ravel()
#%%
map_indexes_number = np.arange(1,26401)
map_indexes = np.array((np.reshape(map_indexes_number,(120,220))),dtype=float)
map_indexes_1D = map_indexes.ravel()
chl_averagecycle_1D = np.reshape(chl_averagecycle, (26400 , 12))
# Exclude pixels that do not have data September to March
chl_averagecycle_1D_sepdec = chl_averagecycle_1D[:, 8:]
chl_averagecycle_1D_janmar = chl_averagecycle_1D[:, :3]
chl_averagecycle_1D_sepmar = np.hstack((chl_averagecycle_1D_sepdec, chl_averagecycle_1D_janmar))
#a = chl_averagecycle_1D_sepmar[~np.isnan(chl_averagecycle_1D_sepmar)]
nonnan_indices = []
for i in range(0, len(chl_averagecycle_1D_sepmar)):
    temp_cycle = chl_averagecycle_1D_sepmar[i, :]
    if ~np.isnan(np.sum(temp_cycle)): #and ~np.isnan(chl_1D[i]):
        nonnan_indices = np.append(nonnan_indices, int(i))

nonnan_indices = nonnan_indices.astype(int)
map_indexes_1D_nonan = map_indexes_1D[nonnan_indices]
chl_averagecycle_1D_sepmar_nonnan = chl_averagecycle_1D_sepmar[nonnan_indices]
# Euclidean k-means
#a = pd.DataFrame(data=chl_averagecycle_1D_sepmar_nonnan, columns = ['S', 'O', 'N', 'D', 'J', 'F', 'M'])
#combined_values = a['J'].map(str) + df1['Month'].map(str) + '-' + df1['Year'].map(str) + ': ' + 'Unemployment: ' + df2['Unemployment Rate'].map(str) + '; ' + 'Interest: ' + df2['Interest Rate'].map(str)
#print (combined_values)
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2)) 

from sklearn.metrics.pairwise import euclidean_distances

distmatrix_eucledian = euclidean_distances(chl_averagecycle_1D_sepmar_nonnan).round(2)

from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram

def hierarchical_clustering(dist_mat, method='complete'):
    if method == 'complete':
        Z = complete(dist_mat)
    if method == 'single':
        Z = single(dist_mat)
    if method == 'average':
        Z = average(dist_mat)
    if method == 'ward':
        Z = ward(dist_mat)
    
    fig = plt.figure(figsize=(16, 8))
    dn = dendrogram(Z)
    plt.title(f"Dendrogram for {method}-linkage with correlation distance")
    plt.show()
    
    return Z

linkage_matrix = hierarchical_clustering(distmatrix_eucledian)

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')  
cluster.fit_predict(chl_averagecycle_1D_sepmar_nonnan)
y_pred = cluster.labels_

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Phenoregions Hierarchical Clustering Dendogram")  
dend = shc.dendrogram(shc.linkage(chl_averagecycle_1D_sepmar_nonnan, method='ward'), truncate_mode = 'lastp',
                                  color_threshold=10, show_contracted=True)
plt.axhline(y=10, c='k', linewidth=2)
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\hierarchicalclustering_10km.png'
plt.savefig(graphs_dir,format = 'png', dpi = 500)
plt.close()


#euclidean_distance(chl_averagecycle_1D_sepmar_nonnan[0], chl_averagecycle_1D_sepmar_nonnan[1])
#%%
plt.figure()
for yi in range(5):
    plt.subplot(5, 5, yi + 1)
    for xx in chl_averagecycle_1D_sepmar_nonnan[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 7)
    #plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

#%%
map_indexes_1D_cluster1 = map_indexes_1D_nonan[y_pred == 0] #cluster1
map_indexes_1D_cluster2 = map_indexes_1D_nonan[y_pred == 1] #cluster2
map_indexes_1D_cluster3 = map_indexes_1D_nonan[y_pred == 2] #cluster3
map_indexes_1D_cluster4 = map_indexes_1D_nonan[y_pred == 3] #cluster3
map_indexes_1D_cluster5 = map_indexes_1D_nonan[y_pred == 4] #cluster3
map_indexes_1D_cluster6 = map_indexes_1D_nonan[y_pred == 5] #cluster3
map_indexes_1D_cluster7 = map_indexes_1D_nonan[y_pred == 6] #cluster3
map_indexes_1D_cluster8 = map_indexes_1D_nonan[y_pred == 7] #cluster3
map_indexes_1D_cluster9 = map_indexes_1D_nonan[y_pred == 8] #cluster3
map_indexes_1D_cluster10 = map_indexes_1D_nonan[y_pred == 9] #cluster3
# match
map_clusters = np.empty((120,220))*np.nan
for k in range(0,120):
    for i in range(0,220):
        if map_indexes[k,i] in map_indexes_1D_cluster1:
            map_clusters[k,i] = 1
        elif map_indexes[k,i] in map_indexes_1D_cluster2:
            map_clusters[k,i] = 2  
        elif map_indexes[k,i] in map_indexes_1D_cluster3:
            map_clusters[k,i] = 3        
        elif map_indexes[k,i] in map_indexes_1D_cluster4:
            map_clusters[k,i] = 4
        elif map_indexes[k,i] in map_indexes_1D_cluster5:
            map_clusters[k,i] = 5
        elif map_indexes[k,i] in map_indexes_1D_cluster6:
            map_clusters[k,i] = 6
        elif map_indexes[k,i] in map_indexes_1D_cluster7:
            map_clusters[k,i] = 7
        elif map_indexes[k,i] in map_indexes_1D_cluster8:
            map_clusters[k,i] = 8
        elif map_indexes[k,i] in map_indexes_1D_cluster9:
            map_clusters[k,i] = 9
        elif map_indexes[k,i] in map_indexes_1D_cluster10:
            map_clusters[k,i] = 10

#%%
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60]) 
f1 = map.pcolormesh(lon, lat, map_clusters[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat')
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\hierarchical_4clusters_10km_usingtimeseries.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()








#%% Calculate metrics
# Calculate metrics for each pixel
month_chl_max = np.empty((len(lat), len(lon)))*np.nan
chl_max = np.empty((len(lat), len(lon)))*np.nan
chl_mean = np.empty((len(lat), len(lon)))*np.nan
#chl_mean = np.empty((len(lat), len(lon)))*np.nan
month_chl_startafterwinter = np.empty((len(lat), len(lon)))*np.nan
month_chl_stopafterspring = np.empty((len(lat), len(lon)))*np.nan
total_chla_novfeb = np.empty((len(lat), len(lon)))*np.nan
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        if np.count_nonzero(np.isnan(chl_averagecycle[i,j,:])) > 7:
            continue
        # Calculate max chl
        chl_max_temp = np.nanmax(chl_averagecycle[i,j,:])
        chl_max[i,j] = chl_max_temp       
        # Calculate mean chl
        chl_mean_temp = np.nanmean(chl_averagecycle[i,j,:])
        chl_mean[i,j] = chl_mean_temp          
        # find month where chl reaches its max
        month_chl_max_temp = np.nanargmax(chl_averagecycle[i,j,:]) + 1
        month_chl_max[i,j] = month_chl_max_temp
        # find month where phytoplankton begins to grow (post winter)
        try:
            monthfirstchlafterwinter = next(i for i,v in enumerate(chl_averagecycle[i,j,:][5:]) if v > 0) + 6
        except:
            monthfirstchlafterwinter = next(i for i,v in enumerate(chl_averagecycle[i,j,:]) if v > 0)
        month_chl_startafterwinter[i,j] = monthfirstchlafterwinter  
        # find month where phytoplankton stops growing (after spring)
        monthlastchl = 7 - next(i for i,v in enumerate(np.flip(chl_averagecycle[i,j,:])[5:]) if v > 0)
        month_chl_stopafterspring[i,j] = monthlastchl
        total_chla_novfeb[i,j] = integrate.simps(np.hstack((chl_averagecycle[i,j,-2:],
                                                           chl_averagecycle[i,j,:2])))

#%%
map_indexes_number = np.arange(1,26401)
map_indexes = np.array((np.reshape(map_indexes_number,(120,220))),dtype=float)
map_indexes_1D = map_indexes.ravel()
month_chl_max_1D = month_chl_max.ravel()
chl_max_1D = chl_max.ravel()
chl_mean_1D = chl_mean.ravel()
month_chl_startafterwinter_1D = month_chl_startafterwinter.ravel()
month_chl_stopafterspring_1D = month_chl_stopafterspring.ravel()
total_chla_novfeb_1D = total_chla_novfeb.ravel()
map_indexes_1D_nonan = map_indexes_1D[~np.isnan(month_chl_max_1D)]
month_chl_max_1D_nonan = month_chl_max_1D[~np.isnan(month_chl_max_1D)]
chl_max_1D_nonan = chl_max_1D[~np.isnan(month_chl_max_1D)]
chl_mean_1D_nonan = chl_mean_1D[~np.isnan(month_chl_max_1D)]
month_chl_startafterwinter_1D_nonan = month_chl_startafterwinter_1D[~np.isnan(month_chl_max_1D)]
month_chl_stopafterspring_1D_nonan = month_chl_stopafterspring_1D[~np.isnan(month_chl_max_1D)]
#total_chla_novfeb_1D_nonan = total_chla_novfeb_1D[~np.isnan(month_chl_max_1D)]
x = [month_chl_max_1D_nonan,
     chl_mean_1D_nonan,
     month_chl_startafterwinter_1D_nonan,
     month_chl_stopafterspring_1D_nonan
     ]
df = pd.DataFrame(x)
df = df.T
df.columns = ['Peak', 'Mean', 'Initiation', 'Termination']
import seaborn as sns
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(corr, mask=mask, cmap=plt.cm.seismic, vmax=1,vmin=-1,center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
#%%
# Separating out the features
features = ['Peak', 'Mean', 'Initiation', 'Termination']
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
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)
y_hdbscan = cluster.labels_

#%%
map_indexes_1D_hdbscan_cluster1 = map_indexes_1D_nonan[y_hdbscan == 0] #cluster1
map_indexes_1D_hdbscan_cluster2 = map_indexes_1D_nonan[y_hdbscan == 1] #cluster2
map_indexes_1D_hdbscan_cluster3 = map_indexes_1D_nonan[y_hdbscan == 2] #cluster3
map_indexes_1D_hdbscan_cluster4 = map_indexes_1D_nonan[y_hdbscan == 3] #cluster3
map_indexes_1D_hdbscan_cluster5 = map_indexes_1D_nonan[y_hdbscan == 4] #cluster3
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\clustering\\hierarchical_4clusters_10km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()



plt.figure(figsize=(12,9))
map = plt.axes(projection = ccrs.PlateCarree())
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([lon[0],lon[-1],lat[-1],lat[0]])
map.add_feature(cartopy.feature.NaturalEarthFeature(category='physical', name='land',
                            scale='10m', facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon,lat,map_clusters_hdbscan,shading='flat',cmap=newcmp)
gl = map.gridlines(draw_labels = True,alpha=0.5, linestyle='dotted', color='black')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18,'color': 'black'}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([44, 42, 40, 38, 36])
gl.xlocator = mticker.FixedLocator([-12, -10, -8, -6])
#plt.title('Phenology HDBSCAN Clusters', fontsize=11)
plt.colorbar(f1)
#plt.tight_layout()
graphs_dir = 'C:\\Users\\Afonso\\Documents\\Trabalho\\Artigos\\Phenology WIC\\2020\\2019\\CCI\\Graphs\\CCI_v5\\rf\\phenoregions_map_5_v42.png'
plt.savefig(graphs_dir,format = 'png', dpi = 500)
plt.close()


