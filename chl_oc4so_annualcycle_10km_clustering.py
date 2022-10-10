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
#%% Calculate metrics
# Calculate metrics for each pixel
month_chl_max = np.empty((len(lat), len(lon)))*np.nan
chl_max = np.empty((len(lat), len(lon)))*np.nan
chl_mean = np.empty((len(lat), len(lon)))*np.nan
#chl_mean = np.empty((len(lat), len(lon)))*np.nan
month_chl_startafterwinter = np.empty((len(lat), len(lon)))*np.nan
month_chl_stopafterspring = np.empty((len(lat), len(lon)))*np.nan
total_chla_novfeb = np.empty((len(lat), len(lon)))*np.nan
chl_amp = np.empty((len(lat), len(lon)))*np.nan
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        if np.count_nonzero(np.isnan(chl_averagecycle[i,j,:])) > 7:
            continue
        # Calculate max chl
        chl_max_temp = np.nanmax(chl_averagecycle[i,j,:])
        chl_max[i,j] = chl_max_temp       
        # Calculate chl amplitude
        chl_amp_temp = np.nanmax(chl_averagecycle[i,j,:]) - np.nanmean(chl_averagecycle[i,j,:])
        chl_amp[i,j] = chl_amp_temp 
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

#%% Plot maps for each
# Chl-a Peak
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, month_chl_max[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmap_new,
                    vmin=1, vmax=12)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=np.arange(1,13),
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=14)
cbar.set_label('Chl-a Peak Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlpeak_10km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Chl-a Init
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, month_chl_startafterwinter[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmap_new,
                    vmin=1, vmax=12)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=np.arange(1,13),
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=14)
cbar.set_label('Chl-a Peak Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlinit_10km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Chl-a Term
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, month_chl_stopafterspring[:-1,:-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmap_new,
                    vmin=1, vmax=12)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=np.arange(1,13),
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=14)
cbar.set_label('Chl-a Peak Month', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlterm_10km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Chl-a Int Nov-Feb
cmap_data = np.loadtxt("erdc_iceFire.txt")
cmap_new = LinearSegmentedColormap.from_list('my_colormap', cmap_data)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(total_chla_novfeb[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(1),
                    vmax=np.log10(100), cmap=plt.cm.viridis)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(1), np.log10(3), np.log10(10), np.log10(30), np.log10(100)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['1', '3', '10', '30', '100'], fontsize=14)
cbar.set_label('Chl-a Int Nov-Feb', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chlintnovfeb_10km.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%%
map_indexes_number = np.arange(1,26401)
map_indexes = np.array((np.reshape(map_indexes_number,(120,220))),dtype=float)
map_indexes_1D = map_indexes.ravel()
month_chl_max_1D = month_chl_max.ravel()
chl_max_1D = chl_max.ravel()
chl_mean_1D = chl_mean.ravel()
chl_amp_1D = chl_amp.ravel()
total_chla_novfeb_1D = total_chla_novfeb.ravel()
month_chl_startafterwinter_1D = month_chl_startafterwinter.ravel()
month_chl_stopafterspring_1D = month_chl_stopafterspring.ravel()
total_chla_novfeb_1D = total_chla_novfeb.ravel()
map_indexes_1D_nonan = map_indexes_1D[~np.isnan(month_chl_max_1D) & ~np.isnan(total_chla_novfeb_1D)]
month_chl_max_1D_nonan = month_chl_max_1D[~np.isnan(month_chl_max_1D) & ~np.isnan(total_chla_novfeb_1D)]
chl_max_1D_nonan = chl_max_1D[~np.isnan(month_chl_max_1D) & ~np.isnan(total_chla_novfeb_1D)]
chl_mean_1D_nonan = chl_mean_1D[~np.isnan(month_chl_max_1D) & ~np.isnan(total_chla_novfeb_1D)]
chl_amp_1D_nonan = chl_amp_1D[~np.isnan(month_chl_max_1D) & ~np.isnan(total_chla_novfeb_1D)]
total_chla_novfeb_1D_nonan = total_chla_novfeb_1D[~np.isnan(month_chl_max_1D) & ~np.isnan(total_chla_novfeb_1D)]
month_chl_startafterwinter_1D_nonan = month_chl_startafterwinter_1D[~np.isnan(month_chl_max_1D) & ~np.isnan(total_chla_novfeb_1D)]
month_chl_stopafterspring_1D_nonan = month_chl_stopafterspring_1D[~np.isnan(month_chl_max_1D) & ~np.isnan(total_chla_novfeb_1D)]





#total_chla_novfeb_1D_nonan = total_chla_novfeb_1D[~np.isnan(month_chl_max_1D)]
x = [month_chl_max_1D_nonan,
     chl_mean_1D_nonan,
     month_chl_startafterwinter_1D_nonan,
     month_chl_stopafterspring_1D_nonan,
     chl_amp_1D_nonan,
     total_chla_novfeb_1D_nonan
     ]
df = pd.DataFrame(x)
df = df.T
df.columns = ['Peak', 'Mean', 'Initiation', 'Termination', 'Amplitude', 'Integrated Nov-Feb']
import seaborn as sns
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(corr, mask=mask, cmap=plt.cm.seismic, vmax=1,vmin=-1,center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
#%%
# Separating out the features
features = ['Peak', 'Initiation', 'Termination','Integrated Nov-Feb']
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

