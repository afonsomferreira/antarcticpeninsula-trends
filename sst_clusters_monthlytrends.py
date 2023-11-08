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
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline
from netCDF4 import Dataset
from sktime.transformations.series.outlier_detection import HampelFilter
import datetime
import cmocean
import dtw as dtw
from scipy import integrate
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.nanmedian(points, axis=0)
    diff = np.nansum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.nanmedian(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
fh = np.load('sst-seaice_19972021_updated.npz', allow_pickle=True)
lat_sst = fh['lat']
lon_sst = fh['lon']
sst = fh['sst']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_sstseaice.npz',allow_pickle = True)
clusters = fh['clusters']
#lat_clusters_sstseaice = fh['lat']
#lon_clusters_sstseaice = fh['lon']
#%% Separar por clusters
#chl_4km_clusters = np.empty((len(lat), len(lon)))*np.nan
#for i in range(0, len(lat)):
#    print(i)
#    for j in range(0, len(lon)):
#        # Find which pixel
#        matchups_lat_closest = np.where(lat_clusters == min(lat_clusters, key=lambda x:abs(x-lat[i])))[0][0]
#        matchups_lon_closest = np.where(lon_clusters == min(lon_clusters, key=lambda x:abs(x-lon[j])))[0][0]
#        # Find which cluster each point belongs to
#        cluster_temp = clusters[matchups_lat_closest, matchups_lon_closest]
#        # Assign to matrix
#        chl_4km_clusters[i,j] = cluster_temp
#np.savez_compressed('clusters_sst', lat = lat, lon = lon, clusters = chl_4km_clusters)
#%% Plot Clusters map
# Create colormap
#cmap = colors.ListedColormap(['white', '#da2b39', '#2c4ea3', '#f2a612', '#6a984e', '#534d41'])
#bounds=[0,1,2,3,4,5,6]
#norm = colors.BoundaryNorm(bounds, cmap.N)
## Plot figure
#plt.figure()
#map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-63))
#map.set_extent([-67, -53, -67, -60])
#f1 = map.pcolormesh(lon, lat, clusters[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat',
#                    cmap=cmap, norm=norm,zorder=0)
#gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
#map.coastlines(resolution='10m', color='black', linewidth=1)
#map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
#                                        edgecolor='k',
#                                        facecolor=cartopy.feature.COLORS['land']))
#legend_elements = [Patch(facecolor='#f2a612', edgecolor='k',label='DRA'),
#                   Patch(facecolor='#6a984e', edgecolor='k',label='BRS'),
#                   Patch(facecolor='#534d41', edgecolor='k',label='WED$_{N}$'),
#                   Patch(facecolor='#2c4ea3', edgecolor='k',label='GES'),
#                   Patch(facecolor='#da2b39', edgecolor='k',label='WED$_{S}$')]
#plt.legend(handles=legend_elements, loc=4)
#plt.tight_layout()
#graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\clusters_upscaled4km.png'
#plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
#plt.close()
#%% Separar para o cluster 1 (WEDsouth)
weds_cluster = sst[clusters == 1,:]
weds_cluster = np.nanmean(weds_cluster,0)
np.nanmedian(weds_cluster)
np.nanmax(weds_cluster)
np.nanmin(weds_cluster)
np.nanstd(weds_cluster)*3
#weds_cluster1 = np.where(weds_cluster > np.nanmedian(weds_cluster)-np.nanstd(weds_cluster)*3, weds_cluster, np.nan)
#weds_cluster1 = np.where(weds_cluster1 < np.nanmedian(weds_cluster)+np.nanstd(weds_cluster)*3, weds_cluster1, np.nan)
#weds_cluster = weds_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = weds_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = weds_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = weds_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = weds_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = weds_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = weds_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = weds_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = weds_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))    

    if i == 1998:
        weds_sep = np.nanmean(yeartemp_sep)
        weds_oct = np.nanmean(yeartemp_oct)
        weds_nov = np.nanmean(yeartemp_nov)
        weds_dec = np.nanmean(yeartemp_dec)
        weds_jan = np.nanmean(yeartemp_jan)
        weds_feb = np.nanmean(yeartemp_feb)
        weds_mar = np.nanmean(yeartemp_mar)
        weds_apr = np.nanmean(yeartemp_apr)
        weds_sepapr = np.nanmean(yeartemp_sepapr)
        weds_decfeb = np.nanmean(yeartemp_decfeb)
        weds_sepnov = np.nanmean(yeartemp_sepnov)
        weds_marapr = np.nanmean(yeartemp_marapr)
    else:
        weds_sep = np.hstack((weds_sep, np.nanmean(yeartemp_sep)))
        weds_oct = np.hstack((weds_oct, np.nanmean(yeartemp_oct)))
        weds_nov = np.hstack((weds_nov, np.nanmean(yeartemp_nov)))
        weds_dec = np.hstack((weds_dec, np.nanmean(yeartemp_dec)))
        weds_jan = np.hstack((weds_jan, np.nanmean(yeartemp_jan)))
        weds_feb = np.hstack((weds_feb, np.nanmean(yeartemp_feb)))
        weds_mar = np.hstack((weds_mar, np.nanmean(yeartemp_mar)))
        weds_apr = np.hstack((weds_apr, np.nanmean(yeartemp_apr)))
        weds_sepapr = np.hstack((weds_sepapr, np.nanmean(yeartemp_sepapr)))
        weds_decfeb = np.hstack((weds_decfeb, np.nanmean(yeartemp_decfeb)))
        weds_sepnov = np.hstack((weds_sepnov, np.nanmean(yeartemp_sepnov)))
        weds_marapr = np.hstack((weds_marapr, np.nanmean(yeartemp_marapr)))

#%% Calculate trends WEDs
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_sep)], weds_sep[~np.isnan(weds_sep)]) # ***
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_oct)], weds_oct[~np.isnan(weds_oct)]) # **
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_nov)], weds_nov[~np.isnan(weds_nov)]) #
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_dec)], weds_dec[~np.isnan(weds_dec)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_jan)], weds_jan[~np.isnan(weds_jan)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_feb)], weds_feb[~np.isnan(weds_feb)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_mar)], weds_mar[~np.isnan(weds_mar)]) #
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_apr)], weds_apr[~np.isnan(weds_apr)]) #
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_sepapr)], weds_sepapr[~np.isnan(weds_sepapr)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(weds_decfeb)], weds_decfeb[~np.isnan(weds_decfeb)])
#%% Separar para o cluster 2 (GES)
ges_cluster = sst[clusters == 2,:]
ges_cluster = np.nanmean(ges_cluster,0)
np.nanmedian(ges_cluster)
np.nanmax(ges_cluster)
np.nanmin(ges_cluster)
np.nanstd(ges_cluster)*3
#ges_cluster1 = np.where(ges_cluster > np.nanmedian(ges_cluster)-np.nanstd(ges_cluster)*3, ges_cluster, np.nan)
#ges_cluster1 = np.where(ges_cluster1 < np.nanmedian(ges_cluster)+np.nanstd(ges_cluster)*3, ges_cluster1, np.nan)
#ges_cluster = ges_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = ges_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = ges_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = ges_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = ges_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = ges_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = ges_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = ges_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = ges_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        ges_sep = np.nanmean(yeartemp_sep)
        ges_oct = np.nanmean(yeartemp_oct)
        ges_nov = np.nanmean(yeartemp_nov)
        ges_dec = np.nanmean(yeartemp_dec)
        ges_jan = np.nanmean(yeartemp_jan)
        ges_feb = np.nanmean(yeartemp_feb)
        ges_mar = np.nanmean(yeartemp_mar)
        ges_apr = np.nanmean(yeartemp_apr)
        ges_sepapr = np.nanmean(yeartemp_sepapr)
        ges_decfeb = np.nanmean(yeartemp_decfeb)
        ges_sepnov = np.nanmean(yeartemp_sepnov)
        ges_marapr = np.nanmean(yeartemp_marapr)
    else:
        ges_sep = np.hstack((ges_sep, np.nanmean(yeartemp_sep)))
        ges_oct = np.hstack((ges_oct, np.nanmean(yeartemp_oct)))
        ges_nov = np.hstack((ges_nov, np.nanmean(yeartemp_nov)))
        ges_dec = np.hstack((ges_dec, np.nanmean(yeartemp_dec)))
        ges_jan = np.hstack((ges_jan, np.nanmean(yeartemp_jan)))
        ges_feb = np.hstack((ges_feb, np.nanmean(yeartemp_feb)))
        ges_mar = np.hstack((ges_mar, np.nanmean(yeartemp_mar)))
        ges_apr = np.hstack((ges_apr, np.nanmean(yeartemp_apr)))
        ges_sepapr = np.hstack((ges_sepapr, np.nanmean(yeartemp_sepapr)))
        ges_decfeb = np.hstack((ges_decfeb, np.nanmean(yeartemp_decfeb)))
        ges_sepnov = np.hstack((ges_sepnov, np.nanmean(yeartemp_sepnov)))
        ges_marapr = np.hstack((ges_marapr, np.nanmean(yeartemp_marapr)))
#%% Calculate trends ges
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_sep)], ges_sep[~np.isnan(ges_sep)]) #
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_oct)], ges_oct[~np.isnan(ges_oct)]) #
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_nov)], ges_nov[~np.isnan(ges_nov)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_dec)], ges_dec[~np.isnan(ges_dec)]) 
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_jan)], ges_jan[~np.isnan(ges_jan)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_feb)], ges_feb[~np.isnan(ges_feb)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_mar)], ges_mar[~np.isnan(ges_mar)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_apr)], ges_apr[~np.isnan(ges_apr)]) # not enough data
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_sepapr)], ges_sepapr[~np.isnan(ges_sepapr)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_decfeb)], ges_decfeb[~np.isnan(ges_decfeb)])
#%% Separar para o cluster 4 (BRS)
brs_cluster = sst[clusters == 4,:]
brs_cluster = np.nanmean(brs_cluster,0)
np.nanmedian(brs_cluster)
np.nanmax(brs_cluster)
np.nanmin(brs_cluster)
np.nanstd(brs_cluster)*3
#brs_cluster1 = np.where(brs_cluster > np.nanmedian(brs_cluster)-np.nanstd(brs_cluster)*3, brs_cluster, np.nan)
#brs_cluster1 = np.where(brs_cluster1 < np.nanmedian(brs_cluster)+np.nanstd(brs_cluster)*3, brs_cluster1, np.nan)
#brs_cluster = brs_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2022):
    yeartemp_sep = brs_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = brs_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = brs_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = brs_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = brs_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = brs_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = brs_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = brs_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        brs_sep = np.nanmean(yeartemp_sep)
        brs_oct = np.nanmean(yeartemp_oct)
        brs_nov = np.nanmean(yeartemp_nov)
        brs_dec = np.nanmean(yeartemp_dec)
        brs_jan = np.nanmean(yeartemp_jan)
        brs_feb = np.nanmean(yeartemp_feb)
        brs_mar = np.nanmean(yeartemp_mar)
        brs_apr = np.nanmean(yeartemp_apr)
        brs_sepapr = np.nanmean(yeartemp_sepapr)
        brs_decfeb = np.nanmean(yeartemp_decfeb)
        brs_sepnov = np.nanmean(yeartemp_sepnov)
        brs_marapr = np.nanmean(yeartemp_marapr)
    else:
        brs_sep = np.hstack((brs_sep, np.nanmean(yeartemp_sep)))
        brs_oct = np.hstack((brs_oct, np.nanmean(yeartemp_oct)))
        brs_nov = np.hstack((brs_nov, np.nanmean(yeartemp_nov)))
        brs_dec = np.hstack((brs_dec, np.nanmean(yeartemp_dec)))
        brs_jan = np.hstack((brs_jan, np.nanmean(yeartemp_jan)))
        brs_feb = np.hstack((brs_feb, np.nanmean(yeartemp_feb)))
        brs_mar = np.hstack((brs_mar, np.nanmean(yeartemp_mar)))
        brs_apr = np.hstack((brs_apr, np.nanmean(yeartemp_apr)))
        brs_sepapr = np.hstack((brs_sepapr, np.nanmean(yeartemp_sepapr)))
        brs_decfeb = np.hstack((brs_decfeb, np.nanmean(yeartemp_decfeb)))
        brs_sepnov = np.hstack((brs_sepnov, np.nanmean(yeartemp_sepnov)))
        brs_marapr = np.hstack((brs_marapr, np.nanmean(yeartemp_marapr)))
#%% Calculate trends brs
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_sep)], brs_sep[~np.isnan(brs_sep)]) # 
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_oct)], brs_oct[~np.isnan(brs_oct)]) # 
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_nov)], brs_nov[~np.isnan(brs_nov)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_dec)], brs_dec[~np.isnan(brs_dec)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_jan)], brs_jan[~np.isnan(brs_jan)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_feb)], brs_feb[~np.isnan(brs_feb)])
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_mar)], brs_mar[~np.isnan(brs_mar)]) #
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_apr)], brs_apr[~np.isnan(brs_apr)]) #
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_sepapr)], brs_sepapr[~np.isnan(brs_sepapr)]) #
stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_decfeb)], brs_decfeb[~np.isnan(brs_decfeb)])
#%% Separar para o cluster 5 (WEDn)
wedn_cluster = sst[clusters == 5,:]
wedn_cluster = np.nanmean(wedn_cluster,0)
np.nanmedian(wedn_cluster)
np.nanmax(wedn_cluster)
np.nanmin(wedn_cluster)
np.nanstd(wedn_cluster)*3
#wedn_cluster1 = np.where(wedn_cluster > np.nanmedian(wedn_cluster)-np.nanstd(wedn_cluster)*3, wedn_cluster, np.nan)
#wedn_cluster1 = np.where(wedn_cluster1 < np.nanmedian(wedn_cluster)+np.nanstd(wedn_cluster)*3, wedn_cluster1, np.nan)
#wedn_cluster = wedn_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2023):
    yeartemp_sep = wedn_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = wedn_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = wedn_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = wedn_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = wedn_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = wedn_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = wedn_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = wedn_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        wedn_sep = np.nanmean(yeartemp_sep)
        wedn_oct = np.nanmean(yeartemp_oct)
        wedn_nov = np.nanmean(yeartemp_nov)
        wedn_dec = np.nanmean(yeartemp_dec)
        wedn_jan = np.nanmean(yeartemp_jan)
        wedn_feb = np.nanmean(yeartemp_feb)
        wedn_mar = np.nanmean(yeartemp_mar)
        wedn_apr = np.nanmean(yeartemp_apr)
        wedn_sepapr = np.nanmean(yeartemp_sepapr)
        wedn_decfeb = np.nanmean(yeartemp_decfeb)
        wedn_sepnov = np.nanmean(yeartemp_sepnov)
        wedn_marapr = np.nanmean(yeartemp_marapr)
    else:
        wedn_sep = np.hstack((wedn_sep, np.nanmean(yeartemp_sep)))
        wedn_oct = np.hstack((wedn_oct, np.nanmean(yeartemp_oct)))
        wedn_nov = np.hstack((wedn_nov, np.nanmean(yeartemp_nov)))
        wedn_dec = np.hstack((wedn_dec, np.nanmean(yeartemp_dec)))
        wedn_jan = np.hstack((wedn_jan, np.nanmean(yeartemp_jan)))
        wedn_feb = np.hstack((wedn_feb, np.nanmean(yeartemp_feb)))
        wedn_mar = np.hstack((wedn_mar, np.nanmean(yeartemp_mar)))
        wedn_apr = np.hstack((wedn_apr, np.nanmean(yeartemp_apr)))
        wedn_sepapr = np.hstack((wedn_sepapr, np.nanmean(yeartemp_sepapr)))
        wedn_decfeb = np.hstack((wedn_decfeb, np.nanmean(yeartemp_decfeb)))
        wedn_sepnov = np.hstack((wedn_sepnov, np.nanmean(yeartemp_sepnov)))
        wedn_marapr = np.hstack((wedn_marapr, np.nanmean(yeartemp_marapr)))
#%% Calculate trends wedn
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_sep)], wedn_sep[~np.isnan(wedn_sep)]) # **
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_oct)], wedn_oct[~np.isnan(wedn_oct)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_nov)], wedn_nov[~np.isnan(wedn_nov)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_dec)], wedn_dec[~np.isnan(wedn_dec)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_jan)], wedn_jan[~np.isnan(wedn_jan)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_feb)], wedn_feb[~np.isnan(wedn_feb)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_mar)], wedn_mar[~np.isnan(wedn_mar)]) # 
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_apr)], wedn_apr[~np.isnan(wedn_apr)]) #
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_sepapr)], wedn_sepapr[~np.isnan(wedn_sepapr)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(wedn_decfeb)], wedn_decfeb[~np.isnan(wedn_decfeb)])
#%% Separar para o cluster 3 (DRA)
dra_cluster = sst[clusters == 3,:]
dra_cluster = np.nanmean(dra_cluster,0)
np.nanmedian(dra_cluster)
np.nanmax(dra_cluster)
np.nanmin(dra_cluster)
np.nanstd(dra_cluster)*3
#dra_cluster1 = np.where(dra_cluster > np.nanmedian(dra_cluster)-np.nanstd(dra_cluster)*3, dra_cluster, np.nan)
#dra_cluster1 = np.where(dra_cluster1 < np.nanmedian(dra_cluster)+np.nanstd(dra_cluster)*3, dra_cluster1, np.nan)
#dra_cluster = dra_cluster1
# Separate for each month plus December-February plus September-April
for i in np.arange(1998, 2023):
    yeartemp_sep = dra_cluster[(time_date_years == i-1) & (time_date_months == 9)]
    yeartemp_oct = dra_cluster[(time_date_years == i-1) & (time_date_months == 10)]
    yeartemp_nov = dra_cluster[(time_date_years == i-1) & (time_date_months == 11)]
    yeartemp_dec = dra_cluster[(time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = dra_cluster[(time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = dra_cluster[(time_date_years == i) & (time_date_months == 2)]
    yeartemp_mar = dra_cluster[(time_date_years == i) & (time_date_months == 3)]
    yeartemp_apr = dra_cluster[(time_date_years == i) & (time_date_months == 4)]
    yeartemp_sepapr = np.hstack((yeartemp_sep, yeartemp_oct, yeartemp_nov, yeartemp_dec,
                                                yeartemp_jan, yeartemp_feb, yeartemp_mar, yeartemp_apr))
    yeartemp_decfeb = np.hstack((yeartemp_dec,yeartemp_jan, yeartemp_feb))
    yeartemp_sepnov = np.hstack((yeartemp_sep,yeartemp_oct, yeartemp_nov))    
    yeartemp_marapr = np.hstack((yeartemp_mar,yeartemp_apr))  
    if i == 1998:
        dra_sep = np.nanmean(yeartemp_sep)
        dra_oct = np.nanmean(yeartemp_oct)
        dra_nov = np.nanmean(yeartemp_nov)
        dra_dec = np.nanmean(yeartemp_dec)
        dra_jan = np.nanmean(yeartemp_jan)
        dra_feb = np.nanmean(yeartemp_feb)
        dra_mar = np.nanmean(yeartemp_mar)
        dra_apr = np.nanmean(yeartemp_apr)
        dra_sepapr = np.nanmean(yeartemp_sepapr)
        dra_decfeb = np.nanmean(yeartemp_decfeb)
        dra_sepnov = np.nanmean(yeartemp_sepnov)
        dra_marapr = np.nanmean(yeartemp_marapr)
    else:
        dra_sep = np.hstack((dra_sep, np.nanmean(yeartemp_sep)))
        dra_oct = np.hstack((dra_oct, np.nanmean(yeartemp_oct)))
        dra_nov = np.hstack((dra_nov, np.nanmean(yeartemp_nov)))
        dra_dec = np.hstack((dra_dec, np.nanmean(yeartemp_dec)))
        dra_jan = np.hstack((dra_jan, np.nanmean(yeartemp_jan)))
        dra_feb = np.hstack((dra_feb, np.nanmean(yeartemp_feb)))
        dra_mar = np.hstack((dra_mar, np.nanmean(yeartemp_mar)))
        dra_apr = np.hstack((dra_apr, np.nanmean(yeartemp_apr)))
        dra_sepapr = np.hstack((dra_sepapr, np.nanmean(yeartemp_sepapr)))
        dra_decfeb = np.hstack((dra_decfeb, np.nanmean(yeartemp_decfeb)))
        dra_sepnov = np.hstack((dra_sepnov, np.nanmean(yeartemp_sepnov)))
        dra_marapr = np.hstack((dra_marapr, np.nanmean(yeartemp_marapr)))
#%% Calculate trends dra
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_sep)], dra_sep[~np.isnan(dra_sep)]) #
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_oct)], dra_oct[~np.isnan(dra_oct)]) #
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_nov)], dra_nov[~np.isnan(dra_nov)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_dec)], dra_dec[~np.isnan(dra_dec)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_jan)], dra_jan[~np.isnan(dra_jan)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_feb)], dra_feb[~np.isnan(dra_feb)])
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_mar)], dra_mar[~np.isnan(dra_mar)]) #
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_apr)], dra_apr[~np.isnan(dra_apr)]) #
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_sepapr)], dra_sepapr[~np.isnan(dra_sepapr)]) #
stats.linregress(np.arange(1998, 2023)[~np.isnan(dra_decfeb)], dra_decfeb[~np.isnan(dra_decfeb)])
