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
#%% Load Chl-a data
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\cciv6data\\')
fh = np.load('chloc4so_19972022.npz', allow_pickle=True)
lat_chl = fh['lat'][144:]
lon_chl = fh['lon']
chl = fh['chl_oc4so'][144:,:,:]
time_date_chl = fh['time_date']
# Correct values
chl[chl > 100] = 100
# Load upscaled 4km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_4km.npz',allow_pickle = True)
clusters_chl = fh['clusters']
lat_clusters_chl = fh['lat']
lon_clusters_chl = fh['lon']
#%% SST
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\old\\')
fh = np.load('sst-seaice_19972021_updated.npz', allow_pickle=True)
lat_sst = fh['lat']
lon_sst = fh['lon']
sst = fh['sst']
time_date_sst = fh['time_date']
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_sstseaice.npz',allow_pickle = True)
clusters_sstseaice = fh['clusters']
#lat_clusters_sstseaice = fh['lat']
#lon_clusters_sstseaice = fh['lon']
#%% Sea Ice
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\old\\')
fh = np.load('sst-seaice_19972021_updated.npz', allow_pickle=True)
#lat_seaice = fh['lat']
#lon_seaice = fh['lon']
seaice = fh['seaice']
time_date_seaice = fh['time_date']
#%% PAR
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\par\\')
fh = np.load('par_19972021_new.npz', allow_pickle=True)
lat_par = fh['lat']
lon_par = fh['lon']
par = fh['par']
time_date_par = fh['time_date']
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_par.npz',allow_pickle = True)
clusters_par = fh['clusters']
#%% Winds
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\winds\\')
fh = np.load('winds_19972022_era5.npz', allow_pickle=True)
lat_winds = fh['lat']
lon_winds = fh['lon']
winds_u = fh['wind_u']
winds_v = fh['wind_v']
time_date_winds = fh['time_date']
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_winds.npz',allow_pickle = True)
clusters_winds = fh['clusters']
#%% Convert clusters to each variable
# Load original 10km clusters
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('antarcticpeninsula_newclusters_seaicebelow15.npz',allow_pickle = True)
clusters_10km = fh['clusters']
lat_clusters_10km = fh['lat']
lon_clusters_10km = fh['lon'] 
# SST and seaice
#sstseaice_clusters = np.empty((len(lat_sst), len(lon_sst)))*np.nan
#for i in range(0, len(lat_sst)):
#    print(i)
#    for j in range(0, len(lon_sst)):
#        # Find which pixel
#        matchups_lat_closest = np.where(lat_clusters_10km == min(lat_clusters_10km, key=lambda x:abs(x-lat_sst[i])))[0][0]
#        matchups_lon_closest = np.where(lon_clusters_10km == min(lon_clusters_10km, key=lambda x:abs(x-lon_sst[j])))[0][0]
#        # Find which cluster each point belongs to
#        cluster_temp = clusters_10km[matchups_lat_closest, matchups_lon_closest]
#        # Assign to matrix
#        sstseaice_clusters[i,j] = cluster_temp
#np.savez_compressed('clusters_upscaled_sstseaice', lat = lat_sst, lon = lon_sst, clusters = sstseaice_clusters)
#  PAR
#clusters_par = np.empty((len(lat_par), len(lon_par)))*np.nan
#for i in range(0, len(lat_par)):
#    print(i)
#    for j in range(0, len(lon_par)):
#        # Find which pixel
#        matchups_lat_closest = np.where(lat_clusters_10km == min(lat_clusters_10km, key=lambda x:abs(x-lat_par[i])))[0][0]
#        matchups_lon_closest = np.where(lon_clusters_10km == min(lon_clusters_10km, key=lambda x:abs(x-lon_par[j])))[0][0]
#        # Find which cluster each point belongs to
#        cluster_temp = clusters_10km[matchups_lat_closest, matchups_lon_closest]
#        # Assign to matrix
#        clusters_par[i,j] = cluster_temp
#np.savez_compressed('clusters_upscaled_par', lat = lat_par, lon = lon_par, clusters = clusters_par)
#  Winds
#clusters_winds = np.empty((len(lat_winds), len(lon_winds)))*np.nan
#for i in range(0, len(lat_winds)):
#    print(i)
#    for j in range(0, len(lon_winds)):
#        # Find which pixel
#        matchups_lat_closest = np.where(lat_clusters_10km == min(lat_clusters_10km, key=lambda x:abs(x-lat_winds[i])))[0][0]
#        matchups_lon_closest = np.where(lon_clusters_10km == min(lon_clusters_10km, key=lambda x:abs(x-lon_winds[j])))[0][0]
#        # Find which cluster each point belongs to
#        cluster_temp = clusters_10km[matchups_lat_closest, matchups_lon_closest]
#        # Assign to matrix
#        clusters_winds[i,j] = cluster_temp
#np.savez_compressed('clusters_upscaled_winds', lat = lat_winds, lon = lon_winds, clusters = clusters_winds)
#%% Testing clusters map
#plt.figure()
#map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-63))
#map.set_extent([-67, -53, -67, -60])
#f1 = map.pcolormesh(lon_winds, lat_winds, clusters_winds[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat')
#gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
#map.coastlines(resolution='10m', color='black', linewidth=1)
#map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
#                                        edgecolor='k',
#                                        facecolor=cartopy.feature.COLORS['land']))
#plt.tight_layout()


#%% DRA cluster (3)
# Chl-a
dra_chl = chl[clusters_chl == 3,:]
dra_chl = np.nanmean(dra_chl,0)
dra_chl = np.where(dra_chl > np.nanmedian(dra_chl)-np.nanstd(dra_chl)*3, dra_chl, np.nan)
dra_chl = np.where(dra_chl < np.nanmedian(dra_chl)+np.nanstd(dra_chl)*3, dra_chl, np.nan)
dra_chl_df = pd.Series(dra_chl, index=time_date_chl, name='chl')
dra_chl_df = dra_chl_df.groupby([dra_chl_df.index.month, dra_chl_df.index.day]).mean()
dra_chl_decfeb = np.hstack((dra_chl_df[335:366].values, dra_chl_df[0:59].values))
# SST
dra_sst = sst[clusters_sstseaice == 3,:]
dra_sst = np.nanmean(dra_sst,0)
#dra_sst = np.where(dra_sst > np.nanmedian(dra_sst)-np.nanstd(dra_sst)*3, dra_sst, np.nan)
#dra_sst = np.where(dra_sst < np.nanmedian(dra_sst)+np.nanstd(dra_sst)*3, dra_sst, np.nan)
dra_sst_df = pd.Series(dra_sst, index=time_date_sst, name='sst')
dra_sst_df = dra_sst_df.groupby([dra_sst_df.index.month, dra_sst_df.index.day]).mean()
dra_sst_decfeb = np.hstack((dra_sst_df[335:366].values, dra_sst_df[0:59].values))
# Sea Ice
dra_seaice = seaice[clusters_sstseaice == 3,:]
dra_seaice = np.nanmean(dra_seaice,0)
#dra_seaice = np.where(dra_seaice > np.nanmedian(dra_seaice)-np.nanstd(dra_seaice)*3, dra_seaice, np.nan)
#dra_seaice = np.where(dra_seaice < np.nanmedian(dra_seaice)+np.nanstd(dra_seaice)*3, dra_seaice, np.nan)
dra_seaice_df = pd.Series(dra_seaice, index=time_date_seaice, name='seaice')
dra_seaice_df = dra_seaice_df.groupby([dra_seaice_df.index.month, dra_seaice_df.index.day]).mean()
dra_seaice_decfeb = np.hstack((dra_seaice_df[335:366].values, dra_seaice_df[0:59].values))
# PAR
dra_par = par[clusters_par == 3,:]
dra_par = np.nanmean(dra_par,0, dtype=np.float64)
#dra_par = np.where(dra_par > np.nanmedian(dra_par)-np.nanstd(dra_par)*3, dra_par, np.nan)
#dra_par = np.where(dra_par < np.nanmedian(dra_par)+np.nanstd(dra_par)*3, dra_par, np.nan)
dra_par_df = pd.Series(dra_par, index=time_date_par, name='par')
dra_par_df = dra_par_df.groupby([dra_par_df.index.month, dra_par_df.index.day]).mean()
dra_par_decfeb = np.hstack((dra_par_df[335:366].values, dra_par_df[0:59].values))# Winds
# Wind
dra_winds_u = winds_u[clusters_winds == 3,:]
dra_winds_u = np.nanmean(dra_winds_u,0)
#dra_winds_u = np.where(dra_winds_u > np.nanmedian(dra_winds_u)-np.nanstd(dra_winds_u)*3, dra_winds_u, np.nan)
#dra_winds_u = np.where(dra_winds_u < np.nanmedian(dra_winds_u)+np.nanstd(dra_winds_u)*3, dra_winds_u, np.nan)
dra_winds_u_df = pd.Series(dra_winds_u, index=time_date_winds, name='winds_u')
dra_winds_u_df = dra_winds_u_df.groupby([dra_winds_u_df.index.month, dra_winds_u_df.index.day]).mean()
dra_winds_u_decfeb = np.hstack((dra_winds_u_df[335:366].values, dra_winds_u_df[0:59].values))
dra_winds_v = winds_v[clusters_winds == 3,:]
dra_winds_v = np.nanmean(dra_winds_v,0)
#dra_winds_v = np.where(dra_winds_v > np.nanmedian(dra_winds_v)-np.nanstd(dra_winds_v)*3, dra_winds_v, np.nan)
#dra_winds_v = np.where(dra_winds_v < np.nanmedian(dra_winds_v)+np.nanstd(dra_winds_v)*3, dra_winds_v, np.nan)
dra_winds_v_df = pd.Series(dra_winds_v, index=time_date_winds, name='winds_v')
dra_winds_v_df = dra_winds_v_df.groupby([dra_winds_v_df.index.month, dra_winds_v_df.index.day]).mean()
dra_winds_v_decfeb = np.hstack((dra_winds_v_df[335:366].values, dra_winds_v_df[0:59].values))
dra_windsspeed = np.sqrt(dra_winds_u_decfeb**2 + dra_winds_v_decfeb**2)
#%% BRS cluster (4)
# Chl-a
brs_chl = chl[clusters_chl == 4,:]
brs_chl = np.nanmean(brs_chl,0)
brs_chl = np.where(brs_chl > np.nanmedian(brs_chl)-np.nanstd(brs_chl)*3, brs_chl, np.nan)
brs_chl = np.where(brs_chl < np.nanmedian(brs_chl)+np.nanstd(brs_chl)*3, brs_chl, np.nan)
brs_chl_df = pd.Series(brs_chl, index=time_date_chl, name='chl')
brs_chl_df = brs_chl_df.groupby([brs_chl_df.index.month, brs_chl_df.index.day]).mean()
brs_chl_decfeb = np.hstack((brs_chl_df[335:366].values, brs_chl_df[0:59].values))
# SST
brs_sst = sst[clusters_sstseaice == 4,:]
brs_sst = np.nanmean(brs_sst,0)
#brs_sst = np.where(brs_sst > np.nanmedian(brs_sst)-np.nanstd(brs_sst)*3, brs_sst, np.nan)
#brs_sst = np.where(brs_sst < np.nanmedian(brs_sst)+np.nanstd(brs_sst)*3, brs_sst, np.nan)
brs_sst_df = pd.Series(brs_sst, index=time_date_sst, name='sst')
brs_sst_df = brs_sst_df.groupby([brs_sst_df.index.month, brs_sst_df.index.day]).mean()
brs_sst_decfeb = np.hstack((brs_sst_df[335:366].values, brs_sst_df[0:59].values))
# Sea Ice
brs_seaice = seaice[clusters_sstseaice == 4,:]
brs_seaice = np.nanmean(brs_seaice,0)
#brs_seaice = np.where(brs_seaice > np.nanmedian(brs_seaice)-np.nanstd(brs_seaice)*3, brs_seaice, np.nan)
#brs_seaice = np.where(brs_seaice < np.nanmedian(brs_seaice)+np.nanstd(brs_seaice)*3, brs_seaice, np.nan)
brs_seaice_df = pd.Series(brs_seaice, index=time_date_seaice, name='seaice')
brs_seaice_df = brs_seaice_df.groupby([brs_seaice_df.index.month, brs_seaice_df.index.day]).mean()
brs_seaice_decfeb = np.hstack((brs_seaice_df[335:366].values, brs_seaice_df[0:59].values))
# PAR
brs_par = par[clusters_par == 4,:]
brs_par = np.nanmean(brs_par,0, dtype=np.float64)
#brs_par = np.where(brs_par > np.nanmedian(brs_par)-np.nanstd(brs_par)*3, brs_par, np.nan)
#brs_par = np.where(brs_par < np.nanmedian(brs_par)+np.nanstd(brs_par)*3, brs_par, np.nan)
brs_par_df = pd.Series(brs_par, index=time_date_par, name='par')
brs_par_df = brs_par_df.groupby([brs_par_df.index.month, brs_par_df.index.day]).mean()
brs_par_decfeb = np.hstack((brs_par_df[335:366].values, brs_par_df[0:59].values))# Winds
# Wind
brs_winds_u = winds_u[clusters_winds == 4,:]
brs_winds_u = np.nanmean(brs_winds_u,0)
#brs_winds_u = np.where(brs_winds_u > np.nanmedian(brs_winds_u)-np.nanstd(brs_winds_u)*3, brs_winds_u, np.nan)
#brs_winds_u = np.where(brs_winds_u < np.nanmedian(brs_winds_u)+np.nanstd(brs_winds_u)*3, brs_winds_u, np.nan)
brs_winds_u_df = pd.Series(brs_winds_u, index=time_date_winds, name='winds_u')
brs_winds_u_df = brs_winds_u_df.groupby([brs_winds_u_df.index.month, brs_winds_u_df.index.day]).mean()
brs_winds_u_decfeb = np.hstack((brs_winds_u_df[335:366].values, brs_winds_u_df[0:59].values))
brs_winds_v = winds_v[clusters_winds == 4,:]
brs_winds_v = np.nanmean(brs_winds_v,0)
#brs_winds_v = np.where(brs_winds_v > np.nanmedian(brs_winds_v)-np.nanstd(brs_winds_v)*3, brs_winds_v, np.nan)
#brs_winds_v = np.where(brs_winds_v < np.nanmedian(brs_winds_v)+np.nanstd(brs_winds_v)*3, brs_winds_v, np.nan)
brs_winds_v_df = pd.Series(brs_winds_v, index=time_date_winds, name='winds_v')
brs_winds_v_df = brs_winds_v_df.groupby([brs_winds_v_df.index.month, brs_winds_v_df.index.day]).mean()
brs_winds_v_decfeb = np.hstack((brs_winds_v_df[335:366].values, brs_winds_v_df[0:59].values))
brs_windsspeed = np.sqrt(brs_winds_u_decfeb**2 + brs_winds_v_decfeb**2)
#%% WEDn cluster (5)
# Chl-a
wedn_chl = chl[clusters_chl == 5,:]
wedn_chl = np.nanmean(wedn_chl,0)
#wedn_chl = np.where(wedn_chl > np.nanmedian(wedn_chl)-np.nanstd(wedn_chl)*3, wedn_chl, np.nan)
#wedn_chl = np.where(wedn_chl < np.nanmedian(wedn_chl)+np.nanstd(wedn_chl)*3, wedn_chl, np.nan)
wedn_chl_df = pd.Series(wedn_chl, index=time_date_chl, name='chl')
wedn_chl_df = wedn_chl_df.groupby([wedn_chl_df.index.month, wedn_chl_df.index.day]).mean()
wedn_chl_decfeb = np.hstack((wedn_chl_df[335:366].values, wedn_chl_df[0:59].values))
# SST
wedn_sst = sst[clusters_sstseaice == 5,:]
wedn_sst = np.nanmean(wedn_sst,0)
#wedn_sst = np.where(wedn_sst > np.nanmedian(wedn_sst)-np.nanstd(wedn_sst)*3, wedn_sst, np.nan)
#wedn_sst = np.where(wedn_sst < np.nanmedian(wedn_sst)+np.nanstd(wedn_sst)*3, wedn_sst, np.nan)
wedn_sst_df = pd.Series(wedn_sst, index=time_date_sst, name='sst')
wedn_sst_df = wedn_sst_df.groupby([wedn_sst_df.index.month, wedn_sst_df.index.day]).mean()
wedn_sst_decfeb = np.hstack((wedn_sst_df[335:366].values, wedn_sst_df[0:59].values))
# Sea Ice
wedn_seaice = seaice[clusters_sstseaice == 5,:]
wedn_seaice = np.nanmean(wedn_seaice,0)
#wedn_seaice = np.where(wedn_seaice > np.nanmedian(wedn_seaice)-np.nanstd(wedn_seaice)*3, wedn_seaice, np.nan)
#wedn_seaice = np.where(wedn_seaice < np.nanmedian(wedn_seaice)+np.nanstd(wedn_seaice)*3, wedn_seaice, np.nan)
wedn_seaice_df = pd.Series(wedn_seaice, index=time_date_seaice, name='seaice')
wedn_seaice_df = wedn_seaice_df.groupby([wedn_seaice_df.index.month, wedn_seaice_df.index.day]).mean()
wedn_seaice_decfeb = np.hstack((wedn_seaice_df[335:366].values, wedn_seaice_df[0:59].values))
# PAR
wedn_par = par[clusters_par == 5,:]
wedn_par = np.nanmean(wedn_par,0, dtype=np.float64)
#wedn_par = np.where(wedn_par > np.nanmedian(wedn_par)-np.nanstd(wedn_par)*3, wedn_par, np.nan)
#wedn_par = np.where(wedn_par < np.nanmedian(wedn_par)+np.nanstd(wedn_par)*3, wedn_par, np.nan)
wedn_par_df = pd.Series(wedn_par, index=time_date_par, name='par')
wedn_par_df = wedn_par_df.groupby([wedn_par_df.index.month, wedn_par_df.index.day]).mean()
wedn_par_decfeb = np.hstack((wedn_par_df[335:366].values, wedn_par_df[0:59].values))# Winds
# Wind
wedn_winds_u = winds_u[clusters_winds == 5,:]
wedn_winds_u = np.nanmean(wedn_winds_u,0)
#wedn_winds_u = np.where(wedn_winds_u > np.nanmedian(wedn_winds_u)-np.nanstd(wedn_winds_u)*3, wedn_winds_u, np.nan)
#wedn_winds_u = np.where(wedn_winds_u < np.nanmedian(wedn_winds_u)+np.nanstd(wedn_winds_u)*3, wedn_winds_u, np.nan)
wedn_winds_u_df = pd.Series(wedn_winds_u, index=time_date_winds, name='winds_u')
wedn_winds_u_df = wedn_winds_u_df.groupby([wedn_winds_u_df.index.month, wedn_winds_u_df.index.day]).mean()
wedn_winds_u_decfeb = np.hstack((wedn_winds_u_df[335:366].values, wedn_winds_u_df[0:59].values))
wedn_winds_v = winds_v[clusters_winds == 5,:]
wedn_winds_v = np.nanmean(wedn_winds_v,0)
#wedn_winds_v = np.where(wedn_winds_v > np.nanmedian(wedn_winds_v)-np.nanstd(wedn_winds_v)*3, wedn_winds_v, np.nan)
#wedn_winds_v = np.where(wedn_winds_v < np.nanmedian(wedn_winds_v)+np.nanstd(wedn_winds_v)*3, wedn_winds_v, np.nan)
wedn_winds_v_df = pd.Series(wedn_winds_v, index=time_date_winds, name='winds_v')
wedn_winds_v_df = wedn_winds_v_df.groupby([wedn_winds_v_df.index.month, wedn_winds_v_df.index.day]).mean()
wedn_winds_v_decfeb = np.hstack((wedn_winds_v_df[335:366].values, wedn_winds_v_df[0:59].values))
wedn_windsspeed = np.sqrt(wedn_winds_u_decfeb**2 + wedn_winds_v_decfeb**2)
#%% GES cluster (2)
# Chl-a
ges_chl = chl[clusters_chl == 2,:]
ges_chl = np.nanmean(ges_chl,0)
#ges_chl = np.where(ges_chl > np.nanmedian(ges_chl)-np.nanstd(ges_chl)*3, ges_chl, np.nan)
#ges_chl = np.where(ges_chl < np.nanmedian(ges_chl)+np.nanstd(ges_chl)*3, ges_chl, np.nan)
ges_chl_df = pd.Series(ges_chl, index=time_date_chl, name='chl')
ges_chl_df = ges_chl_df.groupby([ges_chl_df.index.month, ges_chl_df.index.day]).mean()
ges_chl_decfeb = np.hstack((ges_chl_df[335:366].values, ges_chl_df[0:59].values))
# SST
ges_sst = sst[clusters_sstseaice == 2,:]
ges_sst = np.nanmean(ges_sst,0)
#ges_sst = np.where(ges_sst > np.nanmedian(ges_sst)-np.nanstd(ges_sst)*3, ges_sst, np.nan)
#ges_sst = np.where(ges_sst < np.nanmedian(ges_sst)+np.nanstd(ges_sst)*3, ges_sst, np.nan)
ges_sst_df = pd.Series(ges_sst, index=time_date_sst, name='sst')
ges_sst_df = ges_sst_df.groupby([ges_sst_df.index.month, ges_sst_df.index.day]).mean()
ges_sst_decfeb = np.hstack((ges_sst_df[335:366].values, ges_sst_df[0:59].values))
# Sea Ice
ges_seaice = seaice[clusters_sstseaice == 2,:]
ges_seaice = np.nanmean(ges_seaice,0)
#ges_seaice = np.where(ges_seaice > np.nanmedian(ges_seaice)-np.nanstd(ges_seaice)*3, ges_seaice, np.nan)
#ges_seaice = np.where(ges_seaice < np.nanmedian(ges_seaice)+np.nanstd(ges_seaice)*3, ges_seaice, np.nan)
ges_seaice_df = pd.Series(ges_seaice, index=time_date_seaice, name='seaice')
ges_seaice_df = ges_seaice_df.groupby([ges_seaice_df.index.month, ges_seaice_df.index.day]).mean()
ges_seaice_decfeb = np.hstack((ges_seaice_df[335:366].values, ges_seaice_df[0:59].values))
# PAR
ges_par = par[clusters_par == 2,:]
ges_par = np.nanmean(ges_par,0, dtype=np.float64)
#ges_par = np.where(ges_par > np.nanmedian(ges_par)-np.nanstd(ges_par)*3, ges_par, np.nan)
#ges_par = np.where(ges_par < np.nanmedian(ges_par)+np.nanstd(ges_par)*3, ges_par, np.nan)
ges_par_df = pd.Series(ges_par, index=time_date_par, name='par')
ges_par_df = ges_par_df.groupby([ges_par_df.index.month, ges_par_df.index.day]).mean()
ges_par_decfeb = np.hstack((ges_par_df[335:366].values, ges_par_df[0:59].values))# Winds
# Wind
ges_winds_u = winds_u[clusters_winds == 2,:]
ges_winds_u = np.nanmean(ges_winds_u,0)
#ges_winds_u = np.where(ges_winds_u > np.nanmedian(ges_winds_u)-np.nanstd(ges_winds_u)*3, ges_winds_u, np.nan)
#ges_winds_u = np.where(ges_winds_u < np.nanmedian(ges_winds_u)+np.nanstd(ges_winds_u)*3, ges_winds_u, np.nan)
ges_winds_u_df = pd.Series(ges_winds_u, index=time_date_winds, name='winds_u')
ges_winds_u_df = ges_winds_u_df.groupby([ges_winds_u_df.index.month, ges_winds_u_df.index.day]).mean()
ges_winds_u_decfeb = np.hstack((ges_winds_u_df[335:366].values, ges_winds_u_df[0:59].values))
ges_winds_v = winds_v[clusters_winds == 2,:]
ges_winds_v = np.nanmean(ges_winds_v,0)
#ges_winds_v = np.where(ges_winds_v > np.nanmedian(ges_winds_v)-np.nanstd(ges_winds_v)*3, ges_winds_v, np.nan)
#ges_winds_v = np.where(ges_winds_v < np.nanmedian(ges_winds_v)+np.nanstd(ges_winds_v)*3, ges_winds_v, np.nan)
ges_winds_v_df = pd.Series(ges_winds_v, index=time_date_winds, name='winds_v')
ges_winds_v_df = ges_winds_v_df.groupby([ges_winds_v_df.index.month, ges_winds_v_df.index.day]).mean()
ges_winds_v_decfeb = np.hstack((ges_winds_v_df[335:366].values, ges_winds_v_df[0:59].values))
ges_windsspeed = np.sqrt(ges_winds_u_decfeb**2 + ges_winds_v_decfeb**2)
#%% WEDs cluster (1)
# Chl-a
weds_chl = chl[clusters_chl == 1,:]
weds_chl = np.nanmean(weds_chl,0)
#weds_chl = np.where(weds_chl > np.nanmedian(weds_chl)-np.nanstd(weds_chl)*3, weds_chl, np.nan)
#weds_chl = np.where(weds_chl < np.nanmedian(weds_chl)+np.nanstd(weds_chl)*3, weds_chl, np.nan)
weds_chl_df = pd.Series(weds_chl, index=time_date_chl, name='chl')
weds_chl_df = weds_chl_df.groupby([weds_chl_df.index.month, weds_chl_df.index.day]).mean()
weds_chl_decfeb = np.hstack((weds_chl_df[335:366].values, weds_chl_df[0:59].values))
# SST
weds_sst = sst[clusters_sstseaice == 1,:]
weds_sst = np.nanmean(weds_sst,0)
#weds_sst = np.where(weds_sst > np.nanmedian(weds_sst)-np.nanstd(weds_sst)*3, weds_sst, np.nan)
#weds_sst = np.where(weds_sst < np.nanmedian(weds_sst)+np.nanstd(weds_sst)*3, weds_sst, np.nan)
weds_sst_df = pd.Series(weds_sst, index=time_date_sst, name='sst')
weds_sst_df = weds_sst_df.groupby([weds_sst_df.index.month, weds_sst_df.index.day]).mean()
weds_sst_decfeb = np.hstack((weds_sst_df[335:366].values, weds_sst_df[0:59].values))
# Sea Ice
weds_seaice = seaice[clusters_sstseaice == 1,:]
weds_seaice = np.nanmean(weds_seaice,0)
#weds_seaice = np.where(weds_seaice > np.nanmedian(weds_seaice)-np.nanstd(weds_seaice)*3, weds_seaice, np.nan)
#weds_seaice = np.where(weds_seaice < np.nanmedian(weds_seaice)+np.nanstd(weds_seaice)*3, weds_seaice, np.nan)
weds_seaice_df = pd.Series(weds_seaice, index=time_date_seaice, name='seaice')
weds_seaice_df = weds_seaice_df.groupby([weds_seaice_df.index.month, weds_seaice_df.index.day]).mean()
weds_seaice_decfeb = np.hstack((weds_seaice_df[335:366].values, weds_seaice_df[0:59].values))
# PAR
weds_par = par[clusters_par == 1,:]
weds_par = np.nanmean(weds_par,0, dtype=np.float64)
#weds_par = np.where(weds_par > np.nanmedian(weds_par)-np.nanstd(weds_par)*3, weds_par, np.nan)
#weds_par = np.where(weds_par < np.nanmedian(weds_par)+np.nanstd(weds_par)*3, weds_par, np.nan)
weds_par_df = pd.Series(weds_par, index=time_date_par, name='par')
weds_par_df = weds_par_df.groupby([weds_par_df.index.month, weds_par_df.index.day]).mean()
weds_par_decfeb = np.hstack((weds_par_df[335:366].values, weds_par_df[0:59].values))# Winds
# Wind
weds_winds_u = winds_u[clusters_winds == 1,:]
weds_winds_u = np.nanmean(weds_winds_u,0)
#weds_winds_u = np.where(weds_winds_u > np.nanmedian(weds_winds_u)-np.nanstd(weds_winds_u)*3, weds_winds_u, np.nan)
#weds_winds_u = np.where(weds_winds_u < np.nanmedian(weds_winds_u)+np.nanstd(weds_winds_u)*3, weds_winds_u, np.nan)
weds_winds_u_df = pd.Series(weds_winds_u, index=time_date_winds, name='winds_u')
weds_winds_u_df = weds_winds_u_df.groupby([weds_winds_u_df.index.month, weds_winds_u_df.index.day]).mean()
weds_winds_u_decfeb = np.hstack((weds_winds_u_df[335:366].values, weds_winds_u_df[0:59].values))
weds_winds_v = winds_v[clusters_winds == 1,:]
weds_winds_v = np.nanmean(weds_winds_v,0)
#weds_winds_v = np.where(weds_winds_v > np.nanmedian(weds_winds_v)-np.nanstd(weds_winds_v)*3, weds_winds_v, np.nan)
#weds_winds_v = np.where(weds_winds_v < np.nanmedian(weds_winds_v)+np.nanstd(weds_winds_v)*3, weds_winds_v, np.nan)
weds_winds_v_df = pd.Series(weds_winds_v, index=time_date_winds, name='winds_v')
weds_winds_v_df = weds_winds_v_df.groupby([weds_winds_v_df.index.month, weds_winds_v_df.index.day]).mean()
weds_winds_v_decfeb = np.hstack((weds_winds_v_df[335:366].values, weds_winds_v_df[0:59].values))
weds_windsspeed = np.sqrt(weds_winds_u_decfeb**2 + weds_winds_v_decfeb**2)
#%%
labels_clusters = ['DRA', 'BRS', 'WED$_{N}$', 'GES', 'WED$_{S}$']
boxcolors = ['#f2a612', '#6a984e', '#534d41', '#2c4ea3', '#da2b39']
legend_elements = [Patch(facecolor='#f2a612', edgecolor='k',label='DRA'),
                   Patch(facecolor='#6a984e', edgecolor='k',label='BRS'),
                   Patch(facecolor='#534d41', edgecolor='k',label='WED$_{N}$'),
                   Patch(facecolor='#2c4ea3', edgecolor='k',label='GES'),
                   Patch(facecolor='#da2b39', edgecolor='k',label='WED$_{S}$')]
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
# Subplot 2 - SST
sst_boxplots = [dra_sst_decfeb, brs_sst_decfeb, wedn_sst_decfeb, ges_sst_decfeb, weds_sst_decfeb]
sst_boxplots_df = pd.DataFrame(np.transpose(sst_boxplots), columns=labels_clusters)
ax2 = fig.add_subplot(gs[0, 0])
bp2 = sns.boxenplot(data=sst_boxplots_df, palette=boxcolors, linewidth=1)
#plt.setp(ax2.collections, line_kws )
#boxplot(x=sst_boxplots, labels = labels_clusters, patch_artist = True, notch ='True')
#bp2 = ax2.boxplot(x=sst_boxplots, labels = labels_clusters, patch_artist = True, notch ='True')
#for patch, color in zip(bp2['boxes'], boxcolors):
#    patch.set_facecolor(color)
#for cap in bp2['caps']:
#    cap.set(color ='k',
#            linewidth = 2)
#for median in bp2['medians']:
#    median.set(color ='w',
#               linewidth = 1)
#for flier in bp2['fliers']:
#    flier.set(marker ='x',
#              color ='k')
ax2.set_ylabel('SST (°C)', fontsize=14)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax2.set_ylim(-2, 3)
# Subplot 3 - Sea Ice
seaice_boxplots = [dra_seaice_decfeb*100, brs_seaice_decfeb*100, wedn_seaice_decfeb*100,
                   ges_seaice_decfeb*100, weds_seaice_decfeb*100]
seaice_boxplots_df = pd.DataFrame(np.transpose(seaice_boxplots), columns=labels_clusters)
ax3 = fig.add_subplot(gs[0, 1])
bp3 = sns.boxenplot(data=seaice_boxplots_df, palette=boxcolors, linewidth=1)
#bp3 = ax3.boxplot(x=seaice_boxplots, labels = labels_clusters, patch_artist = True, notch ='True')
#for patch, color in zip(bp3['boxes'], boxcolors):
#    patch.set_facecolor(color)
#for cap in bp3['caps']:
#    cap.set(color ='k',
#            linewidth = 2)
#for median in bp3['medians']:
#    median.set(color ='w',
#               linewidth = 1)
#for flier in bp3['fliers']:
#    flier.set(marker ='x',
#              color ='k')
ax3.set_ylabel('Sea Ice (%)', fontsize=14)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)
ax3.legend(handles=legend_elements, fontsize=14)
# Subplot 4 - PAR
par_boxplots = [dra_par_decfeb, brs_par_decfeb, wedn_par_decfeb,
                   ges_par_decfeb, weds_par_decfeb]
ax4 = fig.add_subplot(gs[1, 0])
par_boxplots_df = pd.DataFrame(np.transpose(par_boxplots), columns=labels_clusters)
bp4 = sns.boxenplot(data=par_boxplots_df, palette=boxcolors, linewidth=1)
#bp4 = ax4.boxplot(x=par_boxplots, labels = labels_clusters, patch_artist = True, notch ='True')
#for patch, color in zip(bp4['boxes'], boxcolors):
#    patch.set_facecolor(color)
#for cap in bp4['caps']:
#    cap.set(color ='k',
#            linewidth = 2)
#for median in bp4['medians']:
#    median.set(color ='w',
#               linewidth = 1)
#for flier in bp4['fliers']:
#    flier.set(marker ='x',
#              color ='k')
ax4.set_ylabel('PAR (E m$^{-2}$ d$^{-1}$)', fontsize=14)
ax4.tick_params(axis='x', labelsize=12)
ax4.tick_params(axis='y', labelsize=12)
# Subplot 5 - Wind Speed
windspeed_boxplots = [dra_windsspeed, brs_windsspeed, wedn_windsspeed,
                   ges_windsspeed, weds_windsspeed]
ax5 = fig.add_subplot(gs[1, 1])
windspeed_boxplots_df = pd.DataFrame(np.transpose(windspeed_boxplots), columns=labels_clusters)
bp5 = sns.boxenplot(data=windspeed_boxplots_df, palette=boxcolors, linewidth=1)
#bp5 = ax5.boxplot(x=windspeed_boxplots, labels = labels_clusters, patch_artist = True, notch ='True')
#for patch, color in zip(bp5['boxes'], boxcolors):
#    patch.set_facecolor(color)
#for cap in bp5['caps']:
#    cap.set(color ='k',
#            linewidth = 2)
#for median in bp5['medians']:
#    median.set(color ='w',
#               linewidth = 1)
#for flier in bp5['fliers']:
#    flier.set(marker ='x',
#              color ='k')
ax5.set_ylabel('Wind Speed (m$^{-1}$)', fontsize=14)
ax5.tick_params(axis='x', labelsize=12)
ax5.tick_params(axis='y', labelsize=12)
fig.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\SupFig1.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()









