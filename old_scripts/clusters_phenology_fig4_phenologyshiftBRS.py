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
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.path import Path
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from netCDF4 import Dataset
from sktime.transformations.series.outlier_detection import HampelFilter
import datetime
import cmocean
import dtw as dtw
from scipy import integrate
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.patches as mpatches
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
#%% Phenology BRS
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('phenology_BRS_10km.npz', allow_pickle=True)
brs_binit = fh['b_init']
brs_bterm = fh['b_term']
brs_bpeak = fh['b_peak']
brs_chlmax = fh['chl_max']
brs_barea = fh['b_area']
brs_bdur = fh['b_dur']
brs_timeyears = fh['time_years']
#%%
fig, ax1 = plt.subplots(1,figsize=(6,4))
for i in range(0,len(brs_timeyears)):
    if brs_binit[i] < brs_bterm[i]:
        print(i)
        rect = mpl.patches.Rectangle([brs_timeyears[i]-0.4,brs_binit[i]],0.5,brs_bterm[i]-brs_binit[i],facecolor=[106/256, 153/256, 78/256, .7],
                                     linewidth=1)
        ax1.add_patch(rect)
    else:
        print(i)
        rect = mpl.patches.Rectangle([brs_timeyears[i]-0.25,brs_binit[i]],0.5,365-brs_binit[i],facecolor=[106/256, 153/256, 78/256, .7],
                                     linewidth=1)
        rect2 = mpl.patches.Rectangle([brs_timeyears[i]-0.25,0],0.5,brs_bterm[i],facecolor=[106/256, 153/256, 78/256, .7],
                                     linewidth=1)
        ax1.add_patch(rect)
        ax1.add_patch(rect2)
        
for i in range(0,len(brs_timeyears)):
    ax1.hlines(y=brs_bpeak[i],xmin=brs_timeyears[i]-0.4,xmax=brs_timeyears[i]+0.05,linewidth=5,color='k')
    #ax1.hlines(y=brs_binit[i],xmin=brs_timeyears[i]-0.25,xmax=brs_timeyears[i]+0.25,linewidth=2,color='k')
    #ax1.hlines(y=brs_bterm[i],xmin=brs_timeyears[i]-0.25,xmax=brs_timeyears[i]+0.25,linewidth=2,color='k')
#    ax1.text(x=years[i]-0.15,y=brs_binit[i]-15,s='BI', fontsize=14)
#    ax1.text(x=years[i]-0.18,y=brs_bterm[i]+5,s='BT', fontsize=14)
#plt.axvline(2009.5,0,35, linestyle='-', color='k',linewidth=2)
plt.axhline(21.08,0,.5, linestyle='--', color='k',linewidth=2)
plt.axhline(24.42,.5,1, linestyle='--', color='k',linewidth=2)

#plt.scatter(years,B_start,s=150,marker='_')
#plt.scatter(years,B_end,s=150,marker='_')
#plt.scatter(years,B_max)
#plt.ylim(1,39)
plt.yticks(ticks=[1, 4, 8, 12, 16, 20, 23, 27, 31, 35, 38],
           labels=['AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN'],
           fontsize=16)
plt.ylim(11,33)
plt.xticks(ticks=[2000, 2005, 2010, 2015, 2020], fontsize=12)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\brs_blooms.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()