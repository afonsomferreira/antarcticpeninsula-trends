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
#%% PAR
# Load 1997-2021 data
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\par\\')
fh = np.load('par_19972021.npz', allow_pickle=True)
lat_par19972021 = fh['lat'][289:]
lon_par19972021 = fh['lon'][72:553]
par19972021 = fh['par'][289:, 72:553, :7732]
time_date_par19972021 = fh['time_date'][:7732]
# Load 2022 data
#fh = np.load('par_2022.npz', allow_pickle=True)
#lat_par2022 = fh['lat']
#lon_par2022 = fh['lon']
#par2022 = fh['par']
#time_date_par2022 = fh['time_date']
# Join both and save
#par19972022 = np.dstack((par19972021, par2022))
#time_date_par19972022 = np.hstack((time_date_par19972021, time_date_par2022))
np.savez_compressed('par_19972021_new', lat = lat_par19972021, lon = lon_par19972021, par = par19972021,
                    time_date = time_date_par19972021)
del(lat_par19972021, lon_par19972021, par19972021, time_date_par19972021)
#%% SST
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
fh = np.load('sst_19972021.npz', allow_pickle=True)
lat_sst19972021 = fh['lat']
lon_sst19972021 = fh['lon']
sst19972021 = fh['sst']
time_date_sst19972021 = fh['time_date']
#np.savez_compressed('sst_19972021_new', lat = lat_sst19972021, lon = lon_sst19972021, sst = sst19972021,
#                    time_date = time_date_sst19972021)
del(lat_sst19972021, lon_sst19972021, sst19972021, time_date_sst19972021)
#%% Sea Ice
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
fh = np.load('seaice_19972021.npz', allow_pickle=True)
lat_seaice19972021 = fh['lat'][:201]
lon_seaice19972021 = fh['lon'][59:460]
seaice19972021 = fh['seaice'][:201, 59:460, :]
time_date_seaice19972021 = fh['time_date']
np.savez_compressed('seaice_19972021_new', lat = lat_seaice19972021, lon = lon_seaice19972021, seaice = seaice19972021,
                    time_date = time_date_seaice19972021)
del(lat_seaice19972021, lon_seaice19972021, seaice19972021, time_date_seaice19972021)
#%% Testing



