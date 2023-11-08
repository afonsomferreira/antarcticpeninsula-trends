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
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\oc4so-chl\\')
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
### Load data 1998-2020
fh = np.load('chloc4so_19972021_15km.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
chl = fh['chl']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
chl[chl > 100] = 100
#%% Calculate average yearly cycle for each pixel
yearlyaveragecycles = np.empty((147, 320, 365))
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        pixel_alldata = chl[i, j, :]
        # Convert to pandas series
        pixel_alldata_series = pd.Series(pixel_alldata, index=time_date)
        if pixel_alldata_series.isnull().all():
            yearlyaveragecycles[i, j, :] = np.empty((365))*np.nan
            continue
        else:
            pixel_alldata_series = pixel_alldata_series[~((pixel_alldata_series.index.month == 2) & (pixel_alldata_series.index.day == 29))]
            pixel_yearlyaveragecycle = pixel_alldata_series.groupby([pixel_alldata_series.index.month, pixel_alldata_series.index.day]).mean().values
            yearlyaveragecycles[i, j, :] = pixel_yearlyaveragecycle
# Save datasets with lower resolution
np.savez_compressed('chloc4so_19972021_yearlyaveragecycles_15km', chl_averagecycle=yearlyaveragecycles,time_date=time_date,
                    lat=lat, lon=lon)
