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
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1970, 1, 1, 0, 0) + datetime.timedelta(days=srl_no)
    return new_date
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\rrs\\20042009')
### Load and join data
files = os.listdir()
for i in files:
    print(i)
    fh = Dataset(i, mode='r')
    rrs_443_temp = fh.variables['Rrs_443'][:]
    rrs_443_temp = np.swapaxes(np.swapaxes(rrs_443_temp, 0, 2), 0, 1)
    rrs_443_temp[rrs_443_temp == 9.96921e+36] = np.nan
    rrs_490_temp = fh.variables['Rrs_490'][:]
    rrs_490_temp = np.swapaxes(np.swapaxes(rrs_490_temp, 0, 2), 0, 1)
    rrs_490_temp[rrs_490_temp == 9.96921e+36] = np.nan
    rrs_510_temp = fh.variables['Rrs_510'][:]
    rrs_510_temp = np.swapaxes(np.swapaxes(rrs_510_temp, 0, 2), 0, 1)
    rrs_510_temp[rrs_510_temp == 9.96921e+36] = np.nan
    rrs_560_temp = fh.variables['Rrs_560'][:]
    rrs_560_temp = np.swapaxes(np.swapaxes(rrs_560_temp, 0, 2), 0, 1)
    rrs_560_temp[rrs_560_temp == 9.96921e+36] = np.nan
    time = fh.variables['time'][:]
    time_date_temp = np.empty_like(time, dtype=object)
    for k in range(0, len(time)):
        time_date_temp[k] = serial_date_to_string(int(time[k]))
    if i == files[0]:
        rrs_443 = rrs_443_temp
        rrs_490 = rrs_490_temp
        rrs_510 = rrs_510_temp
        rrs_560 = rrs_560_temp
        lat = fh.variables['lat'][:]
        lon = fh.variables['lon'][:]
        time_date = time_date_temp
    else:
        rrs_443 = np.dstack((rrs_443, rrs_443_temp))
        rrs_490 = np.dstack((rrs_490, rrs_490_temp))
        rrs_510 = np.dstack((rrs_510, rrs_510_temp))
        rrs_560 = np.dstack((rrs_560, rrs_560_temp))
        time_date = np.hstack((time_date, time_date_temp))
    del(fh, rrs_443_temp, rrs_490_temp, rrs_510_temp, rrs_560_temp, time_date_temp)
       
np.savez_compressed('cci5_rrs_20042009_daily', rrs_443=rrs_443,
                    rrs_490=rrs_490, rrs_510=rrs_510,
                    rrs_560=rrs_560, time_date=time_date,
                    lat=lat, lon=lon)