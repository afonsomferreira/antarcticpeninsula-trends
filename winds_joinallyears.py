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
    new_date = datetime.datetime(1900, 1, 1, 0, 0) + datetime.timedelta(hours=srl_no)
    return new_date
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\winds\\')
### Load and join data
files = os.listdir()
for i in files:
    print(i)
    fh = Dataset(i, mode='r')
    if i == files[0]:
        lat = np.array(fh['lat'])
        lon = np.array(fh['lon'])
        #sst = np.array(fh['analysed_sst'])
        #sst[sst == -32768] = np.nan
        #sst = sst-273.15
        #sst = np.float16(sst)
        #sst = np.swapaxes(np.swapaxes(sst, 0, 2), 0, 1)
        eastward_wind = np.array(fh['eastward_wind'])
        eastward_wind[eastward_wind == 9.969209968386869e+36] = np.nan
        eastward_wind = np.float16(eastward_wind)
        eastward_wind = np.swapaxes(np.swapaxes(eastward_wind, 0, 2), 0, 1)
        northward_wind = np.array(fh['northward_wind'])
        northward_wind[northward_wind == 9.969209968386869e+36] = np.nan
        northward_wind = np.float16(northward_wind)
        northward_wind = np.swapaxes(np.swapaxes(northward_wind, 0, 2), 0, 1)
        time = np.array(fh['time'])
        #Converts to datetime
        time_date = np.empty((len(time)), dtype=object)
        for i in range(0, len(time)):
            time_date[i] = serial_date_to_string(int(time[i]))
    else:
        northward_wind_temp = np.array(fh['northward_wind'])
        northward_wind_temp[northward_wind_temp == 9.969209968386869e+36] = np.nan
        northward_wind_temp = np.float16(northward_wind_temp)
        northward_wind_temp = np.swapaxes(np.swapaxes(northward_wind_temp, 0, 2), 0, 1)
        eastward_wind_temp = np.array(fh['eastward_wind'])
        eastward_wind_temp[eastward_wind_temp == 9.969209968386869e+36] = np.nan
        eastward_wind_temp = np.float16(eastward_wind_temp)
        eastward_wind_temp = np.swapaxes(np.swapaxes(eastward_wind_temp, 0, 2), 0, 1)
        time_temp = np.array(fh['time'])
        #Converts to datetime
        time_date_temp = np.empty((len(time_temp)), dtype=object)
        for i in range(0, len(time_temp)):
            time_date_temp[i] = serial_date_to_string(int(time_temp[i]))
        
        northward_wind = np.dstack((northward_wind, northward_wind_temp))
        eastward_wind = np.dstack((eastward_wind, eastward_wind_temp))
        time_date = np.hstack((time_date, time_date_temp))
        print(time_date_temp[0])

np.savez_compressed('winds_19921996', northward_wind=northward_wind, #seaice=seaice,
                    time_date=time_date, eastward_wind=eastward_wind,
                    lat=lat, lon=lon)