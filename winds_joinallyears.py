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
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\winds\\')
### Load and join data
files = os.listdir()
for i in files:
    print(i)
    fh = Dataset(i, mode='r')
    if i == files[0]:
        lat = np.array(fh['latitude'])
        lon = np.array(fh['longitude'])
        wind_u = np.array(fh['u10'])
        wind_u[wind_u == -32767] = np.nan
        wind_u = np.swapaxes(np.swapaxes(wind_u, 0, 2), 0, 1)
        wind_v = np.array(fh['v10'])
        wind_v[wind_v == -32767] = np.nan
        wind_v = np.swapaxes(np.swapaxes(wind_v, 0, 2), 0, 1)        
        time = np.array(fh['time'])
        #Converts to datetime
        time_date = np.empty((len(time)), dtype=object)
        for i in range(0, len(time)):
            time_date[i] = serial_date_to_string(int(time[i]))
    else:
        wind_u_temp = np.array(fh['u10'])
        wind_u_temp[wind_u_temp == -32767] = np.nan
        wind_u_temp = np.swapaxes(np.swapaxes(wind_u_temp, 0, 2), 0, 1)
        wind_v_temp = np.array(fh['v10'])
        wind_v_temp[wind_v_temp == -32767] = np.nan
        wind_v_temp = np.swapaxes(np.swapaxes(wind_v_temp, 0, 2), 0, 1)   
        time_temp = np.array(fh['time'])
        #Converts to datetime
        time_date_temp = np.empty((len(time_temp)), dtype=object)
        for i in range(0, len(time_temp)):
            time_date_temp[i] = serial_date_to_string(int(time_temp[i]))
        # Stack
        wind_u = np.dstack((wind_u, wind_u_temp))
        wind_v = np.dstack((wind_v, wind_v_temp))
        time_date = np.hstack((time_date, time_date_temp))
#        print(time_date_temp[0])

np.savez_compressed('winds_19972022_era5', #sst=sst, #seaice=seaice,
                    time_date=time_date, wind_u=wind_u, wind_v=wind_v,
                    lat=lat, lon=lon)