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
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
### Load and join data
files = os.listdir()
for i in files:
    print(i)
    fh = Dataset(i, mode='r')
    if i == files[0]:
        lat = np.array(fh['lat'])
        lon = np.array(fh['lon'])
        sst = np.array(fh['analysed_sst'])
        sst[sst == -32768] = np.nan
        sst = sst-273.15
        sst = np.float16(sst)
        sst = np.swapaxes(np.swapaxes(sst, 0, 2), 0, 1)
        seaice = np.array(fh['sea_ice_fraction'])
        seaice[seaice == -128] = np.nan
        #seaice = np.float16(seaice)
        seaice = np.swapaxes(np.swapaxes(seaice, 0, 2), 0, 1)
        time = np.array(fh['time'])
        #Converts to datetime
        time_date = np.empty((len(time)), dtype=object)
        for i in range(0, len(time)):
            time_date[i] = serial_date_to_string(int(time[i]))
    else:
        sst_temp = np.array(fh['analysed_sst'])
        sst_temp[sst_temp == -32768] = np.nan
        sst_temp = sst_temp-273.15
        #sst_temp = np.float16(sst_temp)
        sst_temp = np.swapaxes(np.swapaxes(sst_temp, 0, 2), 0, 1)
        seaice_temp = np.array(fh['sea_ice_fraction'])
        seaice_temp[seaice_temp == -128] = np.nan
        #seaice_temp = np.float16(seaice_temp)
        seaice_temp = np.swapaxes(np.swapaxes(seaice_temp, 0, 2), 0, 1)
        time_temp = np.array(fh['time'])
        #Converts to datetime
        time_date_temp = np.empty((len(time_temp)), dtype=object)
        for i in range(0, len(time_temp)):
            time_date_temp[i] = serial_date_to_string(int(time_temp[i]))
        
        sst = np.dstack((sst, sst_temp))
        seaice = np.dstack((seaice, seaice_temp))
        time_date = np.hstack((time_date, time_date_temp))
        print(time_date_temp[0])

np.savez_compressed('sst-seaice_19972021', #sst=sst, #seaice=seaice,
                    time_date=time_date, seaice=seaice, sst=sst,
                    lat=lat, lon=lon)