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
    new_date = datetime.datetime(1601, 1, 1, 0, 0) + datetime.timedelta(days=srl_no)
    return new_date
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\seaice_noaansidc\\')
### Load and join data
files = os.listdir()
for i in files:
    print(i)
    fh = Dataset(i, mode='r')
    if i == files[0]:
        ygrid = np.array(fh['ygrid'])
        xgrid = np.array(fh['xgrid'])
        lat = np.array(fh['latitude'])
        lon = np.array(fh['longitude'])        
        seaice = np.array(fh['cdr_seaice_conc'])
        seaice[seaice < 0] = np.nan
        seaice[seaice == 255] = np.nan
        seaice = np.swapaxes(np.swapaxes(seaice, 0, 2), 0, 1)
        time = np.array(fh['time'])
        #Converts to datetime
        time_date = np.empty((len(time)), dtype=object)
        for i in range(0, len(time)):
            time_date[i] = serial_date_to_string(int(time[i]))
    else:
        seaice_temp = np.array(fh['cdr_seaice_conc'])
        seaice_temp[seaice_temp < 0] = np.nan
        seaice_temp[seaice_temp == 255] = np.nan
        seaice_temp = np.swapaxes(np.swapaxes(seaice_temp, 0, 2), 0, 1)
        time_temp = np.array(fh['time'])
        #Converts to datetime
        time_date_temp = np.empty((len(time_temp)), dtype=object)
        for i in range(0, len(time_temp)):
            time_date_temp[i] = serial_date_to_string(int(time_temp[i]))
        seaice = np.dstack((seaice, seaice_temp))
        time_date = np.hstack((time_date, time_date_temp))
        print(time_date_temp[0])

np.savez_compressed('seaice_19782023', time_date=time_date, seaice=seaice,
                    lat=lat, lon=lon, ygrid=ygrid, xgrid=xgrid)