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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\par\\20202022')
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
        par = np.array(fh['PAR_mean'])
        par[par == -999.0] = np.nan
        par = np.float16(par)
        #par = np.swapaxes(np.swapaxes(par, 0, 2), 0, 1)
        time_date = datetime.datetime(year=int(i[4:8]), month=int(i[8:10]), day=int(i[10:12]))
    else:
        #sst_temp = np.array(fh['analysed_sst'])
        #sst_temp[sst_temp == -32768] = np.nan
        #sst_temp = sst_temp-273.15
        #sst_temp = np.float16(sst_temp)
        #sst_temp = np.swapaxes(np.swapaxes(sst_temp, 0, 2), 0, 1)
        par_temp = np.array(fh['PAR_mean'])
        par_temp[par_temp == -999.0] = np.nan
        par_temp = np.float16(par_temp)
        #par_temp = np.swapaxes(np.swapaxes(par_temp, 0, 2), 0, 1)
        #Converts to datetime
        time_date_temp = datetime.datetime(year=int(i[4:8]), month=int(i[8:10]), day=int(i[10:12]))
        #sst = np.dstack((sst, sst_temp))
        par = np.dstack((par, par_temp))
        time_date = np.hstack((time_date, time_date_temp))
        #print(time_date_temp)

np.savez_compressed('par_20202022', #sst=sst, #seaice=seaice,
                    time_date=time_date, par=par,
                    lat=lat, lon=lon)