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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\par\\par\\')
### Load and join data
files = os.listdir()
for i in files:
    print(i)
    fh = np.load(i, allow_pickle=True)
    if i == files[0]:
        lat = fh['lat']
        lon = fh['lon']
        par = fh['par']
        time_date = fh['time_date']
    else:
        par_temp = fh['par']
        time_date_temp = fh['time_date']
        par = np.dstack((par, par_temp))
        time_date = np.hstack((time_date, time_date_temp))
       
np.savez_compressed('par_19972021', par=par,
                    time_date=time_date,
                    lat=lat, lon=lon)