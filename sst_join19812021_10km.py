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
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia\\')
### Load 1981-1996
fh = np.load('sst_19811996_10km.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
sst_1 = fh['sst']
time_date_1 = fh['time_date']
### Load 1997-2021
fh = np.load('sst_19972021_10km.npz', allow_pickle=True)
sst_2 = fh['sst']
time_date_2 = fh['time_date']
## Join data
sst = np.dstack((sst_1, sst_2))
time_date = np.hstack((time_date_1, time_date_2))
np.savez_compressed('sst_19812021_10km', #sst=sst, #seaice=seaice,
                    time_date=time_date, sst=sst,
                    lat=lat, lon=lon)

