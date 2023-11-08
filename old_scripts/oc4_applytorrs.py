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
#OC4-SO - poly4
xx = np.linspace(-5, 5, 1000)
a1_poly4 = 18.64112254
a2_poly4 = -26.78898243
a3_poly4 = 11.17267554
a4_poly4 = -3.20361818
a5_poly4 = 0.60158625
newsoalg_poly4 = a1_poly4*xx**4 + a2_poly4*xx**3 + a3_poly4*xx**2 + a4_poly4*xx + a5_poly4
def newsoalg_poly4_func(mbr_value):
    newsoalg_value = a1_poly4*mbr_value**4 + a2_poly4*mbr_value**3 + a3_poly4*mbr_value**2 + a4_poly4*mbr_value + a5_poly4
    return newsoalg_value
#OC4-SO - poly3
a1_poly3 = -0.57160031
a2_poly3 = 0.15706528
a3_poly3 = -1.94561204
a4_poly3 = 0.6366794
newsoalg_poly3 = a1_poly3*xx**3 + a2_poly3*xx**2 + a3_poly3*xx + a4_poly3
def newsoalg_poly3_func(mbr_value):
    newsoalg_value = a1_poly3*mbr_value**3 + a2_poly3*mbr_value**2 + a3_poly3*mbr_value + a4_poly3
    return newsoalg_value
def oc4so_2D(mbr_value):
    chl_so_newalg = np.empty_like(mbr_value)*np.nan
    for x in range(0, np.size(mbr_value, 0)):
        for y in range(0, np.size(mbr_value, 1)):     
            if np.isnan(mbr_value[x, y]):
                continue
            else:
                if mbr_value[x, y] < 3:
                    chl_so = newsoalg_poly4_func(np.log10(mbr_value[x, y]))
                    chl_so_newalg[x, y] = 10**chl_so
                if mbr_value[x, y] >= 3 and mbr_value[x, y] <= 5:
                    poly3_weight = (mbr_value[x, y] - 3)/(5-3)
                    new_alg_weight = 1-poly3_weight
                    chl_so = newsoalg_poly4_func(np.log10(mbr_value[x, y]))*new_alg_weight + newsoalg_poly3_func(np.log10(mbr_value[x, y]))*poly3_weight
                    chl_so_newalg[x, y] = 10**chl_so
                if mbr_value[x, y] > 5:
                    chl_so = newsoalg_poly3_func(np.log10(mbr_value[x, y]))
                    chl_so_newalg[x, y] = 10**chl_so
    return chl_so_newalg
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\rrs\\20042009\\')
### Load data
fh = np.load('cci5_rrs_20042009_daily.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
#lat = lat[167:]
#lon = lon[71:793]
rrs_443 = fh['rrs_490']
rrs_443[rrs_443 == 9.96921E36] = np.nan
rrs_443 = np.float16(rrs_443)
rrs_490 = fh['rrs_490']
rrs_490[rrs_490 == 9.96921E36] = np.nan
rrs_490 = np.float16(rrs_490)
rrs_510 = fh['rrs_510']
rrs_510[rrs_510 == 9.96921E36] = np.nan
rrs_510 = np.float16(rrs_510)
rrs_560 = fh['rrs_560']
rrs_560[rrs_560 == 9.96921E36] = np.nan
rrs_560 = np.float16(rrs_560)
time_date = fh['time_date']
# Convert to OC4-SO
mbr_19982020 = np.empty((np.size(rrs_443,0), np.size(rrs_443,1), np.size(rrs_443,2)), dtype='float16')*np.nan
chl_oc4so = mbr_19982020
for i in tqdm(range(0, len(time_date))):
    rrs_443_temp = rrs_443[:,:, i]
    rrs_490_temp = rrs_490[:,:, i]
    rrs_510_temp = rrs_510[:,:, i]
    rrs_560_temp = rrs_560[:,:, i]    
    for k in range(0, len(lat)):
        for j in range(0, len(lon)):
            if np.isnan(rrs_443_temp[k, j]):
                mbr_19982020[k, j, i] = np.nan
                continue
            else:
                mbr_19982020[k, j, i]  = (np.nanmax([rrs_443_temp[k, j], rrs_490_temp[k, j], rrs_510_temp[k, j]]))/rrs_560_temp[k, j]
    chl_oc4so[:,:, i] = oc4so_2D(mbr_19982020[:, :, i])
### Save 8-day dataset
#np.savez_compressed('chloc4so_8day_short', chl=chl_oc4so, time_date=time_date, 
#                    lat=lat, lon=lon)
np.savez_compressed('chloc4so_20042009', chl=chl_oc4so, time_date=time_date, 
                    lat=lat, lon=lon)