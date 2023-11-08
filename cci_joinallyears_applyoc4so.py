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
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\cciv6data\\')
### Load and join data
files = os.listdir()
for i in files:
    print(i)
    fh = np.load(i, allow_pickle=True)
    if i == files[0]:
        lat = fh['lat']
        lon = fh['lon']
        chl_ori = fh['chl']
        rrs443 = fh['rrs443']   
        rrs443[rrs443 == 9.96921E36] = np.nan
        rrs490 = fh['rrs490']     
        rrs490[rrs490 == 9.96921E36] = np.nan
        rrs510 = fh['rrs510']    
        rrs510[rrs510 == 9.96921E36] = np.nan
        rrs560 = fh['rrs560']
        rrs560[rrs560 == 9.96921E36] = np.nan
        time_date = fh['time_date']
        mbr_19982020 = np.empty_like(rrs443)*np.nan
        chl_oc4so = np.empty_like(mbr_19982020)*np.nan
        for t in tqdm(range(0, len(time_date))):
            rrs_443_temp = rrs443[:,:, t]
            rrs_490_temp = rrs490[:,:, t]
            rrs_510_temp = rrs510[:,:, t]
            rrs_560_temp = rrs560[:,:, t]    
            for k in range(0, len(lat)):
                for j in range(0, len(lon)):
                    if np.isnan(rrs_443_temp[k, j]):
                        mbr_19982020[k, j, t] = np.nan
                        continue
                    else:
                        mbr_19982020[k, j, t]  = (np.nanmax([rrs_443_temp[k, j], rrs_490_temp[k, j], rrs_510_temp[k, j]]))/rrs_560_temp[k, j]
            chl_oc4so[:,:, t] = oc4so_2D(mbr_19982020[:, :, t])        
    else:
        chl_ori_temp = fh['chl']
        rrs443_temp = fh['rrs443']
        rrs443_temp[rrs443_temp == 9.96921E36] = np.nan
        rrs490_temp = fh['rrs490']
        rrs490_temp[rrs490_temp == 9.96921E36] = np.nan
        rrs510_temp = fh['rrs510']
        rrs510_temp[rrs510_temp == 9.96921E36] = np.nan
        rrs560_temp = fh['rrs560']
        rrs560_temp[rrs560_temp == 9.96921E36] = np.nan
        time_date_temp = fh['time_date']
        mbr_19982020_temp = np.empty_like(rrs443_temp)*np.nan
        chl_oc4so_temp = np.empty_like(mbr_19982020_temp)*np.nan
        for t in tqdm(range(0, len(time_date_temp))):
            rrs443_temp_temp = rrs443_temp[:,:, t]
            rrs490_temp_temp = rrs490_temp[:,:, t]
            rrs510_temp_temp = rrs510_temp[:,:, t]
            rrs560_temp_temp = rrs560_temp[:,:, t]    
            for k in range(0, len(lat)):
                for j in range(0, len(lon)):
                    if np.isnan(rrs443_temp_temp[k, j]):
                        mbr_19982020_temp[k, j, t] = np.nan
                        continue
                    else:
                        mbr_19982020_temp[k, j, t]  = (np.nanmax([rrs443_temp_temp[k, j], rrs490_temp_temp[k, j], rrs510_temp_temp[k, j]]))/rrs560_temp_temp[k, j]
            chl_oc4so_temp[:,:, t] = oc4so_2D(mbr_19982020_temp[:, :, t])     
        chl_ori = np.dstack((chl_ori, chl_ori_temp))
        chl_oc4so = np.dstack((chl_oc4so, chl_oc4so_temp))
        time_date = np.hstack((time_date, time_date_temp))

chl_ori[chl_ori>100] = 100
chl_oc4so[chl_oc4so>100] = 100
np.savez_compressed('chloc4so_19972022', chl_ori=chl_ori, chl_oc4so=chl_oc4so,
                    time_date=time_date,
                    lat=lat, lon=lon)