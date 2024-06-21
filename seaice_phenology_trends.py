# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:33:06 2020

@author: Afonso
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.path import Path
from tqdm import tqdm
from scipy import integrate
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.patches as mpatches
import shap
import sklearn
from matplotlib import patches
import math
from netCDF4 import Dataset
import seaborn as sns
import cmocean
import netCDF4 as nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pyproj import CRS
from pyproj import Transformer

def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
def serial_date_to_string_mld(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1950, 1, 1, 0, 0) + datetime.timedelta(hours=srl_no)
    return new_date
def serial_date_to_string_ssh(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1950, 1, 1, 0, 0) + datetime.timedelta(days=srl_no)
    return new_date
def serial_date_to_string_ugos(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1950, 1, 1, 0, 0) + datetime.timedelta(hours=srl_no)
    return new_date
def serial_date_to_string_winds(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1900, 1, 1, 0, 0) + datetime.timedelta(hours=srl_no)
    return new_date
def check_for_bloominit(yearly_timeseries):
    arr = yearly_timeseries.values.copy()                   # avoid mutating the original list
    counting = []                      # keep track of True indexes, to count them later
    for i in range(len(arr)):          # cycle by index
        is_last = i + 1 >= len(arr)    # True if this is the last index in the array
        if arr[i] == True:
            counting.append(i)         # add value to list if True
        if is_last or arr[i] == False: # when we are at the last entry, or find a False
            if len(counting) < 2:      # check the length of our True indexes, and if less than 6
                for j in counting:
                    arr[j] = False     # make each False
            counting = []
    return arr
#%% Load sea ice timings - LTR3 indices
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\sst-seaice\\ostia\\ostiaseaice_19822022\\')
fh = np.load('ostia_seaice_19822022.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
seaice = fh['seaice']
time_date = fh['time_date']
time_date_seaice_year = np.empty_like(time_date)
time_date_seaice_month = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_seaice_year[i] = time_date[i].year
    time_date_seaice_month[i] = time_date[i].month
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('clusters_upscaled_sstseaice.npz',allow_pickle = True)
clusters_seaice = fh['clusters']
clusters_seaice = clusters_seaice[1:,:]
## SST
dra_seaice = seaice[clusters_seaice == 3,:]
#%% Calculate sea ice extent for each day
for year in np.arange(1982, 2022):
    # Separate full year:
    dra_seaice_year = dra_seaice[:, time_date_seaice_year == year]
    dra_seaice_yeardates = time_date[time_date_seaice_year == year]
    # Calculate sea ice extent for each day
    for i in np.arange(0, len(dra_seaice_yeardates)):
        dra_seaiceextent_year_temp = np.sum(dra_seaice_year[:,i] >= 0.15) * 25
        dra_seaiceextent_year_percentage_temp = (np.sum(dra_seaice_year[:,i] >= 0.15)/len(dra_seaice_year[:,i]))*100
        if i == 0:
            dra_seaiceextent_year = dra_seaiceextent_year_temp
            dra_seaiceextent_percentage_year = dra_seaiceextent_year_percentage_temp
        else:
            dra_seaiceextent_year = np.hstack((dra_seaiceextent_year, dra_seaiceextent_year_temp))
            dra_seaiceextent_percentage_year = np.hstack((dra_seaiceextent_percentage_year, dra_seaiceextent_year_percentage_temp))
    # remove 29th february
    if len(dra_seaiceextent_year) == 366:
        dra_seaiceextent_year = np.delete(dra_seaiceextent_year, 59)
        dra_seaiceextent_percentage_year = np.delete(dra_seaiceextent_percentage_year, 59)
    # join for each year
    if year == 1982:
        dra_seaiceextent = dra_seaiceextent_year
        dra_seaiceextent_percentage = dra_seaiceextent_percentage_year
    else:
        dra_seaiceextent = np.vstack((dra_seaiceextent, dra_seaiceextent_year))
        dra_seaiceextent_percentage = np.vstack((dra_seaiceextent_percentage, dra_seaiceextent_percentage_year))        
#%% Calculate annual mean sea extent
allyears = np.arange(1982, 2022)
dra_seaiceextent_annual_19822021 = np.nanmean(dra_seaiceextent, 1)
# Calculate Spring (September-November) sea ice extent
start_date = datetime.datetime(1998, 1, 1)
end_date   = datetime.datetime(1999, 1, 1)
dates_1998 = np.array([start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days))])
for i in np.arange(0, len(allyears)):
    dra_seaiceextent_yearly = dra_seaiceextent[i, :] 
    dra_seaiceextent_yearly_df = pd.DataFrame(data=dra_seaiceextent_yearly, index=dates_1998)
    # Extract Spring (September to November)
    months_inds = dra_seaiceextent_yearly_df.index.month
    dra_seaiceextent_sep_temp = dra_seaiceextent_yearly[months_inds == 9] #Sep
    dra_seaiceextent_oct_temp = dra_seaiceextent_yearly[months_inds == 10] #Oct
    dra_seaiceextent_nov_temp = dra_seaiceextent_yearly[months_inds == 11] #Nov
    # Extract Summer (December to February)
    dra_seaiceextent_dec_temp = dra_seaiceextent_yearly[months_inds == 12] #Dec
    dra_seaiceextent_jan_temp = dra_seaiceextent_yearly[months_inds == 1] #Jan
    dra_seaiceextent_feb_temp = dra_seaiceextent_yearly[months_inds == 2] #Feb
    # Extract Autumn (March to May)
    dra_seaiceextent_mar_temp = dra_seaiceextent_yearly[months_inds == 3] #Mar
    dra_seaiceextent_apr_temp = dra_seaiceextent_yearly[months_inds == 4] #Apr
    dra_seaiceextent_may_temp = dra_seaiceextent_yearly[months_inds == 5] #May
    # Average and join all together
    if i == 0:
        dra_seaiceextent_spring = np.nanmean(np.hstack((dra_seaiceextent_sep_temp, dra_seaiceextent_oct_temp, dra_seaiceextent_nov_temp)))
        dra_seaiceextent_summer = np.nanmean(np.hstack((dra_seaiceextent_dec_temp, dra_seaiceextent_jan_temp, dra_seaiceextent_feb_temp)))
        dra_seaiceextent_autumn = np.nanmean(np.hstack((dra_seaiceextent_mar_temp, dra_seaiceextent_apr_temp, dra_seaiceextent_may_temp)))
    else:
        dra_seaiceextent_spring = np.vstack((dra_seaiceextent_spring, np.nanmean(np.hstack((dra_seaiceextent_sep_temp, dra_seaiceextent_oct_temp, dra_seaiceextent_nov_temp)))))    
        dra_seaiceextent_summer = np.vstack((dra_seaiceextent_summer, np.nanmean(np.hstack((dra_seaiceextent_dec_temp, dra_seaiceextent_jan_temp, dra_seaiceextent_feb_temp)))))    
        dra_seaiceextent_autumn = np.vstack((dra_seaiceextent_autumn, np.nanmean(np.hstack((dra_seaiceextent_mar_temp, dra_seaiceextent_apr_temp, dra_seaiceextent_may_temp)))))    
#%%
plt.scatter(np.arange(1982, 2022), dra_seaiceextent_spring)
plt.scatter(np.arange(1982, 2022), dra_seaiceextent_summer)
plt.scatter(np.arange(1982, 2022), dra_seaiceextent_autumn)

stats.linregress(allyears, dra_seaiceextent_spring.ravel())
stats.linregress(allyears, dra_seaiceextent_summer.ravel())
stats.linregress(allyears, dra_seaiceextent_autumn.ravel())
#%% Plot seasonal cycle comparing prior to 1998 and after it
plt.plot(np.nanmean(dra_seaiceextent[:16], axis=0))
plt.plot(np.nanmean(dra_seaiceextent[16:], axis=0))
#%% Now for BRS
brs_seaice = seaice[clusters_seaice == 4,:]
for year in np.arange(1982, 2022):
    # Separate full year:
    brs_seaice_year = brs_seaice[:, time_date_seaice_year == year]
    brs_seaice_yeardates = time_date[time_date_seaice_year == year]
    # Calculate sea ice extent for each day
    for i in np.arange(0, len(brs_seaice_yeardates)):
        brs_seaiceextent_year_temp = np.sum(brs_seaice_year[:,i] >= 0.15) * 25
        brs_seaiceextent_year_percentage_temp = (np.sum(brs_seaice_year[:,i] >= 0.15)/len(brs_seaice_year[:,i]))*100
        if i == 0:
            brs_seaiceextent_year = brs_seaiceextent_year_temp
            brs_seaiceextent_percentage_year = brs_seaiceextent_year_percentage_temp
        else:
            brs_seaiceextent_year = np.hstack((brs_seaiceextent_year, brs_seaiceextent_year_temp))
            brs_seaiceextent_percentage_year = np.hstack((brs_seaiceextent_percentage_year, brs_seaiceextent_year_percentage_temp))
    # remove 29th february
    if len(brs_seaiceextent_year) == 366:
        brs_seaiceextent_year = np.delete(brs_seaiceextent_year, 59)
        brs_seaiceextent_percentage_year = np.delete(brs_seaiceextent_percentage_year, 59)
    # join for each year
    if year == 1982:
        brs_seaiceextent = brs_seaiceextent_year
        brs_seaiceextent_percentage = brs_seaiceextent_percentage_year
    else:
        brs_seaiceextent = np.vstack((brs_seaiceextent, brs_seaiceextent_year))
        brs_seaiceextent_percentage = np.vstack((brs_seaiceextent_percentage, brs_seaiceextent_percentage_year))        
#%% Calculate annual mean sea extent
allyears = np.arange(1982, 2022)
brs_seaiceextent_annual_19822021 = np.nanmean(brs_seaiceextent, 1)
# Calculate Spring (September-November) sea ice extent
start_date = datetime.datetime(1998, 1, 1)
end_date   = datetime.datetime(1999, 1, 1)
dates_1998 = np.array([start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days))])
for i in np.arange(0, len(allyears)):
    brs_seaiceextent_yearly = brs_seaiceextent[i, :] 
    brs_seaiceextent_yearly_df = pd.DataFrame(data=brs_seaiceextent_yearly, index=dates_1998)
    # Extract Spring (September to November)
    months_inds = brs_seaiceextent_yearly_df.index.month
    brs_seaiceextent_sep_temp = brs_seaiceextent_yearly[months_inds == 9] #Sep
    brs_seaiceextent_oct_temp = brs_seaiceextent_yearly[months_inds == 10] #Oct
    brs_seaiceextent_nov_temp = brs_seaiceextent_yearly[months_inds == 11] #Nov
    # Extract Summer (December to February)
    brs_seaiceextent_dec_temp = brs_seaiceextent_yearly[months_inds == 12] #Dec
    brs_seaiceextent_jan_temp = brs_seaiceextent_yearly[months_inds == 1] #Jan
    brs_seaiceextent_feb_temp = brs_seaiceextent_yearly[months_inds == 2] #Feb
    # Extract Autumn (March to May)
    brs_seaiceextent_mar_temp = brs_seaiceextent_yearly[months_inds == 3] #Mar
    brs_seaiceextent_apr_temp = brs_seaiceextent_yearly[months_inds == 4] #Apr
    brs_seaiceextent_may_temp = brs_seaiceextent_yearly[months_inds == 5] #May
    # Average and join all together
    if i == 0:
        brs_seaiceextent_spring = np.nanmean(np.hstack((brs_seaiceextent_sep_temp, brs_seaiceextent_oct_temp, brs_seaiceextent_nov_temp)))
        brs_seaiceextent_summer = np.nanmean(np.hstack((brs_seaiceextent_dec_temp, brs_seaiceextent_jan_temp, brs_seaiceextent_feb_temp)))
        brs_seaiceextent_autumn = np.nanmean(np.hstack((brs_seaiceextent_mar_temp, brs_seaiceextent_apr_temp, brs_seaiceextent_may_temp)))
    else:
        brs_seaiceextent_spring = np.vstack((brs_seaiceextent_spring, np.nanmean(np.hstack((brs_seaiceextent_sep_temp, brs_seaiceextent_oct_temp, brs_seaiceextent_nov_temp)))))    
        brs_seaiceextent_summer = np.vstack((brs_seaiceextent_summer, np.nanmean(np.hstack((brs_seaiceextent_dec_temp, brs_seaiceextent_jan_temp, brs_seaiceextent_feb_temp)))))    
        brs_seaiceextent_autumn = np.vstack((brs_seaiceextent_autumn, np.nanmean(np.hstack((brs_seaiceextent_mar_temp, brs_seaiceextent_apr_temp, brs_seaiceextent_may_temp)))))    
#%%
plt.scatter(np.arange(1982, 2022), brs_seaiceextent_spring)
plt.scatter(np.arange(1982, 2022), brs_seaiceextent_summer)
plt.scatter(np.arange(1982, 2022), brs_seaiceextent_autumn)

stats.linregress(allyears, brs_seaiceextent_spring.ravel())
stats.linregress(allyears, brs_seaiceextent_summer.ravel())
stats.linregress(allyears, brs_seaiceextent_autumn.ravel())
#%% Plot seasonal cycle comparing prior to 1998 and after it
plt.plot(np.nanmean(brs_seaiceextent[:16], axis=0))
plt.plot(np.nanmean(brs_seaiceextent[16:], axis=0))
#%% Now for GES
ges_seaice = seaice[clusters_seaice == 2,:]
for year in np.arange(1982, 2022):
    # Separate full year:
    ges_seaice_year = ges_seaice[:, time_date_seaice_year == year]
    ges_seaice_yeardates = time_date[time_date_seaice_year == year]
    # Calculate sea ice extent for each day
    for i in np.arange(0, len(ges_seaice_yeardates)):
        ges_seaiceextent_year_temp = np.sum(ges_seaice_year[:,i] >= 0.15) * 25
        ges_seaiceextent_year_percentage_temp = (np.sum(ges_seaice_year[:,i] >= 0.15)/len(ges_seaice_year[:,i]))*100
        if i == 0:
            ges_seaiceextent_year = ges_seaiceextent_year_temp
            ges_seaiceextent_percentage_year = ges_seaiceextent_year_percentage_temp
        else:
            ges_seaiceextent_year = np.hstack((ges_seaiceextent_year, ges_seaiceextent_year_temp))
            ges_seaiceextent_percentage_year = np.hstack((ges_seaiceextent_percentage_year, ges_seaiceextent_year_percentage_temp))
    # remove 29th february
    if len(ges_seaiceextent_year) == 366:
        ges_seaiceextent_year = np.delete(ges_seaiceextent_year, 59)
        ges_seaiceextent_percentage_year = np.delete(ges_seaiceextent_percentage_year, 59)
    # join for each year
    if year == 1982:
        ges_seaiceextent = ges_seaiceextent_year
        ges_seaiceextent_percentage = ges_seaiceextent_percentage_year
    else:
        ges_seaiceextent = np.vstack((ges_seaiceextent, ges_seaiceextent_year))
        ges_seaiceextent_percentage = np.vstack((ges_seaiceextent_percentage, ges_seaiceextent_percentage_year))        
#%% Calculate annual mean sea extent
allyears = np.arange(1982, 2022)
ges_seaiceextent_annual_19822021 = np.nanmean(ges_seaiceextent, 1)
# Calculate Spring (September-November) sea ice extent
start_date = datetime.datetime(1998, 1, 1)
end_date   = datetime.datetime(1999, 1, 1)
dates_1998 = np.array([start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days))])
for i in np.arange(0, len(allyears)):
    ges_seaiceextent_yearly = ges_seaiceextent[i, :] 
    ges_seaiceextent_yearly_df = pd.DataFrame(data=ges_seaiceextent_yearly, index=dates_1998)
    # Extract Spring (September to November)
    months_inds = ges_seaiceextent_yearly_df.index.month
    ges_seaiceextent_sep_temp = ges_seaiceextent_yearly[months_inds == 9] #Sep
    ges_seaiceextent_oct_temp = ges_seaiceextent_yearly[months_inds == 10] #Oct
    ges_seaiceextent_nov_temp = ges_seaiceextent_yearly[months_inds == 11] #Nov
    # Extract Summer (December to February)
    ges_seaiceextent_dec_temp = ges_seaiceextent_yearly[months_inds == 12] #Dec
    ges_seaiceextent_jan_temp = ges_seaiceextent_yearly[months_inds == 1] #Jan
    ges_seaiceextent_feb_temp = ges_seaiceextent_yearly[months_inds == 2] #Feb
    # Extract Autumn (March to May)
    ges_seaiceextent_mar_temp = ges_seaiceextent_yearly[months_inds == 3] #Mar
    ges_seaiceextent_apr_temp = ges_seaiceextent_yearly[months_inds == 4] #Apr
    ges_seaiceextent_may_temp = ges_seaiceextent_yearly[months_inds == 5] #May
    # Average and join all together
    if i == 0:
        ges_seaiceextent_spring = np.nanmean(np.hstack((ges_seaiceextent_sep_temp, ges_seaiceextent_oct_temp, ges_seaiceextent_nov_temp)))
        ges_seaiceextent_summer = np.nanmean(np.hstack((ges_seaiceextent_dec_temp, ges_seaiceextent_jan_temp, ges_seaiceextent_feb_temp)))
        ges_seaiceextent_autumn = np.nanmean(np.hstack((ges_seaiceextent_mar_temp, ges_seaiceextent_apr_temp, ges_seaiceextent_may_temp)))
    else:
        ges_seaiceextent_spring = np.vstack((ges_seaiceextent_spring, np.nanmean(np.hstack((ges_seaiceextent_sep_temp, ges_seaiceextent_oct_temp, ges_seaiceextent_nov_temp)))))    
        ges_seaiceextent_summer = np.vstack((ges_seaiceextent_summer, np.nanmean(np.hstack((ges_seaiceextent_dec_temp, ges_seaiceextent_jan_temp, ges_seaiceextent_feb_temp)))))    
        ges_seaiceextent_autumn = np.vstack((ges_seaiceextent_autumn, np.nanmean(np.hstack((ges_seaiceextent_mar_temp, ges_seaiceextent_apr_temp, ges_seaiceextent_may_temp)))))    
#%%
plt.scatter(np.arange(1982, 2022), ges_seaiceextent_spring)
plt.scatter(np.arange(1982, 2022), ges_seaiceextent_summer)
plt.scatter(np.arange(1982, 2022), ges_seaiceextent_autumn)

stats.linregress(allyears, ges_seaiceextent_spring.ravel())
stats.linregress(allyears, ges_seaiceextent_summer.ravel())
stats.linregress(allyears, ges_seaiceextent_autumn.ravel())
#%% Plot seasonal cycle comparing prior to 1998 and after it
plt.plot(np.nanmean(ges_seaiceextent[:16], axis=0))
plt.plot(np.nanmean(ges_seaiceextent[16:], axis=0))

#%% Now calculate the first day and last day above 15% ice
# DRA
allyears[1]
plt.plot(dra_seaiceextent_percentage[7,:])
dates_1998[dra_seaiceextent_percentage[3,:] > 15][-1]

datetime.datetime(1998, 7, 5, 0, 0) - datetime.datetime(1998, 11, 26, 0, 0) # 1982
datetime.datetime(1998, 8, 13, 0, 0) - datetime.datetime(1998, 10, 8, 0, 0) # 1983
datetime.datetime(1998, 6, 26, 0, 0) - datetime.datetime(1998, 9, 1, 0, 0) # 1984

#%% DRA
# join all together
dra_seaiceextent_percentage_allyears = dra_seaiceextent_percentage.ravel()


#plt.plot(dra_seaiceextent_percentage[40, :])


for i in range(0, len(allyears)):    
    if sum(dra_seaiceextent_percentage[i,:] > 15) < 10:
        dra_seaiceretreat_temp = np.nan
        dra_seaiceadvance_temp = np.nan
    else:
        dra_seaiceretreat_temp = dates_1998[dra_seaiceextent_percentage[i,:] > 15][-1].timetuple().tm_yday
        dra_seaiceadvance_temp = dates_1998[dra_seaiceextent_percentage[i,:] > 15][0].timetuple().tm_yday
    if i == 0:
        dra_seaiceretreat = dra_seaiceretreat_temp
        dra_seaiceadvance = dra_seaiceadvance_temp
    else:
        dra_seaiceretreat = np.hstack((dra_seaiceretreat, dra_seaiceretreat_temp))
        dra_seaiceadvance = np.hstack((dra_seaiceadvance, dra_seaiceadvance_temp))
plt.plot(dra_seaiceretreat)
plt.plot(dra_seaiceadvance)
plt.scatter(np.arange(1982, 2022), dra_seaiceretreat)
plt.scatter(np.arange(1982, 2022), dra_seaiceadvance)
#%% BRS
#brs_seaiceextent_percentage_allyears = brs_seaiceextent_percentage.ravel()
#brs_dates_all = np.tile(dates_1998, 40)
#brs_seaiceextent_percentage_allyears_df = pd.DataFrame(data=brs_seaiceextent_percentage_allyears, index=brs_dates_all)
#plt.plot(brs_seaiceextent_percentage_allyears_df.values[11680:12045])
#a = brs_seaiceextent_percentage_allyears_df[11680:12045]
#plt.plot(brs_seaiceextent_percentage[39,:])
for i in range(0, len(allyears)):
    if sum(brs_seaiceextent_percentage[i,:] > 15) < 10:
        brs_seaiceretreat_temp = np.nan
        brs_seaiceadvance_temp = np.nan
    else:
        brs_seaiceretreat_temp = dates_1998[brs_seaiceextent_percentage[i,:] > 15][-1].timetuple().tm_yday
        brs_seaiceadvance_temp = dates_1998[brs_seaiceextent_percentage[i,:] > 15][0].timetuple().tm_yday
    if i == 0:
        brs_seaiceretreat = brs_seaiceretreat_temp
        brs_seaiceadvance = brs_seaiceadvance_temp
    else:
        brs_seaiceretreat = np.hstack((brs_seaiceretreat, brs_seaiceretreat_temp))
        brs_seaiceadvance = np.hstack((brs_seaiceadvance, brs_seaiceadvance_temp))
# MANUAL
brs_seaiceretreat[4] = 8
brs_seaiceadvance[5] = 102
brs_seaiceadvance[22] = 120
brs_seaiceretreat[30] = 27
brs_seaiceadvance[31] = 133
brs_seaiceretreat[31] = 4
brs_seaiceadvance[32] = 144
#
plt.plot(brs_seaiceretreat)
plt.plot(brs_seaiceadvance)
plt.scatter(np.arange(1982, 2022), brs_seaiceretreat)
plt.scatter(np.arange(1982, 2022), brs_seaiceadvance)
#%% GES
#ges_seaiceextent_percentage_allyears = ges_seaiceextent_percentage.ravel()
#ges_dates_all = np.tile(dates_1998, 40)
#ges_seaiceextent_percentage_allyears_df = pd.DataFrame(data=ges_seaiceextent_percentage_allyears, index=ges_dates_all)
#plt.plot(ges_seaiceextent_percentage_allyears_df.values[2800:3285])
#a = ges_seaiceextent_percentage_allyears_df[2800:3285]
#plt.plot(ges_seaiceextent_percentage[40,:])
#plt.axhline(y=15, c='k')
for i in range(0, len(allyears)):
    if sum(ges_seaiceextent_percentage[i,:] > 15) < 10:
        ges_seaiceretreat_temp = np.nan
        ges_seaiceadvance_temp = np.nan
    else:
        ges_seaiceretreat_temp = dates_1998[ges_seaiceextent_percentage[i,:] < 15][0].timetuple().tm_yday
        ges_seaiceadvance_temp = dates_1998[ges_seaiceextent_percentage[i,:] < 15][-1].timetuple().tm_yday
    if i == 0:
        ges_seaiceretreat = ges_seaiceretreat_temp
        ges_seaiceadvance = ges_seaiceadvance_temp
    else:
        ges_seaiceretreat = np.hstack((ges_seaiceretreat, ges_seaiceretreat_temp))
        ges_seaiceadvance = np.hstack((ges_seaiceadvance, ges_seaiceadvance_temp))
# MANUAL
ges_seaiceadvance[1] = 172
ges_seaiceretreat[2] = 14
ges_seaiceadvance[2] = 155
ges_seaiceretreat[3] = 365 # 31Dec
ges_seaiceadvance[6] = 177
ges_seaiceadvance[7] = 202
ges_seaiceretreat[7] = 349 
ges_seaiceretreat[8] = 302
ges_seaiceadvance[8] = 209 
ges_seaiceretreat[9] = 356 
ges_seaiceadvance[10] = 139 
ges_seaiceretreat[11] = 2 
ges_seaiceretreat[12] = 24
ges_seaiceadvance[12] = 154 
ges_seaiceretreat[13] = 5
ges_seaiceadvance[15] = 138 
ges_seaiceretreat[16] = 362
ges_seaiceadvance[16] = 155 
ges_seaiceretreat[17] = 356
ges_seaiceadvance[18] = 172
ges_seaiceretreat[19] = 324
ges_seaiceadvance[20] = 138
ges_seaiceretreat[21] = 348
ges_seaiceadvance[21] = 199
ges_seaiceretreat[22] = 354
ges_seaiceadvance[24] = 191
ges_seaiceretreat[25] = 359
ges_seaiceadvance[25] = 186
ges_seaiceretreat[26] = 345
ges_seaiceadvance[26] = 218
ges_seaiceadvance[27] = 172
ges_seaiceretreat[28] = 339
ges_seaiceadvance[28] = 168
ges_seaiceretreat[29] = 353
ges_seaiceadvance[30] = 160
ges_seaiceretreat[31] = 355



plt.plot(ges_seaiceretreat)
plt.plot(ges_seaiceadvance)
plt.scatter(np.arange(1982, 2022), ges_seaiceretreat)
plt.scatter(np.arange(1982, 2022), ges_seaiceadvance)
#%% Test Figure
#DRA
fig, axs = plt.subplots(3, figsize=((6,6)))
# 1982-1997
axs[0].plot(np.arange(1, 366), np.nanmean(dra_seaiceextent[:16], 0)/1000, c='#000080', linewidth=2,
            label='1982-1997 Mean')
# 1998-2022
axs[0].plot(np.arange(1, 366), np.nanmean(dra_seaiceextent[16:], 0)/1000, c='#BE0032', linewidth=2,
            label='1998-2022 Mean')
# Global Mean
axs[0].axvline(177, c='b', alpha=0.3, linestyle='-')
axs[0].axvline(187, c='r', alpha=0.3, linestyle='-')
axs[0].axvline(306, c='b', alpha=0.3, linestyle='--')
axs[0].axvline(280, c='r', alpha=0.3, linestyle='--')
#plt.xlabel('Days', fontsize=12)
#plt.ylabel('Sea Ice Extent (1000 km2)', fontsize=14)
axs[0].set_xticks(ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
               labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axs[0].set_xlim(1, 365)
axs[0].set_ylim(0, 160)
axs[0].legend(loc=0)
# BRS
# 1982-1997
axs[1].plot(np.arange(1, 366), np.nanmean(brs_seaiceextent[:16], 0)/1000, c='#000080', linewidth=2)
# 1998-2022
axs[1].plot(np.arange(1, 366), np.nanmean(brs_seaiceextent[16:], 0)/1000, c='#BE0032', linewidth=2)
# Global Mean
axs[1].axvline(116, c='b', alpha=0.3, linestyle='-')
axs[1].axvline(145, c='r', alpha=0.3, linestyle='-')
axs[1].axvline(320, c='b', alpha=0.3, linestyle='--')
axs[1].axvline(300, c='r', alpha=0.3, linestyle='--')
#plt.xlabel('Days', fontsize=12)
#plt.ylabel('Sea Ice Extent (1000 km2)', fontsize=14)
axs[1].set_xticks(ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
               labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axs[1].set_xlim(1, 365)
axs[1].set_ylim(0, 230)
# GES
# 1982-1997
axs[2].plot(np.arange(1, 366), np.nanmean(ges_seaiceextent[:16], 0)/1000, c='#000080', linewidth=2)
# 1998-2022
axs[2].plot(np.arange(1, 366), np.nanmean(ges_seaiceextent[16:], 0)/1000, c='#BE0032', linewidth=2)
# Global Mean
axs[2].axvline(141, c='b', alpha=0.3, linestyle='-')
axs[2].axvline(146, c='r', alpha=0.3, linestyle='-')
axs[2].axvline(33, c='b', alpha=0.3, linestyle='--')
axs[2].axvline(14, c='r', alpha=0.3, linestyle='--')
#plt.xlabel('Days', fontsize=12)
#plt.ylabel('Sea Ice Extent (1000 km2)', fontsize=14)
axs[2].set_xticks(ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
               labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axs[2].set_xlim(1, 365)
axs[1].set_ylabel('Sea Ice Extent (thousand km${^2}$)', fontsize=16)
axs[2].set_xlabel('Day of the Year', fontsize=16)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\new_Fig4_A.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Test Figure
#DRA
fig, axs = plt.subplots(3, figsize=((6,6)))
# 1982-1997
axs[0].plot(np.arange(1, 366), np.nanmean(dra_seaiceextent[19:29], 0)/1000, c='#000080', linewidth=2,
            label='1982-1997 Mean')
# 1998-2022
axs[0].plot(np.arange(1, 366), np.nanmean(dra_seaiceextent[29:-1], 0)/1000, c='#BE0032', linewidth=2,
            label='1998-2022 Mean')
# Global Mean
axs[0].axvline(177, c='b', alpha=0.3, linestyle='-')
axs[0].axvline(187, c='r', alpha=0.3, linestyle='-')
axs[0].axvline(306, c='b', alpha=0.3, linestyle='--')
axs[0].axvline(280, c='r', alpha=0.3, linestyle='--')
#plt.xlabel('Days', fontsize=12)
#plt.ylabel('Sea Ice Extent (1000 km2)', fontsize=14)
axs[0].set_xticks(ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
               labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axs[0].set_xlim(1, 365)
axs[0].set_ylim(0, 160)
axs[0].legend(loc=0)
# BRS
# 1982-1997
axs[1].plot(np.arange(1, 366), np.nanmean(brs_seaiceextent[19:29], 0)/1000, c='#000080', linewidth=2)
# 1998-2022
axs[1].plot(np.arange(1, 366), np.nanmean(brs_seaiceextent[29:-1], 0)/1000, c='#BE0032', linewidth=2)
# Global Mean
axs[1].axvline(116, c='b', alpha=0.3, linestyle='-')
axs[1].axvline(145, c='r', alpha=0.3, linestyle='-')
axs[1].axvline(320, c='b', alpha=0.3, linestyle='--')
axs[1].axvline(300, c='r', alpha=0.3, linestyle='--')
#plt.xlabel('Days', fontsize=12)
#plt.ylabel('Sea Ice Extent (1000 km2)', fontsize=14)
axs[1].set_xticks(ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
               labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axs[1].set_xlim(1, 365)
axs[1].set_ylim(0, 230)
# GES
# 1982-1997
axs[2].plot(np.arange(1, 366), np.nanmean(ges_seaiceextent[19:29], 0)/1000, c='#000080', linewidth=2)
# 1998-2022
axs[2].plot(np.arange(1, 366), np.nanmean(ges_seaiceextent[29:-1], 0)/1000, c='#BE0032', linewidth=2)
# Global Mean
axs[2].axvline(141, c='b', alpha=0.3, linestyle='-')
axs[2].axvline(146, c='r', alpha=0.3, linestyle='-')
axs[2].axvline(33, c='b', alpha=0.3, linestyle='--')
axs[2].axvline(14, c='r', alpha=0.3, linestyle='--')
#plt.xlabel('Days', fontsize=12)
#plt.ylabel('Sea Ice Extent (1000 km2)', fontsize=14)
axs[2].set_xticks(ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
               labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axs[2].set_xlim(1, 365)
axs[1].set_ylabel('Sea Ice Extent (thousand km${^2}$)', fontsize=16)
axs[2].set_xlabel('Day of the Year', fontsize=16)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\new_Fig4_A.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Calculate trends in sea ice phenology
# DRA
plt.plot(dates_1998, dra_seaiceextent_percentage[32,:])
plt.axhline(y=15, c='k')
print('SEA ICE ADVANCE')
print(dates_1998[np.argwhere(dra_seaiceextent_percentage[32,:] >= 15)[0][0]])
# sea ice advance
print(dates_1998[np.argwhere(dra_seaiceextent_percentage[32,:] >= 15)[0][0]].timetuple().tm_yday)
print('SEA ICE RETREAT')
print(dates_1998[np.argwhere(dra_seaiceextent_percentage[32,:] >= 15)[-1][0]])
# sea ice retreat
print(dates_1998[np.argwhere(dra_seaiceextent_percentage[32,:] >= 15)[-1][0]].timetuple().tm_yday)
plt.scatter(allyears, dra_seaiceadvance)
plt.scatter(allyears, dra_seaiceretreat)
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(dra_seaiceadvance)], dra_seaiceadvance[~np.isnan(dra_seaiceadvance)])
plt.plot(allyears, allyears*slope+intercept, c='k', linestyle='-')
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(dra_seaiceretreat)], dra_seaiceretreat[~np.isnan(dra_seaiceretreat)])
plt.plot(allyears, allyears*slope+intercept, c='k', linestyle='-')

slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], dra_seaiceadvance[28:])
plt.plot(allyears[28:], allyears[28:]*slope+intercept, c='k', linestyle='--')

slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], dra_seaiceretreat[28:])
plt.plot(allyears[28:], allyears[28:]*slope+intercept, c='k', linestyle='--')

#%%
# Rearrange dates
#dra_seaiceadvance = dra_seaiceadvance_original
#dra_seaiceretreat = dra_seaiceretreat_original
dra_seaiceadvance_original = dra_seaiceadvance
dra_seaiceadvance = dra_seaiceadvance+365
dra_seaiceretreat_original = dra_seaiceretreat
#dra_seaiceretreat = dra_seaiceretreat+244
#brs_seaiceadvance = brs_seaiceadvance_original
#brs_seaiceretreat = brs_seaiceretreat_original
brs_seaiceretreat[4] = brs_seaiceretreat[4] + 365
brs_seaiceretreat[30] = brs_seaiceretreat[30] + 365
brs_seaiceretreat[31] = brs_seaiceretreat[31] + 365
brs_seaiceadvance_original = brs_seaiceadvance
brs_seaiceadvance = brs_seaiceadvance+365
brs_seaiceretreat_original = brs_seaiceretreat
#brs_seaiceretreat = brs_seaiceretreat+244
#dra_seaiceretreat = dra_seaiceretreat+244
#brs_seaiceadvance = brs_seaiceadvance_original
#brs_seaiceretreat = brs_seaiceretreat_original
ges_seaiceretreat[0] = ges_seaiceretreat[0] + 365
ges_seaiceretreat[1] = ges_seaiceretreat[1] + 365
ges_seaiceretreat[2] = ges_seaiceretreat[2] + 365
ges_seaiceretreat[4] = ges_seaiceretreat[4] + 365
ges_seaiceretreat[5] = ges_seaiceretreat[5] + 365
ges_seaiceretreat[6] = ges_seaiceretreat[6] + 365
ges_seaiceretreat[10] = ges_seaiceretreat[10] + 365
ges_seaiceretreat[11] = ges_seaiceretreat[11] + 365
ges_seaiceretreat[12] = ges_seaiceretreat[12] + 365
ges_seaiceretreat[13] = ges_seaiceretreat[13] + 365
ges_seaiceretreat[14] = ges_seaiceretreat[14] + 365
ges_seaiceretreat[15] = ges_seaiceretreat[15] + 365
ges_seaiceretreat[18] = ges_seaiceretreat[18] + 365
ges_seaiceretreat[20] = ges_seaiceretreat[20] + 365
ges_seaiceretreat[23] = ges_seaiceretreat[23] + 365
ges_seaiceretreat[24] = ges_seaiceretreat[24] + 365
ges_seaiceretreat[27] = ges_seaiceretreat[27] + 365
ges_seaiceretreat[30] = ges_seaiceretreat[30] + 365
ges_seaiceretreat[32] = ges_seaiceretreat[32] + 365
ges_seaiceretreat[33] = ges_seaiceretreat[33] + 365
ges_seaiceretreat[34] = ges_seaiceretreat[34] + 365
ges_seaiceretreat[35] = ges_seaiceretreat[35] + 365
ges_seaiceretreat[36] = ges_seaiceretreat[36] + 365
ges_seaiceretreat[37] = ges_seaiceretreat[37] + 365
ges_seaiceretreat[38] = ges_seaiceretreat[38] + 365
ges_seaiceretreat[39] = ges_seaiceretreat[39] + 365
ges_seaiceadvance_original = ges_seaiceadvance
ges_seaiceadvance = ges_seaiceadvance+365
ges_seaiceretreat_original = ges_seaiceretreat
#brs_seaiceretreat = brs_seaiceretreat+244

#
fig, axs = plt.subplots(3, figsize=((6,12)))
## DRA
# 1982-2022
axs[0].scatter(allyears, dra_seaiceadvance, c='#000080', label='Sea Ice Advance')
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(dra_seaiceadvance)], dra_seaiceadvance[~np.isnan(dra_seaiceadvance)])
axs[0].plot(allyears, allyears*slope+intercept, c='#000080', alpha=0.5, label='1982-2022')
axs[0].scatter(allyears, dra_seaiceretreat, c='#BE0032', label='Sea Ice Retreat')
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(dra_seaiceretreat)], dra_seaiceretreat[~np.isnan(dra_seaiceretreat)])
axs[0].plot(allyears, allyears*slope+intercept, c='#BE0032', alpha=0.5, label='1982-2022')
# 1982-2022
#axs[0].scatter(allyears, dra_seaiceadvance, c='#000080', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], dra_seaiceadvance[28:])
axs[0].plot(allyears[28:], allyears[28:]*slope+intercept, c='#000080', alpha=0.5, linestyle='--', label='2010-2022')
#axs[0].scatter(allyears, dra_seaiceretreat, c='#BE0032', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], dra_seaiceretreat[28:])
axs[0].plot(allyears[28:], allyears[28:]*slope+intercept, c='#BE0032', alpha=0.5, linestyle='--', label='2010-2022')
axs[0].set_yticks(ticks = [1+244, 32+244, 60+244, 91+244, 121+244, 152+244, 182+244, 213+244, 244+244, 274+244, 305+244, 335+244],
               labels = [ 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A',])
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
axs[0].set_ylabel('Month', fontsize=14)
axs[0].set_xlabel('Years', fontsize=14)
## BRS
# 1982-2022
axs[1].scatter(allyears, brs_seaiceadvance, c='#000080', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(brs_seaiceadvance)], brs_seaiceadvance[~np.isnan(brs_seaiceadvance)])
axs[1].plot(allyears, allyears*slope+intercept, c='#000080', alpha=0.5)
axs[1].scatter(allyears, brs_seaiceretreat, c='#BE0032', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(brs_seaiceretreat)], brs_seaiceretreat[~np.isnan(brs_seaiceretreat)])
axs[1].plot(allyears, allyears*slope+intercept, c='#BE0032', alpha=0.5)
# 1982-2022
#axs[0].scatter(allyears, brs_seaiceadvance, c='#000080', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], brs_seaiceadvance[28:])
axs[1].plot(allyears[28:], allyears[28:]*slope+intercept, c='#000080', alpha=0.5, linestyle='--')
#axs[0].scatter(allyears, brs_seaiceretreat, c='#BE0032', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], brs_seaiceretreat[28:])
axs[1].plot(allyears[28:], allyears[28:]*slope+intercept, c='#BE0032', alpha=0.5, linestyle='--')
axs[1].set_yticks(ticks = [1+244, 32+244, 60+244, 91+244, 121+244, 152+244, 182+244, 213+244, 244+244, 274+244, 305+244, 335+244],
               labels = [ 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A',])
axs[1].set_ylabel('Month', fontsize=14)
axs[1].set_xlabel('Years', fontsize=14)
## GES
# 1982-2022
axs[2].scatter(allyears, ges_seaiceadvance, c='#000080', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(ges_seaiceadvance)], ges_seaiceadvance[~np.isnan(ges_seaiceadvance)])
axs[2].plot(allyears, allyears*slope+intercept, c='#000080', alpha=0.5)
axs[2].scatter(allyears, ges_seaiceretreat, c='#BE0032', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(ges_seaiceretreat)], ges_seaiceretreat[~np.isnan(ges_seaiceretreat)])
axs[2].plot(allyears, allyears*slope+intercept, c='#BE0032', alpha=0.5)
# 1982-2022
#axs[0].scatter(allyears, ges_seaiceadvance, c='#000080', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], ges_seaiceadvance[28:])
axs[2].plot(allyears[28:], allyears[28:]*slope+intercept, c='#000080', alpha=0.5, linestyle='--')
#axs[0].scatter(allyears, ges_seaiceretreat, c='#BE0032', label='1982-1997 Mean')
slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], ges_seaiceretreat[28:])
axs[2].plot(allyears[28:], allyears[28:]*slope+intercept, c='#BE0032', alpha=0.5, linestyle='--')
axs[2].set_yticks(ticks = [1+244, 32+244, 60+244, 91+244, 121+244, 152+244, 182+244, 213+244, 244+244, 274+244, 305+244, 335+244],
               labels = [ 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A',])
axs[2].set_ylabel('Month', fontsize=14)
axs[2].set_xlabel('Years', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarctic-peninsula-trends-2021\\analysis\\sup_Fig6.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()




#%% BRS
plt.plot(dates_1998, brs_seaiceextent_percentage[32,:])
plt.axhline(y=15, c='k')
print('SEA ICE ADVANCE')
print(dates_1998[np.argwhere(brs_seaiceextent_percentage[32,:] >= 15)[0][0]])
# sea ice advance
print(dates_1998[np.argwhere(brs_seaiceextent_percentage[32,:] >= 15)[0][0]].timetuple().tm_yday)
print('SEA ICE RETREAT')
print(dates_1998[np.argwhere(brs_seaiceextent_percentage[32,:] >= 15)[-1][0]])
# sea ice retreat
print(dates_1998[np.argwhere(brs_seaiceextent_percentage[32,:] >= 15)[-1][0]].timetuple().tm_yday)
# convert sea ice retreats to next year by adding 365
brs_seaiceretreat[4] = brs_seaiceretreat[4] + 365
brs_seaiceretreat[30] = brs_seaiceretreat[30] + 365
brs_seaiceretreat[31] = brs_seaiceretreat[31] + 365

plt.scatter(allyears, brs_seaiceadvance)
plt.scatter(allyears, brs_seaiceretreat)
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(brs_seaiceadvance)], brs_seaiceadvance[~np.isnan(brs_seaiceadvance)])
plt.plot(allyears, allyears*slope+intercept, c='k', linestyle='-')
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(brs_seaiceretreat)], brs_seaiceretreat[~np.isnan(brs_seaiceretreat)])
plt.plot(allyears, allyears*slope+intercept, c='k', linestyle='-')

slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], brs_seaiceadvance[28:])
plt.plot(allyears[28:], allyears[28:]*slope+intercept, c='k', linestyle='--')

slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], brs_seaiceretreat[28:])
plt.plot(allyears[28:], allyears[28:]*slope+intercept, c='k', linestyle='--')
#%% GES
plt.plot(dates_1998, ges_seaiceextent_percentage[32,:])
plt.axhline(y=15, c='k')
print('SEA ICE ADVANCE')
print(dates_1998[np.argwhere(ges_seaiceextent_percentage[32,:] >= 15)[0][0]])
# sea ice advance
print(dates_1998[np.argwhere(ges_seaiceextent_percentage[32,:] >= 15)[0][0]].timetuple().tm_yday)
print('SEA ICE RETREAT')
print(dates_1998[np.argwhere(ges_seaiceextent_percentage[32,:] >= 15)[-1][0]])
# sea ice retreat
print(dates_1998[np.argwhere(ges_seaiceextent_percentage[32,:] >= 15)[-1][0]].timetuple().tm_yday)
# convert sea ice retreats to next year by adding 365
ges_seaiceretreat[0] = ges_seaiceretreat[0] + 365
ges_seaiceretreat[1] = ges_seaiceretreat[1] + 365
ges_seaiceretreat[2] = ges_seaiceretreat[2] + 365
ges_seaiceretreat[4] = ges_seaiceretreat[4] + 365
ges_seaiceretreat[5] = ges_seaiceretreat[5] + 365
ges_seaiceretreat[6] = ges_seaiceretreat[6] + 365
ges_seaiceretreat[10] = ges_seaiceretreat[10] + 365
ges_seaiceretreat[11] = ges_seaiceretreat[11] + 365
ges_seaiceretreat[12] = ges_seaiceretreat[12] + 365
ges_seaiceretreat[13] = ges_seaiceretreat[13] + 365
ges_seaiceretreat[14] = ges_seaiceretreat[14] + 365
ges_seaiceretreat[15] = ges_seaiceretreat[15] + 365
ges_seaiceretreat[18] = ges_seaiceretreat[18] + 365
ges_seaiceretreat[20] = ges_seaiceretreat[20] + 365
ges_seaiceretreat[23] = ges_seaiceretreat[23] + 365
ges_seaiceretreat[24] = ges_seaiceretreat[24] + 365
ges_seaiceretreat[27] = ges_seaiceretreat[27] + 365
ges_seaiceretreat[30] = ges_seaiceretreat[30] + 365
ges_seaiceretreat[32] = ges_seaiceretreat[32] + 365
ges_seaiceretreat[33] = ges_seaiceretreat[33] + 365
ges_seaiceretreat[34] = ges_seaiceretreat[34] + 365
ges_seaiceretreat[35] = ges_seaiceretreat[35] + 365
ges_seaiceretreat[36] = ges_seaiceretreat[36] + 365
ges_seaiceretreat[37] = ges_seaiceretreat[37] + 365
ges_seaiceretreat[38] = ges_seaiceretreat[38] + 365
ges_seaiceretreat[39] = ges_seaiceretreat[39] + 365







plt.scatter(allyears, ges_seaiceadvance)
plt.scatter(allyears, ges_seaiceretreat)
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(ges_seaiceadvance)], ges_seaiceadvance[~np.isnan(ges_seaiceadvance)])
plt.plot(allyears, allyears*slope+intercept, c='k', linestyle='-')
slope, intercept, rval, pval, _ = stats.linregress(allyears[~np.isnan(ges_seaiceretreat)], ges_seaiceretreat[~np.isnan(ges_seaiceretreat)])
plt.plot(allyears, allyears*slope+intercept, c='k', linestyle='-')

slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], ges_seaiceadvance[28:])
plt.plot(allyears[28:], allyears[28:]*slope+intercept, c='k', linestyle='--')

slope, intercept, rval, pval, _ = stats.linregress(allyears[28:], ges_seaiceretreat[28:])
plt.plot(allyears[28:], allyears[28:]*slope+intercept, c='k', linestyle='--')






















#%% Supplementary Material with sea ice extent for each season of the year vs. each different decades
# Anomalies for each month/season between each decade
# average sea ice extent per season (1982-2022)
# Each month
dra_seaiceextent_jan = dra_seaiceextent[:,:31]
dra_seaiceextent_feb = dra_seaiceextent[:,31:59]
dra_seaiceextent_mar = dra_seaiceextent[:,59:90]
dra_seaiceextent_apr = dra_seaiceextent[:,90:120]
dra_seaiceextent_may = dra_seaiceextent[:,120:151]
dra_seaiceextent_jun = dra_seaiceextent[:,151:181]
dra_seaiceextent_jul = dra_seaiceextent[:,181:212]
dra_seaiceextent_aug = dra_seaiceextent[:,212:243]
dra_seaiceextent_sep = dra_seaiceextent[:,243:273]
dra_seaiceextent_oct = dra_seaiceextent[:,273:304]
dra_seaiceextent_nov = dra_seaiceextent[:,304:334]
dra_seaiceextent_dec = dra_seaiceextent[:,334:]
# Join per season
dra_seaiceextent_spring = np.hstack((dra_seaiceextent_sep, dra_seaiceextent_oct, dra_seaiceextent_nov))
dra_seaiceextent_summer = np.hstack((dra_seaiceextent_dec, dra_seaiceextent_jan, dra_seaiceextent_feb))
dra_seaiceextent_autumn = np.hstack((dra_seaiceextent_mar, dra_seaiceextent_apr, dra_seaiceextent_may))
dra_seaiceextent_winter = np.hstack((dra_seaiceextent_jun, dra_seaiceextent_jul, dra_seaiceextent_aug))
## Calculate boxplots SPRING
dra_seaiceextent_spring_19822022_boxplot = np.nanmean(dra_seaiceextent_spring, axis=1)
dra_seaiceextent_spring_19821991_boxplot = np.nanmean(dra_seaiceextent_spring[:10, :], axis=1)
dra_seaiceextent_spring_19922001_boxplot = np.nanmean(dra_seaiceextent_spring[10:20, :], axis=1)
dra_seaiceextent_spring_20022011_boxplot = np.nanmean(dra_seaiceextent_spring[20:30, :], axis=1)
dra_seaiceextent_spring_20122022_boxplot = np.nanmean(dra_seaiceextent_spring[30:, :], axis=1)
## Calculate boxplots summer
dra_seaiceextent_summer_19822022_boxplot = np.nanmean(dra_seaiceextent_summer, axis=1)
dra_seaiceextent_summer_19821991_boxplot = np.nanmean(dra_seaiceextent_summer[:10, :], axis=1)
dra_seaiceextent_summer_19922001_boxplot = np.nanmean(dra_seaiceextent_summer[10:20, :], axis=1)
dra_seaiceextent_summer_20022011_boxplot = np.nanmean(dra_seaiceextent_summer[20:30, :], axis=1)
dra_seaiceextent_summer_20122022_boxplot = np.nanmean(dra_seaiceextent_summer[30:, :], axis=1)
## Calculate boxplots autumn
dra_seaiceextent_autumn_19822022_boxplot = np.nanmean(dra_seaiceextent_autumn, axis=1)
dra_seaiceextent_autumn_19821991_boxplot = np.nanmean(dra_seaiceextent_autumn[:10, :], axis=1)
dra_seaiceextent_autumn_19922001_boxplot = np.nanmean(dra_seaiceextent_autumn[10:20, :], axis=1)
dra_seaiceextent_autumn_20022011_boxplot = np.nanmean(dra_seaiceextent_autumn[20:30, :], axis=1)
dra_seaiceextent_autumn_20122022_boxplot = np.nanmean(dra_seaiceextent_autumn[30:, :], axis=1)
## Calculate boxplots winter
dra_seaiceextent_winter_19822022_boxplot = np.nanmean(dra_seaiceextent_winter, axis=1)
dra_seaiceextent_winter_19821991_boxplot = np.nanmean(dra_seaiceextent_winter[:10, :], axis=1)
dra_seaiceextent_winter_19922001_boxplot = np.nanmean(dra_seaiceextent_winter[10:20, :], axis=1)
dra_seaiceextent_winter_20022011_boxplot = np.nanmean(dra_seaiceextent_winter[20:30, :], axis=1)
dra_seaiceextent_winter_20122022_boxplot = np.nanmean(dra_seaiceextent_winter[30:, :], axis=1)
#%%
fig, axs = plt.subplots(4, figsize=((8,6)))
axs[0].boxplot([dra_seaiceextent_spring_19822022_boxplot, dra_seaiceextent_spring_19821991_boxplot,
             dra_seaiceextent_spring_19922001_boxplot, dra_seaiceextent_spring_20022011_boxplot,
             dra_seaiceextent_spring_20122022_boxplot], showmeans=True, meanline=True)
axs[1].boxplot([dra_seaiceextent_summer_19822022_boxplot, dra_seaiceextent_summer_19821991_boxplot,
             dra_seaiceextent_summer_19922001_boxplot, dra_seaiceextent_summer_20022011_boxplot,
             dra_seaiceextent_summer_20122022_boxplot], showmeans=True, meanline=True)
axs[2].boxplot([dra_seaiceextent_autumn_19822022_boxplot, dra_seaiceextent_autumn_19821991_boxplot,
             dra_seaiceextent_autumn_19922001_boxplot, dra_seaiceextent_autumn_20022011_boxplot,
             dra_seaiceextent_autumn_20122022_boxplot], showmeans=True, meanline=True)
axs[3].boxplot([dra_seaiceextent_winter_19822022_boxplot, dra_seaiceextent_winter_19821991_boxplot,
             dra_seaiceextent_winter_19922001_boxplot, dra_seaiceextent_winter_20022011_boxplot,
             dra_seaiceextent_winter_20122022_boxplot], showmeans=True, meanline=True)



#%% Jan
dra_seaiceextent_jan_19822022_boxplot = np.nanmean(dra_seaiceextent_jan)/1000
dra_seaiceextent_jan_19821991_boxplot = np.nanmean(dra_seaiceextent_jan[:10, :])/1000
dra_seaiceextent_jan_19922001_boxplot = np.nanmean(dra_seaiceextent_jan[10:20, :])/1000
dra_seaiceextent_jan_20022011_boxplot = np.nanmean(dra_seaiceextent_jan[20:30, :])/1000
dra_seaiceextent_jan_20122022_boxplot = np.nanmean(dra_seaiceextent_jan[30:, :])/1000
dra_seaiceextent_jan_19822022_boxplot_std = np.nanstd(dra_seaiceextent_jan)/1000
dra_seaiceextent_jan_19821991_boxplot_std = np.nanstd(dra_seaiceextent_jan[:10, :])/1000
dra_seaiceextent_jan_19922001_boxplot_std = np.nanstd(dra_seaiceextent_jan[10:20, :])/1000
dra_seaiceextent_jan_20022011_boxplot_std = np.nanstd(dra_seaiceextent_jan[20:30, :])/1000
dra_seaiceextent_jan_20122022_boxplot_std = np.nanstd(dra_seaiceextent_jan[30:, :])/1000
#feb
dra_seaiceextent_feb_19822022_boxplot = np.nanmean(dra_seaiceextent_feb)/1000
dra_seaiceextent_feb_19821991_boxplot = np.nanmean(dra_seaiceextent_feb[:10, :])/1000
dra_seaiceextent_feb_19922001_boxplot = np.nanmean(dra_seaiceextent_feb[10:20, :])/1000
dra_seaiceextent_feb_20022011_boxplot = np.nanmean(dra_seaiceextent_feb[20:30, :])/1000
dra_seaiceextent_feb_20122022_boxplot = np.nanmean(dra_seaiceextent_feb[30:, :])/1000
dra_seaiceextent_feb_19822022_boxplot_std = np.nanstd(dra_seaiceextent_feb)/1000
dra_seaiceextent_feb_19821991_boxplot_std = np.nanstd(dra_seaiceextent_feb[:10, :])/1000
dra_seaiceextent_feb_19922001_boxplot_std = np.nanstd(dra_seaiceextent_feb[10:20, :])/1000
dra_seaiceextent_feb_20022011_boxplot_std = np.nanstd(dra_seaiceextent_feb[20:30, :])/1000
dra_seaiceextent_feb_20122022_boxplot_std = np.nanstd(dra_seaiceextent_feb[30:, :])/1000
#mar
dra_seaiceextent_mar_19822022_boxplot = np.nanmean(dra_seaiceextent_mar)/1000
dra_seaiceextent_mar_19821991_boxplot = np.nanmean(dra_seaiceextent_mar[:10, :])/1000
dra_seaiceextent_mar_19922001_boxplot = np.nanmean(dra_seaiceextent_mar[10:20, :])/1000
dra_seaiceextent_mar_20022011_boxplot = np.nanmean(dra_seaiceextent_mar[20:30, :])/1000
dra_seaiceextent_mar_20122022_boxplot = np.nanmean(dra_seaiceextent_mar[30:, :])/1000
dra_seaiceextent_mar_19822022_boxplot_std = np.nanstd(dra_seaiceextent_mar)/1000
dra_seaiceextent_mar_19821991_boxplot_std = np.nanstd(dra_seaiceextent_mar[:10, :])/1000
dra_seaiceextent_mar_19922001_boxplot_std = np.nanstd(dra_seaiceextent_mar[10:20, :])/1000
dra_seaiceextent_mar_20022011_boxplot_std = np.nanstd(dra_seaiceextent_mar[20:30, :])/1000
dra_seaiceextent_mar_20122022_boxplot_std = np.nanstd(dra_seaiceextent_mar[30:, :])/1000
#apr
dra_seaiceextent_apr_19822022_boxplot = np.nanmean(dra_seaiceextent_apr)/1000
dra_seaiceextent_apr_19821991_boxplot = np.nanmean(dra_seaiceextent_apr[:10, :])/1000
dra_seaiceextent_apr_19922001_boxplot = np.nanmean(dra_seaiceextent_apr[10:20, :])/1000
dra_seaiceextent_apr_20022011_boxplot = np.nanmean(dra_seaiceextent_apr[20:30, :])/1000
dra_seaiceextent_apr_20122022_boxplot = np.nanmean(dra_seaiceextent_apr[30:, :])/1000
dra_seaiceextent_apr_19822022_boxplot_std = np.nanstd(dra_seaiceextent_apr)/1000
dra_seaiceextent_apr_19821991_boxplot_std = np.nanstd(dra_seaiceextent_apr[:10, :])/1000
dra_seaiceextent_apr_19922001_boxplot_std = np.nanstd(dra_seaiceextent_apr[10:20, :])/1000
dra_seaiceextent_apr_20022011_boxplot_std = np.nanstd(dra_seaiceextent_apr[20:30, :])/1000
dra_seaiceextent_apr_20122022_boxplot_std = np.nanstd(dra_seaiceextent_apr[30:, :])/1000
#may
dra_seaiceextent_may_19822022_boxplot = np.nanmean(dra_seaiceextent_may)/1000
dra_seaiceextent_may_19821991_boxplot = np.nanmean(dra_seaiceextent_may[:10, :])/1000
dra_seaiceextent_may_19922001_boxplot = np.nanmean(dra_seaiceextent_may[10:20, :])/1000
dra_seaiceextent_may_20022011_boxplot = np.nanmean(dra_seaiceextent_may[20:30, :])/1000
dra_seaiceextent_may_20122022_boxplot = np.nanmean(dra_seaiceextent_may[30:, :])/1000
dra_seaiceextent_may_19822022_boxplot_std = np.nanstd(dra_seaiceextent_may)/1000
dra_seaiceextent_may_19821991_boxplot_std = np.nanstd(dra_seaiceextent_may[:10, :])/1000
dra_seaiceextent_may_19922001_boxplot_std = np.nanstd(dra_seaiceextent_may[10:20, :])/1000
dra_seaiceextent_may_20022011_boxplot_std = np.nanstd(dra_seaiceextent_may[20:30, :])/1000
dra_seaiceextent_may_20122022_boxplot_std = np.nanstd(dra_seaiceextent_may[30:, :])/1000
#jun
dra_seaiceextent_jun_19822022_boxplot = np.nanmean(dra_seaiceextent_jun)/1000
dra_seaiceextent_jun_19821991_boxplot = np.nanmean(dra_seaiceextent_jun[:10, :])/1000
dra_seaiceextent_jun_19922001_boxplot = np.nanmean(dra_seaiceextent_jun[10:20, :])/1000
dra_seaiceextent_jun_20022011_boxplot = np.nanmean(dra_seaiceextent_jun[20:30, :])/1000
dra_seaiceextent_jun_20122022_boxplot = np.nanmean(dra_seaiceextent_jun[30:, :])/1000
dra_seaiceextent_jun_19822022_boxplot_std = np.nanstd(dra_seaiceextent_jun)/1000
dra_seaiceextent_jun_19821991_boxplot_std = np.nanstd(dra_seaiceextent_jun[:10, :])/1000
dra_seaiceextent_jun_19922001_boxplot_std = np.nanstd(dra_seaiceextent_jun[10:20, :])/1000
dra_seaiceextent_jun_20022011_boxplot_std = np.nanstd(dra_seaiceextent_jun[20:30, :])/1000
dra_seaiceextent_jun_20122022_boxplot_std = np.nanstd(dra_seaiceextent_jun[30:, :])/1000
#jul
dra_seaiceextent_jul_19822022_boxplot = np.nanmean(dra_seaiceextent_jul)/1000
dra_seaiceextent_jul_19821991_boxplot = np.nanmean(dra_seaiceextent_jul[:10, :])/1000
dra_seaiceextent_jul_19922001_boxplot = np.nanmean(dra_seaiceextent_jul[10:20, :])/1000
dra_seaiceextent_jul_20022011_boxplot = np.nanmean(dra_seaiceextent_jul[20:30, :])/1000
dra_seaiceextent_jul_20122022_boxplot = np.nanmean(dra_seaiceextent_jul[30:, :])/1000
dra_seaiceextent_jul_19822022_boxplot_std = np.nanstd(dra_seaiceextent_jul)/1000
dra_seaiceextent_jul_19821991_boxplot_std = np.nanstd(dra_seaiceextent_jul[:10, :])/1000
dra_seaiceextent_jul_19922001_boxplot_std = np.nanstd(dra_seaiceextent_jul[10:20, :])/1000
dra_seaiceextent_jul_20022011_boxplot_std = np.nanstd(dra_seaiceextent_jul[20:30, :])/1000
dra_seaiceextent_jul_20122022_boxplot_std = np.nanstd(dra_seaiceextent_jul[30:, :])/1000
#aug
dra_seaiceextent_aug_19822022_boxplot = np.nanmean(dra_seaiceextent_aug)/1000
dra_seaiceextent_aug_19821991_boxplot = np.nanmean(dra_seaiceextent_aug[:10, :])/1000
dra_seaiceextent_aug_19922001_boxplot = np.nanmean(dra_seaiceextent_aug[10:20, :])/1000
dra_seaiceextent_aug_20022011_boxplot = np.nanmean(dra_seaiceextent_aug[20:30, :])/1000
dra_seaiceextent_aug_20122022_boxplot = np.nanmean(dra_seaiceextent_aug[30:, :])/1000
dra_seaiceextent_aug_19822022_boxplot_std = np.nanstd(dra_seaiceextent_aug)/1000
dra_seaiceextent_aug_19821991_boxplot_std = np.nanstd(dra_seaiceextent_aug[:10, :])/1000
dra_seaiceextent_aug_19922001_boxplot_std = np.nanstd(dra_seaiceextent_aug[10:20, :])/1000
dra_seaiceextent_aug_20022011_boxplot_std = np.nanstd(dra_seaiceextent_aug[20:30, :])/1000
dra_seaiceextent_aug_20122022_boxplot_std = np.nanstd(dra_seaiceextent_aug[30:, :])/1000
#sep
dra_seaiceextent_sep_19822022_boxplot = np.nanmean(dra_seaiceextent_sep)/1000
dra_seaiceextent_sep_19821991_boxplot = np.nanmean(dra_seaiceextent_sep[:10, :])/1000
dra_seaiceextent_sep_19922001_boxplot = np.nanmean(dra_seaiceextent_sep[10:20, :])/1000
dra_seaiceextent_sep_20022011_boxplot = np.nanmean(dra_seaiceextent_sep[20:30, :])/1000
dra_seaiceextent_sep_20122022_boxplot = np.nanmean(dra_seaiceextent_sep[30:, :])/1000
dra_seaiceextent_sep_19822022_boxplot_std = np.nanstd(dra_seaiceextent_sep)/1000
dra_seaiceextent_sep_19821991_boxplot_std = np.nanstd(dra_seaiceextent_sep[:10, :])/1000
dra_seaiceextent_sep_19922001_boxplot_std = np.nanstd(dra_seaiceextent_sep[10:20, :])/1000
dra_seaiceextent_sep_20022011_boxplot_std = np.nanstd(dra_seaiceextent_sep[20:30, :])/1000
dra_seaiceextent_sep_20122022_boxplot_std = np.nanstd(dra_seaiceextent_sep[30:, :])/1000
#oct
dra_seaiceextent_oct_19822022_boxplot = np.nanmean(dra_seaiceextent_oct)/1000
dra_seaiceextent_oct_19821991_boxplot = np.nanmean(dra_seaiceextent_oct[:10, :])/1000
dra_seaiceextent_oct_19922001_boxplot = np.nanmean(dra_seaiceextent_oct[10:20, :])/1000
dra_seaiceextent_oct_20022011_boxplot = np.nanmean(dra_seaiceextent_oct[20:30, :])/1000
dra_seaiceextent_oct_20122022_boxplot = np.nanmean(dra_seaiceextent_oct[30:, :])/1000
dra_seaiceextent_oct_19822022_boxplot_std = np.nanstd(dra_seaiceextent_oct)/1000
dra_seaiceextent_oct_19821991_boxplot_std = np.nanstd(dra_seaiceextent_oct[:10, :])/1000
dra_seaiceextent_oct_19922001_boxplot_std = np.nanstd(dra_seaiceextent_oct[10:20, :])/1000
dra_seaiceextent_oct_20022011_boxplot_std = np.nanstd(dra_seaiceextent_oct[20:30, :])/1000
dra_seaiceextent_oct_20122022_boxplot_std = np.nanstd(dra_seaiceextent_oct[30:, :])/1000
#nov
dra_seaiceextent_nov_19822022_boxplot = np.nanmean(dra_seaiceextent_nov)/1000
dra_seaiceextent_nov_19821991_boxplot = np.nanmean(dra_seaiceextent_nov[:10, :])/1000
dra_seaiceextent_nov_19922001_boxplot = np.nanmean(dra_seaiceextent_nov[10:20, :])/1000
dra_seaiceextent_nov_20022011_boxplot = np.nanmean(dra_seaiceextent_nov[20:30, :])/1000
dra_seaiceextent_nov_20122022_boxplot = np.nanmean(dra_seaiceextent_nov[30:, :])/1000
dra_seaiceextent_nov_19822022_boxplot_std = np.nanstd(dra_seaiceextent_nov)/1000
dra_seaiceextent_nov_19821991_boxplot_std = np.nanstd(dra_seaiceextent_nov[:10, :])/1000
dra_seaiceextent_nov_19922001_boxplot_std = np.nanstd(dra_seaiceextent_nov[10:20, :])/1000
dra_seaiceextent_nov_20022011_boxplot_std = np.nanstd(dra_seaiceextent_nov[20:30, :])/1000
dra_seaiceextent_nov_20122022_boxplot_std = np.nanstd(dra_seaiceextent_nov[30:, :])/1000
#dec
dra_seaiceextent_dec_19822022_boxplot = np.nanmean(dra_seaiceextent_dec)/1000
dra_seaiceextent_dec_19821991_boxplot = np.nanmean(dra_seaiceextent_dec[:10, :])/1000
dra_seaiceextent_dec_19922001_boxplot = np.nanmean(dra_seaiceextent_dec[10:20, :])/1000
dra_seaiceextent_dec_20022011_boxplot = np.nanmean(dra_seaiceextent_dec[20:30, :])/1000
dra_seaiceextent_dec_20122022_boxplot = np.nanmean(dra_seaiceextent_dec[30:, :])/1000
dra_seaiceextent_dec_19822022_boxplot_std = np.nanstd(dra_seaiceextent_dec)/1000
dra_seaiceextent_dec_19821991_boxplot_std = np.nanstd(dra_seaiceextent_dec[:10, :])/1000
dra_seaiceextent_dec_19922001_boxplot_std = np.nanstd(dra_seaiceextent_dec[10:20, :])/1000
dra_seaiceextent_dec_20022011_boxplot_std = np.nanstd(dra_seaiceextent_dec[20:30, :])/1000
dra_seaiceextent_dec_20122022_boxplot_std = np.nanstd(dra_seaiceextent_dec[30:, :])/1000
#%%
































#%%
## Calculate boxplots summer
dra_seaiceextent_summer_19822022_boxplot = np.nanmean(dra_seaiceextent_summer)
dra_seaiceextent_summer_19821991_boxplot = np.nanmean(dra_seaiceextent_summer[:10, :])
dra_seaiceextent_summer_19922001_boxplot = np.nanmean(dra_seaiceextent_summer[10:20, :])
dra_seaiceextent_summer_20022011_boxplot = np.nanmean(dra_seaiceextent_summer[20:30, :])
dra_seaiceextent_summer_20122022_boxplot = np.nanmean(dra_seaiceextent_summer[30:, :])
## Calculate boxplots autumn
dra_seaiceextent_autumn_19822022_boxplot = np.nanmean(dra_seaiceextent_autumn)
dra_seaiceextent_autumn_19821991_boxplot = np.nanmean(dra_seaiceextent_autumn[:10, :])
dra_seaiceextent_autumn_19922001_boxplot = np.nanmean(dra_seaiceextent_autumn[10:20, :])
dra_seaiceextent_autumn_20022011_boxplot = np.nanmean(dra_seaiceextent_autumn[20:30, :])
dra_seaiceextent_autumn_20122022_boxplot = np.nanmean(dra_seaiceextent_autumn[30:, :])
## Calculate boxplots winter
dra_seaiceextent_winter_19822022_boxplot = np.nanmean(dra_seaiceextent_winter)
dra_seaiceextent_winter_19821991_boxplot = np.nanmean(dra_seaiceextent_winter[:10, :])
dra_seaiceextent_winter_19922001_boxplot = np.nanmean(dra_seaiceextent_winter[10:20, :])
dra_seaiceextent_winter_20022011_boxplot = np.nanmean(dra_seaiceextent_winter[20:30, :])
dra_seaiceextent_winter_20122022_boxplot = np.nanmean(dra_seaiceextent_winter[30:, :])

#%%






















#%%

dra_seaiceextent_spring_grandmean = np.nanmean(dra_seaiceextent_spring, axis=1)
dra_seaiceextent_spring_19821991_allanoms = np.nanmean(dra_seaiceextent_spring[:10, :], axis=1) - dra_seaiceextent_spring_grandmean
dra_seaiceextent_spring_19821991_anommean = np.nanmean(dra_seaiceextent_spring_19821991_allanoms)
dra_seaiceextent_spring_19821991_anomstd = np.nanstd(dra_seaiceextent_spring_19821991_allanoms)
dra_seaiceextent_spring_19922001_allanoms = np.nanmean(dra_seaiceextent_spring[10:20, :], axis=1) - dra_seaiceextent_spring_grandmean
dra_seaiceextent_spring_19922001_anommean = np.nanmean(dra_seaiceextent_spring_19922001_allanoms)
dra_seaiceextent_spring_19922001_anomstd = np.nanstd(dra_seaiceextent_spring_19922001_allanoms)
dra_seaiceextent_spring_20022011_allanoms = np.nanmean(dra_seaiceextent_spring[20:30, :], axis=1) - dra_seaiceextent_spring_grandmean
dra_seaiceextent_spring_20022011_anommean = np.nanmean(dra_seaiceextent_spring_20022011_allanoms)
dra_seaiceextent_spring_20022011_anomstd = np.nanstd(dra_seaiceextent_spring_20022011_allanoms)
dra_seaiceextent_spring_20122022_allanoms = np.nanmean(dra_seaiceextent_spring[30:, :], axis=1) - dra_seaiceextent_spring_grandmean
dra_seaiceextent_spring_20122022_anommean = np.nanmean(dra_seaiceextent_spring_20122022_allanoms)
dra_seaiceextent_spring_20122022_anomstd = np.nanstd(dra_seaiceextent_spring_20122022_allanoms)

#%% Test Figure




plt.bar(x=[1, 2, 3, 4], height = [dra_seaiceextent_spring_19821991_anommean, dra_seaiceextent_spring_19922001_anommean,
        dra_seaiceextent_spring_20022011_anommean, dra_seaiceextent_spring_20122022_anommean])


































#%%


blue = np.nanmean(ges_seaiceextent_percentage[:16], 0)
red = np.nanmean(ges_seaiceextent_percentage[16:], 0)


#%%


plt.plot(np.nanmean(dra_seaiceextent, 0) - np.nanstd(dra_seaiceextent, 0), c='k', linewidth=2, alpha=0.5)




plt.plot(np.nanmean(dra_seaiceextent[:16], 0), c='b', linewidth=1)
plt.plot(np.nanmean(dra_seaiceextent[16:], 0), c='r', linewidth=1)
# DRA 2
# Spring


