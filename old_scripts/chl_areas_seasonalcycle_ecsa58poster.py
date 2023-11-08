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
import cmocean
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
def hampel_filter_forloop(input_series, window_size, n_sigma=3):
    
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    
    indices = []
    
    # possibly use np.nanmedian
    
    for i in range((window_size), (n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigma * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices
    
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\oc4so-chl\\')
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\rrs\\')
### Load data 1998-2020
fh = np.load('chloc4so_8day.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
chl = fh['chl']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Load data 2012-13
fh = np.load('chloc4so_8day_20122013.npz', allow_pickle=True)
chl_20122013 = fh['chl']
# Replace 2012-2013 in original chl
chl[:,:, 644:736] = chl_20122013
chl[chl > 30] = np.nan
#chl[chl == 100] = np.nan
### Create areas
# Central Bransfield Strait
CBS_verts = [(-61, -63.1),
             (-57.5, -62.2),
             (-57.5, -63.1),
             (-61, -64)]
# Gerlache Strait
GS_verts = [(-63.3, -64.8),
             (-61.5, -64),
             (-61, -64.2),
             (-63.1, -64.85)]
# Elephant Island
EI_verts = [(-56, -60.8),
             (-53, -60.8),
             (-53, -61.5),
             (-56, -61.5)]
# Drake Strait
DS_verts = [(-65, -62.3),
             (-58, -61),
             (-58, -61.7),
             (-65, -63)]
# Weddell Sea
WS_verts = [(-55, -63.5),
             (-52, -63.5),
             (-52, -65),
             (-55, -65)]
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -52, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
poly_CBS = Polygon(list(CBS_verts), facecolor=[1,1,1,0], edgecolor='k', linewidth=1, linestyle='--', zorder=2, transform=ccrs.PlateCarree())
plt.gca().add_patch(poly_CBS)
poly_GS = Polygon(list(GS_verts), facecolor=[1,1,1,0], edgecolor='k', linewidth=1, linestyle='--', zorder=2, transform=ccrs.PlateCarree())
plt.gca().add_patch(poly_GS)
poly_EI = Polygon(list(EI_verts), facecolor=[1,1,1,0], edgecolor='k', linewidth=1, linestyle='--', zorder=2, transform=ccrs.PlateCarree())
plt.gca().add_patch(poly_EI)
poly_DS = Polygon(list(DS_verts), facecolor=[1,1,1,0], edgecolor='k', linewidth=1, linestyle='--', zorder=2, transform=ccrs.PlateCarree())
plt.gca().add_patch(poly_DS)
poly_WS = Polygon(list(WS_verts), facecolor=[1,1,1,0], edgecolor='k', linewidth=1, linestyle='--', zorder=2, transform=ccrs.PlateCarree())
plt.gca().add_patch(poly_WS)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\roi\\roi_locations.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
### Calculate seasonal cycle for each area
### Central Bransfield Strait
x, y = np.meshgrid(lon, lat) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(CBS_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon), len(lat))
mask3d = np.repeat([mask], np.size(chl,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
chl_CBS = np.ma.array(chl, mask=~mask3d)
chl_CBS = np.nanmean(chl_CBS, (0,1))
# Calculate average cycle
week8day_num = np.tile(np.arange(1,47),(23))[:-1]
chl_CBS_df = pd.DataFrame(chl_CBS, index=time_date, columns = ['chl'])
chl_CBS_df['8DWeek Num'] = week8day_num
chl_averagecycle_CBS = np.squeeze(chl_CBS_df.groupby(['8DWeek Num']).mean().values)
### Calculate cycle for each year CBS
for i in np.arange(1998, 2021):
    print(i)
    chl_CBS_temp_df = chl_CBS_df[time_date_years == i]
    chl_CBS_temp = chl_CBS_temp_df['chl'].values
    if i == 1998:
        chl_CBS_yearlycycles = chl_CBS_temp
    elif i == 2020:
        chl_CBS_temp = np.pad(chl_CBS_temp, pad_width=(0,1), mode='constant', constant_values=np.nan)
        chl_CBS_yearlycycles = np.vstack((chl_CBS_yearlycycles, chl_CBS_temp))
    else:
        chl_CBS_yearlycycles = np.vstack((chl_CBS_yearlycycles, chl_CBS_temp))
### Plot average cycle and each year cycle for CBS
chl_CBS_cycles_19982005 = np.nanmean(chl_CBS_yearlycycles[:8,:], axis=0)
chl_CBS_cycles_20062013 = np.nanmean(chl_CBS_yearlycycles[8:16,:], axis=0)
chl_CBS_cycles_20142020 = np.nanmean(chl_CBS_yearlycycles[16:,:], axis=0)
chl_CBS_cycles_p90 = np.nanpercentile(chl_CBS_yearlycycles, 90, axis=0)
chl_CBS_cycles_p10 = np.nanpercentile(chl_CBS_yearlycycles, 10, axis=0)
plt.figure(figsize=(12,9))
plt.plot(np.arange(1,29),
         np.hstack((chl_averagecycle_CBS[31:], chl_averagecycle_CBS[:13])), color = 'k', linewidth = 4)
#plt.plot(np.arange(1,29),
#         np.hstack((chl_CBS_cycles_19982005[31:], chl_CBS_cycles_19982005[:13])), color = 'k', linewidth = 2, linestyle='--', label='1998-2005')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_CBS_cycles_20062013[31:], chl_CBS_cycles_20062013[:13])), color = 'k', linewidth = 2, linestyle=':', label='2006-2013')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_CBS_cycles_20142020[31:], chl_CBS_cycles_20142020[:13])), color = 'k', linewidth = 2, linestyle='-.', label='2014-2020')
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p10, color = 'k', linewidth = 1)
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p90, color = 'k', linewidth = 1)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_CBS[31:], chl_averagecycle_CBS[:13])),
                 np.hstack((chl_CBS_cycles_p90[31:], chl_CBS_cycles_p90[:13])),
                 color =[153/256, 0, 204/256, 1], alpha=.5, edgecolor = None)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_CBS[31:], chl_averagecycle_CBS[:13])),
                 np.hstack((chl_CBS_cycles_p10[31:], chl_CBS_cycles_p10[:13])), color =[153/256, 0, 204/256, 1], alpha=.5, edgecolor = None)
plt.xlim(1,28)
#plt.ylim(0.1,1.55)
plt.xticks([1, 5, 8, 12, 16, 20, 24, 28],
           ['Sep','Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'], fontsize=12)
plt.ylabel('Chl-a (mg/m3)', fontsize=14)
plt.title('Central Bransfield Strait', fontsize=16)
#plt.legend(fontsize=16)
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\roi\\bransfield_seasonalcycle_ecsa58.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
### Gerlache Strait
x, y = np.meshgrid(lon, lat) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(GS_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon), len(lat))
mask3d = np.repeat([mask], np.size(chl,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
chl_GS = np.ma.array(chl, mask=~mask3d)
chl_GS = np.nanmean(chl_GS, (0,1))
# Calculate average cycle
week8day_num = np.tile(np.arange(1,47),(23))[:-1]
chl_GS_df = pd.DataFrame(chl_GS, index=time_date, columns = ['chl'])
chl_GS_df['8DWeek Num'] = week8day_num
chl_averagecycle_GS = np.squeeze(chl_GS_df.groupby(['8DWeek Num']).mean().values)
### Calculate cycle for each year GS
for i in np.arange(1998, 2021):
    print(i)
    chl_GS_temp_df = chl_GS_df[time_date_years == i]
    chl_GS_temp = chl_GS_temp_df['chl'].values
    if i == 1998:
        chl_GS_yearlycycles = chl_GS_temp
    elif i == 2020:
        chl_GS_temp = np.pad(chl_GS_temp, pad_width=(0,1), mode='constant', constant_values=np.nan)
        chl_GS_yearlycycles = np.vstack((chl_GS_yearlycycles, chl_GS_temp))
    else:
        chl_GS_yearlycycles = np.vstack((chl_GS_yearlycycles, chl_GS_temp))
### Plot average cycle and each year cycle for GS
chl_GS_cycles_19982005 = np.nanmean(chl_GS_yearlycycles[:8,:], axis=0)
chl_GS_cycles_20062013 = np.nanmean(chl_GS_yearlycycles[8:16,:], axis=0)
chl_GS_cycles_20142020 = np.nanmean(chl_GS_yearlycycles[16:,:], axis=0)
chl_GS_cycles_p90 = np.nanpercentile(chl_GS_yearlycycles, 90, axis=0)
chl_GS_cycles_p10 = np.nanpercentile(chl_GS_yearlycycles, 10, axis=0)
plt.figure(figsize=(12,9))
plt.plot(np.arange(1,29),
         np.hstack((chl_averagecycle_GS[31:], chl_averagecycle_GS[:13])), color = 'k', linewidth = 4)
#plt.plot(np.arange(1,29),
#         np.hstack((chl_GS_cycles_19982005[31:], chl_GS_cycles_19982005[:13])), color = 'k', linewidth = 2, linestyle='--', label='1998-2005')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_GS_cycles_20062013[31:], chl_GS_cycles_20062013[:13])), color = 'k', linewidth = 2, linestyle=':', label='2006-2013')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_GS_cycles_20142020[31:], chl_GS_cycles_20142020[:13])), color = 'k', linewidth = 2, linestyle='-.', label='2014-2020')
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p10, color = 'k', linewidth = 1)
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p90, color = 'k', linewidth = 1)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_GS[31:], chl_averagecycle_GS[:13])),
                 np.hstack((chl_GS_cycles_p90[31:], chl_GS_cycles_p90[:13])),
                 color = [171/256, 221/256, 164/256, 1], alpha=.5, edgecolor = None)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_GS[31:], chl_averagecycle_GS[:13])),
                 np.hstack((chl_GS_cycles_p10[31:], chl_GS_cycles_p10[:13])),
                 color =[171/256, 221/256, 164/256, 1], alpha=.5, edgecolor = None)
plt.xlim(1,28)
#plt.ylim(0.1,1.55)
plt.xticks([1, 5, 8, 12, 16, 20, 24, 28],
           ['Sep','Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'], fontsize=12)
plt.ylabel('Chl-a (mg/m3)', fontsize=14)
plt.title('Gerlache Strait', fontsize=16)
#plt.legend(fontsize=16)
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\roi\\gerlache_seasonalcycle_ecsa58.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
### Elephant Island
x, y = np.meshgrid(lon, lat) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(EI_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon), len(lat))
mask3d = np.repeat([mask], np.size(chl,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
chl_EI = np.ma.array(chl, mask=~mask3d)
chl_EI = np.nanmean(chl_EI, (0,1))
# Calculate average cycle
week8day_num = np.tile(np.arange(1,47),(23))[:-1]
chl_EI_df = pd.DataFrame(chl_EI, index=time_date, columns = ['chl'])
chl_EI_df['8DWeek Num'] = week8day_num
chl_averagecycle_EI = np.squeeze(chl_EI_df.groupby(['8DWeek Num']).mean().values)
### Calculate cycle for each year EI
for i in np.arange(1998, 2021):
    print(i)
    chl_EI_temp_df = chl_EI_df[time_date_years == i]
    chl_EI_temp = chl_EI_temp_df['chl'].values
    if i == 1998:
        chl_EI_yearlycycles = chl_EI_temp
    elif i == 2020:
        chl_EI_temp = np.pad(chl_EI_temp, pad_width=(0,1), mode='constant', constant_values=np.nan)
        chl_EI_yearlycycles = np.vstack((chl_EI_yearlycycles, chl_EI_temp))
    else:
        chl_EI_yearlycycles = np.vstack((chl_EI_yearlycycles, chl_EI_temp))
### Plot average cycle and each year cycle for EI
chl_EI_cycles_19982005 = np.nanmean(chl_EI_yearlycycles[:8,:], axis=0)
chl_EI_cycles_20062013 = np.nanmean(chl_EI_yearlycycles[8:16,:], axis=0)
chl_EI_cycles_20142020 = np.nanmean(chl_EI_yearlycycles[16:,:], axis=0)
chl_EI_cycles_p90 = np.nanpercentile(chl_EI_yearlycycles, 90, axis=0)
chl_EI_cycles_p10 = np.nanpercentile(chl_EI_yearlycycles, 10, axis=0)
plt.figure(figsize=(12,9))
plt.plot(np.arange(1,29),
         np.hstack((chl_averagecycle_EI[31:], chl_averagecycle_EI[:13])), color = 'k', linewidth = 4)
#plt.plot(np.arange(1,29),
#         np.hstack((chl_EI_cycles_19982005[31:], chl_EI_cycles_19982005[:13])), color = 'k', linewidth = 2, linestyle='--', label='1998-2005')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_EI_cycles_20062013[31:], chl_EI_cycles_20062013[:13])), color = 'k', linewidth = 2, linestyle=':', label='2006-2013')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_EI_cycles_20142020[31:], chl_EI_cycles_20142020[:13])), color = 'k', linewidth = 2, linestyle='-.', label='2014-2020')
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p10, color = 'k', linewidth = 1)
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p90, color = 'k', linewidth = 1)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_EI[31:], chl_averagecycle_EI[:13])),
                 np.hstack((chl_EI_cycles_p90[31:], chl_EI_cycles_p90[:13])),
                 color = [215/256, 25/256, 28/256, 1], alpha=.5, edgecolor = None)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_EI[31:], chl_averagecycle_EI[:13])),
                 np.hstack((chl_EI_cycles_p10[31:], chl_EI_cycles_p10[:13])),
                 color =[215/256, 25/256, 28/256, 1], alpha=.5, edgecolor = None)
plt.xlim(1,28)
#plt.ylim(0.1,1.55)
plt.xticks([1, 5, 8, 12, 16, 20, 24, 28],
           ['Sep','Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'], fontsize=12)
plt.ylabel('Chl-a (mg/m3)', fontsize=14)
plt.title('Elephant Island', fontsize=16)
#plt.legend(fontsize=16)
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\roi\\elephant_seasonalcycle_ecsa58.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
### Drake Strait
x, y = np.meshgrid(lon, lat) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(DS_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon), len(lat))
mask3d = np.repeat([mask], np.size(chl,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
chl_DS = np.ma.array(chl, mask=~mask3d)
chl_DS = np.nanmean(chl_DS, (0,1))
# Calculate average cycle
week8day_num = np.tile(np.arange(1,47),(23))[:-1]
chl_DS_df = pd.DataFrame(chl_DS, index=time_date, columns = ['chl'])
chl_DS_df['8DWeek Num'] = week8day_num
chl_averagecycle_DS = np.squeeze(chl_DS_df.groupby(['8DWeek Num']).mean().values)
### Calculate cycle for each year DS
for i in np.arange(1998, 2021):
    print(i)
    chl_DS_temp_df = chl_DS_df[time_date_years == i]
    chl_DS_temp = chl_DS_temp_df['chl'].values
    if i == 1998:
        chl_DS_yearlycycles = chl_DS_temp
    elif i == 2020:
        chl_DS_temp = np.pad(chl_DS_temp, pad_width=(0,1), mode='constant', constant_values=np.nan)
        chl_DS_yearlycycles = np.vstack((chl_DS_yearlycycles, chl_DS_temp))
    else:
        chl_DS_yearlycycles = np.vstack((chl_DS_yearlycycles, chl_DS_temp))
### Plot average cycle and each year cycle for DS
chl_DS_cycles_19982005 = np.nanmean(chl_DS_yearlycycles[:8,:], axis=0)
chl_DS_cycles_20062013 = np.nanmean(chl_DS_yearlycycles[8:16,:], axis=0)
chl_DS_cycles_20142020 = np.nanmean(chl_DS_yearlycycles[16:,:], axis=0)
chl_DS_cycles_p90 = np.nanpercentile(chl_DS_yearlycycles, 90, axis=0)
chl_DS_cycles_p10 = np.nanpercentile(chl_DS_yearlycycles, 10, axis=0)
plt.figure(figsize=(12,9))
plt.plot(np.arange(1,29),
         np.hstack((chl_averagecycle_DS[31:], chl_averagecycle_DS[:13])), color = 'k', linewidth = 4)
#plt.plot(np.arange(1,29),
#         np.hstack((chl_DS_cycles_19982005[31:], chl_DS_cycles_19982005[:13])), color = 'k', linewidth = 2, linestyle='--', label='1998-2005')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_DS_cycles_20062013[31:], chl_DS_cycles_20062013[:13])), color = 'k', linewidth = 2, linestyle=':', label='2006-2013')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_DS_cycles_20142020[31:], chl_DS_cycles_20142020[:13])), color = 'k', linewidth = 2, linestyle='-.', label='2014-2020')
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p10, color = 'k', linewidth = 1)
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p90, color = 'k', linewidth = 1)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_DS[31:], chl_averagecycle_DS[:13])),
                 np.hstack((chl_DS_cycles_p90[31:], chl_DS_cycles_p90[:13])),
                 color = [241/256, 180/256, 47/256, 1], alpha=.5, edgecolor = None)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_DS[31:], chl_averagecycle_DS[:13])),
                 np.hstack((chl_DS_cycles_p10[31:], chl_DS_cycles_p10[:13])),
                 color = [241/256, 180/256, 47/256, 1], alpha=.5, edgecolor = None)
plt.xlim(1,28)
#plt.ylim(0.1,1.55)
plt.xticks([1, 5, 8, 12, 16, 20, 24, 28],
           ['Sep','Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'], fontsize=12)
plt.ylabel('Chl-a (mg/m3)', fontsize=14)
plt.title('Drake Passage', fontsize=16)
#plt.legend(fontsize=16)
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\roi\\drake_seasonalcycle_ecsa58.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
### Weddell Sea
x, y = np.meshgrid(lon, lat) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
p = Path(WS_verts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(len(lon), len(lat))
mask3d = np.repeat([mask], np.size(chl,2), axis=0)
mask3d = np.swapaxes(mask3d, 0, 1)
mask3d = np.swapaxes(mask3d, 1, 2)
chl_WS = np.ma.array(chl, mask=~mask3d)
chl_WS = np.nanmean(chl_WS, (0,1))
# Calculate average cycle
week8day_num = np.tile(np.arange(1,47),(23))[:-1]
chl_WS_df = pd.DataFrame(chl_WS, index=time_date, columns = ['chl'])
chl_WS_df['8DWeek Num'] = week8day_num
chl_averagecycle_WS = np.squeeze(chl_WS_df.groupby(['8DWeek Num']).mean().values)
### Calculate cycle for each year WS
for i in np.arange(1998, 2021):
    print(i)
    chl_WS_temp_df = chl_WS_df[time_date_years == i]
    chl_WS_temp = chl_WS_temp_df['chl'].values
    if i == 1998:
        chl_WS_yearlycycles = chl_WS_temp
    elif i == 2020:
        chl_WS_temp = np.pad(chl_WS_temp, pad_width=(0,1), mode='constant', constant_values=np.nan)
        chl_WS_yearlycycles = np.vstack((chl_WS_yearlycycles, chl_WS_temp))
    else:
        chl_WS_yearlycycles = np.vstack((chl_WS_yearlycycles, chl_WS_temp))
### Plot average cycle and each year cycle for WS
chl_WS_cycles_19982005 = np.nanmean(chl_WS_yearlycycles[:8,:], axis=0)
chl_WS_cycles_20062013 = np.nanmean(chl_WS_yearlycycles[8:16,:], axis=0)
chl_WS_cycles_20142020 = np.nanmean(chl_WS_yearlycycles[16:,:], axis=0)
chl_WS_cycles_p90 = np.nanpercentile(chl_WS_yearlycycles, 90, axis=0)
chl_WS_cycles_p10 = np.nanpercentile(chl_WS_yearlycycles, 10, axis=0)
plt.figure(figsize=(12,9))
plt.plot(np.arange(1,29),
         np.hstack((chl_averagecycle_WS[31:], chl_averagecycle_WS[:13])), color = 'k', linewidth = 4)
#plt.plot(np.arange(1,29),
#         np.hstack((chl_WS_cycles_19982005[31:], chl_WS_cycles_19982005[:13])), color = 'k', linewidth = 2, linestyle='--', label='1998-2005')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_WS_cycles_20062013[31:], chl_WS_cycles_20062013[:13])), color = 'k', linewidth = 2, linestyle=':', label='2006-2013')
#plt.plot(np.arange(1,29),
#         np.hstack((chl_WS_cycles_20142020[31:], chl_WS_cycles_20142020[:13])), color = 'k', linewidth = 2, linestyle='-.', label='2014-2020')
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p10, color = 'k', linewidth = 1)
#plt.plot(np.arange(1,366),OcSW_yearlycicles_p90, color = 'k', linewidth = 1)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_WS[31:], chl_averagecycle_WS[:13])),
                 np.hstack((chl_WS_cycles_p90[31:], chl_WS_cycles_p90[:13])),
                 color = [43/256, 131/256, 186/256, 1], alpha=.5, edgecolor = None)
plt.fill_between(np.arange(1,29),
                 np.hstack((chl_averagecycle_WS[31:], chl_averagecycle_WS[:13])),
                 np.hstack((chl_WS_cycles_p10[31:], chl_WS_cycles_p10[:13])),
                 color = [43/256, 131/256, 186/256, 1], alpha=.5, edgecolor = None)
plt.xlim(1,28)
#plt.ylim(0.1,1.55)
plt.xticks([1, 5, 8, 12, 16, 20, 24, 28],
           ['Sep','Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'], fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Chl-a (mg/m3)', fontsize=14)
plt.title('Weddell Sea', fontsize=16)
#plt.legend(fontsize=16)
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\roi\\weddell_seasonalcycle_ecsa58.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
##