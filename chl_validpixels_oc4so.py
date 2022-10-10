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
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarctic-furseal-2021\\resources\\oc4so-chl\\')
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
### Load data 1998-2020
fh = np.load('chloc4so_19972021.npz', allow_pickle=True)
lat = fh['lat']
lon = fh['lon']
chl = fh['chl']
time_date = fh['time_date']
time_date_years = np.empty_like(time_date)
time_date_months = np.empty_like(time_date)
for i in range(0, len(time_date)):
    time_date_years[i] = time_date[i].year
    time_date_months[i] = time_date[i].month
# Correct values
chl[chl > 100] = 100
# Calculate global data availability for each pixel
chl_validpixels = chl[:,:,0]*np.nan
chl_validpixels_percentage = chl_validpixels
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels[i,j] = np.count_nonzero(~np.isnan(chl[i, j, :]))
        chl_validpixels_percentage[i,j] = (np.count_nonzero(~np.isnan(chl[i, j, :])) / len(time_date))*100
# Plot Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 10)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage, chl_validpixels)
# Calculate monthly data availability for each pixel
# January
chl_january = chl[:,:, time_date_months == 1]
chl_validpixels_january = chl[:,:,0]*np.nan
chl_validpixels_percentage_january = chl_validpixels_january
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_january[i,j] = np.count_nonzero(~np.isnan(chl_january[i, j, :]))
        chl_validpixels_percentage_january[i,j] = (np.count_nonzero(~np.isnan(chl_january[i, j, :])) / np.size(chl_january,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_january[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_january.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_january, chl_validpixels_january)
# February
chl_february = chl[:,:, time_date_months == 2]
chl_validpixels_february = chl[:,:,0]*np.nan
chl_validpixels_percentage_february = chl_validpixels_february
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_february[i,j] = np.count_nonzero(~np.isnan(chl_february[i, j, :]))
        chl_validpixels_percentage_february[i,j] = (np.count_nonzero(~np.isnan(chl_february[i, j, :])) / np.size(chl_february,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_february[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_february.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_february, chl_validpixels_february)
# March
chl_march = chl[:,:, time_date_months == 3]
chl_validpixels_march = chl[:,:,0]*np.nan
chl_validpixels_percentage_march = chl_validpixels_march
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_march[i,j] = np.count_nonzero(~np.isnan(chl_march[i, j, :]))
        chl_validpixels_percentage_march[i,j] = (np.count_nonzero(~np.isnan(chl_march[i, j, :])) / np.size(chl_march,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_march[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_march.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_march, chl_validpixels_march)
# April
chl_april = chl[:,:, time_date_months == 4]
chl_validpixels_april = chl[:,:,0]*np.nan
chl_validpixels_percentage_april = chl_validpixels_april
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_april[i,j] = np.count_nonzero(~np.isnan(chl_april[i, j, :]))
        chl_validpixels_percentage_april[i,j] = (np.count_nonzero(~np.isnan(chl_april[i, j, :])) / np.size(chl_april,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_april[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_april.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_april, chl_validpixels_april)
# May
chl_may = chl[:,:, time_date_months == 5]
chl_validpixels_may = chl[:,:,0]*np.nan
chl_validpixels_percentage_may = chl_validpixels_may
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_may[i,j] = np.count_nonzero(~np.isnan(chl_may[i, j, :]))
        chl_validpixels_percentage_may[i,j] = (np.count_nonzero(~np.isnan(chl_may[i, j, :])) / np.size(chl_may,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_may[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_may.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_may, chl_validpixels_may)
# June
chl_june = chl[:,:, time_date_months == 6]
chl_validpixels_june = chl[:,:,0]*np.nan
chl_validpixels_percentage_june = chl_validpixels_june
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_june[i,j] = np.count_nonzero(~np.isnan(chl_june[i, j, :]))
        chl_validpixels_percentage_june[i,j] = (np.count_nonzero(~np.isnan(chl_june[i, j, :])) / np.size(chl_june,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_june[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_june.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_june, chl_validpixels_june)
# July
chl_july = chl[:,:, time_date_months == 7]
chl_validpixels_july = chl[:,:,0]*np.nan
chl_validpixels_percentage_july = chl_validpixels_july
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_july[i,j] = np.count_nonzero(~np.isnan(chl_july[i, j, :]))
        chl_validpixels_percentage_july[i,j] = (np.count_nonzero(~np.isnan(chl_july[i, j, :])) / np.size(chl_july,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_july[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_july.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_july, chl_validpixels_july)
# August
chl_august = chl[:,:, time_date_months == 8]
chl_validpixels_august = chl[:,:,0]*np.nan
chl_validpixels_percentage_august = chl_validpixels_august
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_august[i,j] = np.count_nonzero(~np.isnan(chl_august[i, j, :]))
        chl_validpixels_percentage_august[i,j] = (np.count_nonzero(~np.isnan(chl_august[i, j, :])) / np.size(chl_august,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_august[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_august.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_august, chl_validpixels_august)
# September
chl_september = chl[:,:, time_date_months == 9]
chl_validpixels_september = chl[:,:,0]*np.nan
chl_validpixels_percentage_september = chl_validpixels_september
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_september[i,j] = np.count_nonzero(~np.isnan(chl_september[i, j, :]))
        chl_validpixels_percentage_september[i,j] = (np.count_nonzero(~np.isnan(chl_september[i, j, :])) / np.size(chl_september,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_september[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_september.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_september, chl_validpixels_september)
# October
chl_october = chl[:,:, time_date_months == 10]
chl_validpixels_october = chl[:,:,0]*np.nan
chl_validpixels_percentage_october = chl_validpixels_october
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_october[i,j] = np.count_nonzero(~np.isnan(chl_october[i, j, :]))
        chl_validpixels_percentage_october[i,j] = (np.count_nonzero(~np.isnan(chl_october[i, j, :])) / np.size(chl_october,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_october[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_october.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_october, chl_validpixels_october)
# November
chl_november = chl[:,:, time_date_months == 11]
chl_validpixels_november = chl[:,:,0]*np.nan
chl_validpixels_percentage_november = chl_validpixels_november
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_november[i,j] = np.count_nonzero(~np.isnan(chl_november[i, j, :]))
        chl_validpixels_percentage_november[i,j] = (np.count_nonzero(~np.isnan(chl_november[i, j, :])) / np.size(chl_november,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_november[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_november.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_november, chl_validpixels_november)
# December
chl_december = chl[:,:, time_date_months == 12]
chl_validpixels_december = chl[:,:,0]*np.nan
chl_validpixels_percentage_december = chl_validpixels_december
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels_december[i,j] = np.count_nonzero(~np.isnan(chl_december[i, j, :]))
        chl_validpixels_percentage_december[i,j] = (np.count_nonzero(~np.isnan(chl_december[i, j, :])) / np.size(chl_december,2))*100
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage_december[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=plt.cm.inferno)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1.set_clim(0, 25)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\validpixels_19972021_december.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
del(chl_validpixels_percentage_december, chl_validpixels_december)