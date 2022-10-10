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
years = np.arange(1997, 2022)
months = np.arange(1, 13)
months_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                'September', 'October', 'November', 'December']
s=''
# Plot mean chlorophyll map for each month per each year
for i in years:
    if i == 1997:
        sep1997_mean = np.nanmean(chl[:,:, (time_date_months == 9) & (time_date_years == 1997)],2)
        oct1997_mean = np.nanmean(chl[:,:, (time_date_months == 10) & (time_date_years == 1997)],2)
        nov1997_mean = np.nanmean(chl[:,:, (time_date_months == 11) & (time_date_years == 1997)],2)        
        dec1997_mean = np.nanmean(chl[:,:, (time_date_months == 12) & (time_date_years == 1997)],2)
        # Sep 1997
        plt.figure()
        map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
        map.set_extent([-67, -53, -67, -60])
        f1 = map.pcolormesh(lon, lat, np.log10(sep1997_mean[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
                            vmax=np.log10(10), cmap=plt.cm.viridis)
        gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
        cbar = plt.colorbar(f1, ticks=[np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                            fraction=0.04, pad=0.1)
        cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
        cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
        map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                            edgecolor='k',
                                                            facecolor=cartopy.feature.COLORS['land']))
        map.coastlines(resolution='10m', color='black', linewidth=1)
        #cbar.set_label('Valid Pixels (%)', fontsize=14)
        plt.title('September 1997')
        plt.tight_layout()
        graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\monthsperyear\\September1997.png'
        plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 150)
        plt.close()
        # Oct 1997
        plt.figure()
        map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
        map.set_extent([-67, -53, -67, -60])
        f1 = map.pcolormesh(lon, lat, np.log10(oct1997_mean[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
                            vmax=np.log10(10), cmap=plt.cm.viridis)
        gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
        cbar = plt.colorbar(f1, ticks=[np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                            fraction=0.04, pad=0.1)
        cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
        cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
        map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                            edgecolor='k',
                                                            facecolor=cartopy.feature.COLORS['land']))
        map.coastlines(resolution='10m', color='black', linewidth=1)
        #cbar.set_label('Valid Pixels (%)', fontsize=14)
        plt.title('October 1997')
        plt.tight_layout()
        graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\monthsperyear\\October1997.png'
        plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 150)
        plt.close()
        # Nov 1997
        plt.figure()
        map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
        map.set_extent([-67, -53, -67, -60])
        f1 = map.pcolormesh(lon, lat, np.log10(nov1997_mean[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
                            vmax=np.log10(10), cmap=plt.cm.viridis)
        gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
        cbar = plt.colorbar(f1, ticks=[np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                            fraction=0.04, pad=0.1)
        cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
        cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
        map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                            edgecolor='k',
                                                            facecolor=cartopy.feature.COLORS['land']))
        map.coastlines(resolution='10m', color='black', linewidth=1)
        #cbar.set_label('Valid Pixels (%)', fontsize=14)
        plt.title('November 1997')
        plt.tight_layout()
        graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\monthsperyear\\November1997.png'
        plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 150)
        plt.close()
        # Dec 1997
        plt.figure()
        map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
        map.set_extent([-67, -53, -67, -60])
        f1 = map.pcolormesh(lon, lat, np.log10(dec1997_mean[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
                            vmax=np.log10(10), cmap=plt.cm.viridis)
        gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
        cbar = plt.colorbar(f1, ticks=[np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                            fraction=0.04, pad=0.1)
        cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
        cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
        map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                            edgecolor='k',
                                                            facecolor=cartopy.feature.COLORS['land']))
        map.coastlines(resolution='10m', color='black', linewidth=1)
        #cbar.set_label('Valid Pixels (%)', fontsize=14)
        plt.title('December 1997')
        plt.tight_layout()
        graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\monthsperyear\\December1997.png'
        plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 150)
        plt.close()
        del(sep1997_mean, oct1997_mean, nov1997_mean, dec1997_mean)
        continue
    else:
        for k, item in enumerate(months):
            month_mean = np.nanmean(chl[:,:, (time_date_months == item) & (time_date_years == i)],2)
            # Month mean
            plt.figure()
            map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
            map.set_extent([-67, -53, -67, -60])
            f1 = map.pcolormesh(lon, lat, np.log10(month_mean[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
                                vmax=np.log10(10), cmap=plt.cm.viridis)
            gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
            cbar = plt.colorbar(f1, ticks=[np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                                fraction=0.04, pad=0.1)
            cbar.ax.set_yticklabels(['0.1', '0.5', '1', '3', '10'], fontsize=14)
            cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
            map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                                edgecolor='k',
                                                                facecolor=cartopy.feature.COLORS['land']))
            map.coastlines(resolution='10m', color='black', linewidth=1)
            #cbar.set_label('Valid Pixels (%)', fontsize=14)
            plt.title(s.join([str(months_names[k]), ' ', str(i)]))
            plt.tight_layout()
            graphs_dir = s.join(['C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\monthsperyear\\', str(months_names[k]), str(i), '.png'])
            plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 150)
            plt.close()