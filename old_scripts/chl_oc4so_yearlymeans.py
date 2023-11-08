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
### Calculate monthly means
chl_1998_clim = np.nanmean(chl[:,:, time_date_years == 1998],2)
chl_1999_clim = np.nanmean(chl[:,:, time_date_years == 1999],2)
chl_2000_clim = np.nanmean(chl[:,:, time_date_years == 2000],2)
chl_2001_clim = np.nanmean(chl[:,:, time_date_years == 2001],2)
chl_2002_clim = np.nanmean(chl[:,:, time_date_years == 2002],2)
chl_2003_clim = np.nanmean(chl[:,:, time_date_years == 2003],2)
chl_2004_clim = np.nanmean(chl[:,:, time_date_years == 2004],2)
chl_2005_clim = np.nanmean(chl[:,:, time_date_years == 2005],2)
chl_2006_clim = np.nanmean(chl[:,:, time_date_years == 2006],2)
chl_2007_clim = np.nanmean(chl[:,:, time_date_years == 2007],2)
chl_2008_clim = np.nanmean(chl[:,:, time_date_years == 2008],2)
chl_2009_clim = np.nanmean(chl[:,:, time_date_years == 2009],2)
chl_2010_clim = np.nanmean(chl[:,:, time_date_years == 2010],2)
chl_2011_clim = np.nanmean(chl[:,:, time_date_years == 2011],2)
chl_2012_clim = np.nanmean(chl[:,:, time_date_years == 2012],2)
chl_2013_clim = np.nanmean(chl[:,:, time_date_years == 2013],2)
chl_2014_clim = np.nanmean(chl[:,:, time_date_years == 2014],2)
chl_2015_clim = np.nanmean(chl[:,:, time_date_years == 2015],2)
chl_2016_clim = np.nanmean(chl[:,:, time_date_years == 2016],2)
chl_2017_clim = np.nanmean(chl[:,:, time_date_years == 2017],2)
chl_2018_clim = np.nanmean(chl[:,:, time_date_years == 2018],2)
chl_2019_clim = np.nanmean(chl[:,:, time_date_years == 2019],2)
chl_2020_clim = np.nanmean(chl[:,:, time_date_years == 2020],2)
chl_2021_clim = np.nanmean(chl[:,:, time_date_years == 2021],2)

## 1998 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_1998_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\1998.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 1999 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_1999_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\1999.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2000 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2000_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2000.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2001 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2001_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2001.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2002 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2002_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2002.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2003 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2003_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2003.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2004 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2004_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2004.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2005 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2005_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2005.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2006 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2006_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2006.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2007 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2007_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2007.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2008 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2008_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2008.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2009 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2009_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2009.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2010 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2010_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2010.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2011 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2011_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2011.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2012 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2012_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2012.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2013 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2013_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2013.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2014 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2014_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2014.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2015 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2015_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2015.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2016 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2016_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2016.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2017 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2017_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2017.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2018 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2018_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2018.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2019 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2019_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2019.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2020 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2020_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## 2021 Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_2021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\2021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()




