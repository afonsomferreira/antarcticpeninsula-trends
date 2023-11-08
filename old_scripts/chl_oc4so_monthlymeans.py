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
chl_jan19972021_clim = np.nanmean(chl[:,:, time_date_months == 1],2)
chl_feb19972021_clim = np.nanmean(chl[:,:, time_date_months == 2],2)
chl_mar19972021_clim = np.nanmean(chl[:,:, time_date_months == 3],2)
chl_apr19972021_clim = np.nanmean(chl[:,:, time_date_months == 4],2)
chl_may19972021_clim = np.nanmean(chl[:,:, time_date_months == 5],2)
chl_jun19972021_clim = np.nanmean(chl[:,:, time_date_months == 6],2)
chl_jul19972021_clim = np.nanmean(chl[:,:, time_date_months == 7],2)
chl_aug19972021_clim = np.nanmean(chl[:,:, time_date_months == 8],2)
chl_sep19972021_clim = np.nanmean(chl[:,:, time_date_months == 9],2)
chl_oct19972021_clim = np.nanmean(chl[:,:, time_date_months == 10],2)
chl_nov19972021_clim = np.nanmean(chl[:,:, time_date_months == 11],2)
chl_dec19972021_clim = np.nanmean(chl[:,:, time_date_months == 12],2)
## January Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_jan19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\january19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## February Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_feb19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\february19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## March Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_mar19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\march19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## April Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_apr19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\april19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## May Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_may19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\may19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## June Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_jun19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\june19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## July Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_jul19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\july19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## August Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_aug19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\august19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## September Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_sep19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\september19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## October Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_oct19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\october19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## November Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_nov19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\november19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## December Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.set_extent([-67, -53, -67, -60])
f1 = map.pcolormesh(lon, lat, np.log10(chl_dec19972021_clim[:-1, :-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.1),
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
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\december19972021.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()