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
chl[chl > 100] = 100
# Calculate climatology
chl_clim19982020 = np.nanmean(chl, 2)
# Plot Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_clim19982020), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\climatology19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Calculate data availability for each pixel
chl_validpixels = chl[:,:,0]*np.nan
chl_validpixels_percentage = chl_validpixels
for i in range(0, len(lat)):
    for j in range(0, len(lon)):
        chl_validpixels[i,j] = np.count_nonzero(~np.isnan(chl[i, j, :]))
        chl_validpixels_percentage[i,j] = (np.count_nonzero(~np.isnan(chl[i, j, :])) / 1057)*100
# Plot valid pixels
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, chl_validpixels, transform=ccrs.PlateCarree(), shading='flat', cmap=cmocean.cm.haline)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(fontsize=14)
cbar.set_label('Valid Pixels', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chl_validpixels_9982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
# Plot valid pixels (percentage %)
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, chl_validpixels_percentage, transform=ccrs.PlateCarree(), shading='flat', cmap=cmocean.cm.haline)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, fraction=0.04, pad=0.1)
#cbar.ax.set_yticklabels(fontsize=14)
cbar.set_label('Valid Pixels (%)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\chl_validpixelspercentage_9982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
### Calculate monthly means
chl_jan19982020_clim = np.nanmean(chl[:,:, time_date_months == 1],2)
chl_feb19982020_clim = np.nanmean(chl[:,:, time_date_months == 2],2)
chl_mar19982020_clim = np.nanmean(chl[:,:, time_date_months == 3],2)
chl_apr19982020_clim = np.nanmean(chl[:,:, time_date_months == 4],2)
chl_may19982020_clim = np.nanmean(chl[:,:, time_date_months == 5],2)
chl_jun19982020_clim = np.nanmean(chl[:,:, time_date_months == 6],2)
chl_jul19982020_clim = np.nanmean(chl[:,:, time_date_months == 7],2)
chl_aug19982020_clim = np.nanmean(chl[:,:, time_date_months == 8],2)
chl_sep19982020_clim = np.nanmean(chl[:,:, time_date_months == 9],2)
chl_oct19982020_clim = np.nanmean(chl[:,:, time_date_months == 10],2)
chl_nov19982020_clim = np.nanmean(chl[:,:, time_date_months == 11],2)
chl_dec19982020_clim = np.nanmean(chl[:,:, time_date_months == 12],2)
## January Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_jan19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\january19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## February Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_feb19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\february19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## March Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_mar19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\march19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## April Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_apr19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\april19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## May Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_may19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\may19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## June Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_jun19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\june19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## July Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_jul19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\july19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## August Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_aug19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\august19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## September Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_sep19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\september19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## October Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_oct19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\october19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## November Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_nov19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\november19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
## December Climatology
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']))
f1 = map.pcolormesh(lon, lat, np.log10(chl_dec19982020_clim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                    vmax=np.log10(10), cmap=cmocean.cm.algae)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\months\\december19982020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
### Plot Yearly Climatologies
for i in np.arange(1998, 2021):
    chl_yearlyclim = np.nanmean(chl[:,:, time_date_years == i], 2)
    plt.figure()
    map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
    map.coastlines(resolution='10m', color='black', linewidth=1)
    map.set_extent([-67, -53, -67, -60])
    map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                        edgecolor='k',
                                                        facecolor=cartopy.feature.COLORS['land']))
    f1 = map.pcolormesh(lon, lat, np.log10(chl_yearlyclim), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
                        vmax=np.log10(10), cmap=cmocean.cm.algae)
    gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
    cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                        fraction=0.04, pad=0.1)
    cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
    cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
    #cbar.set_label('Valid Pixels (%)', fontsize=14)
    plt.tight_layout()
    s=""
    graphs_dir = s.join(['C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\chl', str(i), 'mean.png'])
    plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
    plt.close()
### Plot Yearly Anomalies
# Join mean for all years
for i in np.arange(1998,2021):
    chl_yeartempmean = np.nanmean(chl[:,:, time_date_years == i], 2)
    if i == 1998:
        chl_yearsmeans = chl_yeartempmean
    else:
       chl_yearsmeans = np.dstack((chl_yearsmeans, chl_yeartempmean))
# Calculate grand mean
chl_grandmean = np.nanmean(chl_yearsmeans,2)
# Plot anomalies
yearchar = np.arange(1998,2021)
for i in range(0, np.size(chl_yearsmeans,2)):
    chl_yearlyanomalytemp = chl_yearsmeans[:,:, i] - chl_grandmean
    plt.figure()
    map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
    map.coastlines(resolution='10m', color='black', linewidth=1)
    map.set_extent([-67, -53, -67, -60])
    map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                        edgecolor='k',
                                                        facecolor=cartopy.feature.COLORS['land']))
    f1 = map.pcolormesh(lon, lat, chl_yearlyanomalytemp, transform=ccrs.PlateCarree(), shading='flat', vmin=-2,
                        vmax=2, cmap=cmocean.cm.balance)
    gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
    cbar = plt.colorbar(f1)
    #cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
    cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
    #cbar.set_label('Valid Pixels (%)', fontsize=14)
    plt.tight_layout()
    s=""
    graphs_dir = s.join(['C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\years\\chl', str(yearchar[i]), '_anomaly.png'])
    plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
    plt.close()