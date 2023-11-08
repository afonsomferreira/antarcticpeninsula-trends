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
chl_fullresolution_climatology = np.nanmean(chl,2)

### Increase resolution to 25km
# Set new coords
lat_new = np.arange(-48, -70, -.25)
lon_new = np.arange(-73, -25, .50)
chl_reducedresolution = np.empty((len(lat_new), len(lon_new), len(time_date)))*np.nan
# Create new matrix
for i in range(0, len(lat_new)):
    for j in range(0, len(lon_new)):
        if (i == len(lat_new)-1) & (j != len(lon_new)-1):
            lat_indices = (lat < lat_new[i])
            lon_indices = (lon > lon_new[j]) & (lon < lon_new[j+1])
        elif (i == len(lat_new)-1) & (j == len(lon_new)-1):
            lat_indices = (lat < lat_new[i])
            lon_indices = (lon > lon_new[j])                
        elif (i != len(lat_new)-1) & (j == len(lon_new)-1):
            lat_indices = (lat < lat_new[i]) & (lat > lat_new[i+1])
            lon_indices = (lon > lon_new[j])
        else:                  
            lat_indices = (lat < lat_new[i]) & (lat > lat_new[i+1])
            lon_indices = (lon > lon_new[j]) & (lon < lon_new[j+1])
        chl_reducedresolution[i, j, :] = np.nanmean(chl[lat_indices, lon_indices,:], 0)

chl_reducedresolution_climatology = np.nanmean(chl_reducedresolution,2)
chl_reducedresolution_climatology_T = np.transpose(chl_reducedresolution_climatology)

plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']), zorder=1)
#f1 = map.hexbin(lon_new, lat_new, c=chl_reducedresolution_climatology, gridsize=(250))
#f1 = map.hexbin(lon_new, lat_new, np.log10(chl_reducedresolution_climatology), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
#                    vmax=np.log10(10), cmap=cmocean.cm.algae)
#f1 = map.pcolormesh(lon_new, lat_new, chl_reducedresolution_climatology_T[:-1, :-1].T, transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
#                    vmax=np.log10(10), cmap=cmocean.cm.algae)
#f1 = map.pcolor(lon_new, lat_new, chl_reducedresolution_climatology_T[:-1, :-1].T, transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
#                    vmax=np.log10(10), cmap=cmocean.cm.algae)
#f1 = map.pcolormesh(lon_new, lat_new, np.log10(chl_reducedresolution_climatology[:-1,:-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
#                    vmax=np.log10(10), cmap=cmocean.cm.algae, zorder=0)
#f1 = map.pcolormesh(lon, lat, np.log10(chl_fullresolution_climatology[:-1,:-1]), transform=ccrs.PlateCarree(), shading='flat', vmin=np.log10(0.01),
#                    vmax=np.log10(10), cmap=cmocean.cm.algae, zorder=0)
#plt.scatter(lon_new+0.125,lat_new+0.125, color = 'red')
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1, ticks=[np.log10(0.01), np.log10(0.05), np.log10(0.1), np.log10(0.5), np.log10(1), np.log10(3), np.log10(10)],
                    fraction=0.04, pad=0.1)
cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()

#plt.pcolormesh(lon_new, lat_new, np.nanmean(chl_reducedresolution,2))
### Calculate summer means (December-February) for each year
for i in np.arange(1999, 2021):
    yeartemp_dec = chl_reducedresolution[:,:, (time_date_years == i-1) & (time_date_months == 12)]
    yeartemp_jan = chl_reducedresolution[:,:, (time_date_years == i) & (time_date_months == 1)]
    yeartemp_feb = chl_reducedresolution[:,:, (time_date_years == i) & (time_date_months == 2)]
    yeartemp_summermean = np.nanmean(np.dstack((yeartemp_dec, yeartemp_jan, yeartemp_feb)),2)
    if i == 1999:
        years_summermeans = yeartemp_summermean
    else:
        years_summermeans = np.dstack((years_summermeans, yeartemp_summermean))
### Calculate linear trend for each pixel
yearchar = np.arange(1999, 2021)
chl_reducedresolution_summertrend19992020 = np.empty((np.size(lat_new), np.size(lon_new)))*np.nan
chl_reducedresolution_summertrend19992020_significant = np.empty((np.size(lat_new), np.size(lon_new)))*np.nan

for i in range(0, len(lat_new)):
    for j in range(0, len(lon_new)):
        yearchar = np.arange(1999, 2021)
        chl_summerpixel = years_summermeans[i,j,:]
        # keep trend to nan if most years do not have data
        if len(chl_summerpixel[np.isnan(chl_summerpixel)]) > 15:
            continue
        # remove nans if there exist any
        if len(np.isnan(chl_summerpixel)) > 0:
            nans_chl_summerpixel = np.isnan(chl_summerpixel)
            chl_summerpixel = chl_summerpixel[~nans_chl_summerpixel]
            yearchar = yearchar[~nans_chl_summerpixel]
        # calculate linear regression
        slope, _, rvalue, pvalue, _ = stats.linregress(yearchar, chl_summerpixel)
        # add slope value (i.e. change in chl per year) to pixel
        #if pvalue < 0.05:
        chl_reducedresolution_summertrend19992020[i,j] = slope
        if pvalue < 0.05:
            chl_reducedresolution_summertrend19992020_significant[i,j] = 1            
        
# Plot Climatology
chl_reducedresolution_summertrend19992020[chl_reducedresolution_summertrend19992020>1] = np.nan
chl_reducedresolution_summertrend19992020[chl_reducedresolution_summertrend19992020<-1] = np.nan
plt.figure()
map = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-60, central_latitude=-62))
map.coastlines(resolution='10m', color='black', linewidth=1)
map.set_extent([-67, -53, -67, -60])
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor=cartopy.feature.COLORS['land']), zorder=2)
f1 = map.pcolormesh(lon_new, lat_new, chl_reducedresolution_summertrend19992020[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmocean.cm.balance,
                    vmin=-.1, vmax=.1, zorder=0)
hatch = map.fill_between([lon_new[0], lon_new[-1]],lat_new[0],lat_new[-1],hatch='....',color="none",edgecolor='black', transform=ccrs.PlateCarree())
z_masked = np.ma.masked_where(chl_reducedresolution_summertrend19992020_significant == 1, chl_reducedresolution_summertrend19992020)
mesh_masked = map.pcolormesh(lon_new, lat_new, z_masked[:-1, :-1], transform=ccrs.PlateCarree(), shading='flat', cmap=cmocean.cm.balance,
                    vmin=-.1, vmax=.1, zorder=1)
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black')
cbar = plt.colorbar(f1)
#cbar.ax.set_yticklabels(['0.01', '0.05', '0.1', '0.5', '1', '3', '10'], fontsize=14)
cbar.set_label('Yearly increase in Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\chl\\summertrend_19992020.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()