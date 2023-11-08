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
# Plot Climatology
plt.figure()
map = plt.axes(projection=ccrs.SouthPolarStereo())
map.set_extent([-67, -53, -67, -60])
map.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
gl = map.gridlines(draw_labels=True, alpha=0.5, linestyle='dotted', color='black', linewidth=2)
#cbar.set_label('Valid Pixels (%)', fontsize=14)
map.coastlines(resolution='10m', color='black', linewidth=1)
map.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='grey'))
#map.plot([180, -180], [-60, -60] , transform=ccrs.PlateCarree())
ROI_verts = [(-67, -60), (-53, -60), (-53, -67), (-67, -67)]
poly_ROI = Polygon(list(ROI_verts), facecolor=[255/255,170/255,29/255,0.3], edgecolor='k', linewidth=1, linestyle='-', zorder=2, transform=ccrs.PlateCarree())
plt.gca().add_patch(poly_ROI)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\figures_tentative\\figure1inset.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
