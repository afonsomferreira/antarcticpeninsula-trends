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
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.path import Path
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from netCDF4 import Dataset
from sktime.transformations.series.outlier_detection import HampelFilter
import datetime
import cmocean
import dtw as dtw
from scipy import integrate
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
def serial_date_to_string(srl_no):
    """Converts serial number time to datetime"""
    new_date = datetime.datetime(1981, 1, 1, 0, 0) + datetime.timedelta(seconds=srl_no)
    return new_date
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.nanmedian(points, axis=0)
    diff = np.nansum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.nanmedian(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
#%%
ix = pd.date_range(start=datetime.date(1998, 8, 1), end=datetime.date(1999, 5, 31), freq='D')
a_df = pd.DataFrame(np.arange(1, 305), index=ix)
a_df_8day = a_df.resample('8D').mean()
#%% Phenology DRA
os.chdir('C:\\Users\\afons\\OneDrive - Universidade de Lisboa\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('phenology_DRA_10km.npz', allow_pickle=True)
dra_binit = fh['b_init']
dra_bterm = fh['b_term']
dra_bpeak = fh['b_peak']
dra_chlmax = fh['chl_max']
dra_barea = fh['b_area']
dra_bdur = fh['b_dur']
dra_timeyears = fh['time_years']

a_df_8day.index[13]
#%% BRS
fh = np.load('phenology_BRS_10km.npz', allow_pickle=True)
brs_binit = fh['b_init']
brs_bterm = fh['b_term']
brs_bpeak = fh['b_peak']
brs_chlmax = fh['chl_max']
brs_barea = fh['b_area']
brs_bdur = fh['b_dur']
brs_timeyears = fh['time_years']
#%% Phenology WEDi
fh = np.load('phenology_WEDI_10km.npz', allow_pickle=True)
wedi_binit = fh['b_init']
wedi_bterm = fh['b_term']
wedi_bpeak = fh['b_peak']
wedi_chlmax = fh['chl_max']
wedi_barea = fh['b_area']
wedi_bdur = fh['b_dur']
wedi_timeyears = fh['time_years']
#%% Phenology WEDo
fh = np.load('phenology_WEDO_10km.npz', allow_pickle=True)
wedo_binit = fh['b_init']
wedo_bterm = fh['b_term']
wedo_bpeak = fh['b_peak']
wedo_chlmax = fh['chl_max']
wedo_barea = fh['b_area']
wedo_bdur = fh['b_dur']
wedo_timeyears = fh['time_years']
#%% Phenology GES
fh = np.load('phenology_GES_10km.npz', allow_pickle=True)
ges_binit = fh['b_init']
ges_bterm = fh['b_term']
ges_bpeak = fh['b_peak']
ges_chlmax = fh['chl_max']
ges_barea = fh['b_area']
ges_bdur = fh['b_dur']
ges_timeyears = fh['time_years']





#%%

# Calculate linear trends for each metric
# B Init
slope_dra_binit, _, rvalue_dra_binit, pvalue_dra_binit, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(dra_timeyears)], dra_binit[~np.isnan(dra_timeyears)])
# B Term
slope_dra_bterm, _, rvalue_dra_bterm, pvalue_dra_bterm, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(dra_timeyears)], dra_bterm[~np.isnan(dra_timeyears)])
# B Peak
slope_dra_bpeak, _, rvalue_dra_bpeak, pvalue_dra_bpeak, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(dra_timeyears)], dra_bpeak[~np.isnan(dra_timeyears)])
# Chl Max
slope_dra_chlmax, _, rvalue_dra_chlmax, pvalue_dra_chlmax, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(dra_timeyears)], dra_chlmax[~np.isnan(dra_timeyears)])
# B Area
slope_dra_barea, _, rvalue_dra_barea, pvalue_dra_barea, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(dra_timeyears)], dra_barea[~np.isnan(dra_timeyears)])
# B Dur
slope_dra_bdur, _, rvalue_dra_bdur, pvalue_dra_bdur, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(dra_timeyears)], dra_bdur[~np.isnan(dra_timeyears)])
# Prepare data for heatmap
dra_slopes = np.hstack((slope_dra_binit, slope_dra_bterm, slope_dra_bpeak,
                            slope_dra_barea, slope_dra_bdur))
dra_slopes = np.around(dra_slopes, 3)
#%% Phenology BRS
fh = np.load('phenology_BRS_10km.npz', allow_pickle=True)
brs_binit = fh['b_init']
brs_bterm = fh['b_term']
brs_bpeak = fh['b_peak']
brs_chlmax = fh['chl_max']
brs_barea = fh['b_area']
brs_bdur = fh['b_dur']
brs_timeyears = fh['time_years']
# Calculate linear trends for each metric
# B Init
slope_brs_binit, _, rvalue_brs_binit, pvalue_brs_binit, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_timeyears)], brs_binit[~np.isnan(brs_timeyears)])
# B Term
slope_brs_bterm, _, rvalue_brs_bterm, pvalue_brs_bterm, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_timeyears)], brs_bterm[~np.isnan(brs_timeyears)])
# B Peak
slope_brs_bpeak, _, rvalue_brs_bpeak, pvalue_brs_bpeak, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_timeyears)], brs_bpeak[~np.isnan(brs_timeyears)])
# Chl Max
slope_brs_chlmax, _, rvalue_brs_chlmax, pvalue_brs_chlmax, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_timeyears)], brs_chlmax[~np.isnan(brs_timeyears)])
# B Area
slope_brs_barea, _, rvalue_brs_barea, pvalue_brs_barea, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_timeyears)], brs_barea[~np.isnan(brs_timeyears)])
# B Dur
slope_brs_bdur, _, rvalue_brs_bdur, pvalue_brs_bdur, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_timeyears)], brs_bdur[~np.isnan(brs_timeyears)])
# Prepare data for heatmap
brs_slopes = np.hstack((slope_brs_binit, slope_brs_bterm, slope_brs_bpeak,
                            slope_brs_barea, slope_brs_bdur))
brs_slopes = np.around(brs_slopes, 3)
#%% Phenology WEDi
fh = np.load('phenology_WEDI_10km.npz', allow_pickle=True)
wedi_binit = fh['b_init']
wedi_bterm = fh['b_term']
wedi_bpeak = fh['b_peak']
wedi_chlmax = fh['chl_max']
wedi_barea = fh['b_area']
wedi_bdur = fh['b_dur']
wedi_timeyears = fh['time_years']
# Calculate linear trends for each metric
# B Init
slope_wedi_binit, _, rvalue_wedi_binit, pvalue_wedi_binit, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedi_timeyears)], wedi_binit[~np.isnan(wedi_timeyears)])
# B Term
slope_wedi_bterm, _, rvalue_wedi_bterm, pvalue_wedi_bterm, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedi_timeyears)], wedi_bterm[~np.isnan(wedi_timeyears)])
# B Peak
slope_wedi_bpeak, _, rvalue_wedi_bpeak, pvalue_wedi_bpeak, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedi_timeyears)], wedi_bpeak[~np.isnan(wedi_timeyears)])
# Chl Max
slope_wedi_chlmax, _, rvalue_wedi_chlmax, pvalue_wedi_chlmax, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedi_timeyears)], wedi_chlmax[~np.isnan(wedi_timeyears)])
# B Area
slope_wedi_barea, _, rvalue_wedi_barea, pvalue_wedi_barea, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedi_timeyears)], wedi_barea[~np.isnan(wedi_timeyears)])
# B Dur
slope_wedi_bdur, _, rvalue_wedi_bdur, pvalue_wedi_bdur, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedi_timeyears)], wedi_bdur[~np.isnan(wedi_timeyears)])
# Prepare data for heatmap
wedi_slopes = np.hstack((slope_wedi_binit, slope_wedi_bterm, slope_wedi_bpeak,
                            slope_wedi_barea, slope_wedi_bdur))
wedi_slopes = np.around(wedi_slopes, 3)
#%% Phenology GES
fh = np.load('phenology_GES_10km.npz', allow_pickle=True)
ges_binit = fh['b_init']
ges_bterm = fh['b_term']
ges_bpeak = fh['b_peak']
ges_chlmax = fh['chl_max']
ges_barea = fh['b_area']
ges_bdur = fh['b_dur']
ges_timeyears = fh['time_years']
# Calculate linear trends for each metric
# B Init
slope_ges_binit, _, rvalue_ges_binit, pvalue_ges_binit, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_timeyears)], ges_binit[~np.isnan(ges_timeyears)])
# B Term
slope_ges_bterm, _, rvalue_ges_bterm, pvalue_ges_bterm, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_timeyears)], ges_bterm[~np.isnan(ges_timeyears)])
# B Peak
slope_ges_bpeak, _, rvalue_ges_bpeak, pvalue_ges_bpeak, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_timeyears)], ges_bpeak[~np.isnan(ges_timeyears)])
# Chl Max
slope_ges_chlmax, _, rvalue_ges_chlmax, pvalue_ges_chlmax, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_timeyears)], ges_chlmax[~np.isnan(ges_timeyears)])
# B Area
slope_ges_barea, _, rvalue_ges_barea, pvalue_ges_barea, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_timeyears)], ges_barea[~np.isnan(ges_timeyears)])
# B Dur
slope_ges_bdur, _, rvalue_ges_bdur, pvalue_ges_bdur, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(ges_timeyears)], ges_bdur[~np.isnan(ges_timeyears)])
# Prepare data for heatmap
ges_slopes = np.hstack((slope_ges_binit, slope_ges_bterm, slope_ges_bpeak,
                            slope_ges_barea, slope_ges_bdur))
ges_slopes = np.around(ges_slopes, 3)
#%% Phenology WEDo
fh = np.load('phenology_WEDO_10km.npz', allow_pickle=True)
wedo_binit = fh['b_init']
wedo_bterm = fh['b_term']
wedo_bpeak = fh['b_peak']
wedo_chlmax = fh['chl_max']
wedo_barea = fh['b_area']
wedo_bdur = fh['b_dur']
wedo_timeyears = fh['time_years']
# Calculate linear trends for each metric
# B Init
slope_wedo_binit, _, rvalue_wedo_binit, pvalue_wedo_binit, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedo_timeyears)], wedo_binit[~np.isnan(wedo_timeyears)])
# B Term
slope_wedo_bterm, _, rvalue_wedo_bterm, pvalue_wedo_bterm, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedo_timeyears)], wedo_bterm[~np.isnan(wedo_timeyears)])
# B Peak
slope_wedo_bpeak, _, rvalue_wedo_bpeak, pvalue_wedo_bpeak, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedo_timeyears)], wedo_bpeak[~np.isnan(wedo_timeyears)])
# Chl Max
slope_wedo_chlmax, _, rvalue_wedo_chlmax, pvalue_wedo_chlmax, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedo_timeyears)], wedo_chlmax[~np.isnan(wedo_timeyears)])
# B Area
slope_wedo_barea, _, rvalue_wedo_barea, pvalue_wedo_barea, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedo_timeyears)], wedo_barea[~np.isnan(wedo_timeyears)])
# B Dur
slope_wedo_bdur, _, rvalue_wedo_bdur, pvalue_wedo_bdur, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(wedo_timeyears)], wedo_bdur[~np.isnan(wedo_timeyears)])
# Prepare data for heatmap
wedo_slopes = np.hstack((slope_wedo_binit, slope_wedo_bterm, slope_wedo_bpeak,
                            slope_wedo_barea, slope_wedo_bdur))
wedo_slopes = np.around(wedo_slopes, 3)
#%%
annot_ori = [[f"{val:.3f}"  for val in row] for row in [dra_slopes, brs_slopes, wedi_slopes, ges_slopes, wedo_slopes]]
#DRA
annot_ori[0][2] = annot_ori[0][2] + '\n★'
#BRS
annot_ori[1][0] = annot_ori[1][0] + '\n★' + '★'
annot_ori[1][1] = annot_ori[1][1] + '\n★' + '★' + '★'
#annot_ori[1][3] = annot_ori[1][3] + '\n★' + '★'
annot_ori[1][3] = annot_ori[1][3] + '\n★' + '★'
#WEDi
#GES
annot_ori[3][1] = annot_ori[3][1] + '\n★'
#annot_ori[3][3] = annot_ori[3][3] + '\n★'
#WEDo
annot_ori[4][1] = annot_ori[4][1] + '\n★' + '★'
annot_ori[4][3] = annot_ori[4][3] + '\n★' + '★'
#%%
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap([dra_slopes, brs_slopes, wedi_slopes, ges_slopes, wedo_slopes], square=True, annot=annot_ori, fmt='',
            vmin=-.25, vmax=.25, cmap=plt.cm.seismic, cbar_kws={"fraction":0.045, "pad":0.05}, ax=ax)
plt.yticks(ticks=np.arange(0.5, 5), labels=['DRA', 'BRS', 'WED$_\mathrm{I}$', 'GES', 'WED$_\mathrm{O}$'], fontsize=12, rotation = 360)
plt.xticks(ticks=np.arange(0.5, 5), labels=['BINIT', 'BTERM', 'BPEAK', 'BAREA', 'BDUR'], fontsize=12)
plt.xlabel('Period', fontsize=14)
plt.ylabel('Region', fontsize=14)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\phenologytrends.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% BRS Binit
slope_BRS_binit_19982021, intercept_BRS_binit_19982021, _, _, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_binit)], brs_binit[~np.isnan(brs_binit)])
#slope_BRS_bterm_19982021, intercept_BRS_bterm_19982021, _, _, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_bterm)], brs_bterm[~np.isnan(brs_bterm)])




#slope_BRS_apr_19982021, intercept_BRS_apr_19982021, _, _, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(BRS_apr_19982021)], BRS_apr_19982021[~np.isnan(BRS_apr_19982021)])
#slope_GES_mar_19982021, intercept_GES_mar_19982021, _, _, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(GES_mar_19982021)], GES_mar_19982021[~np.isnan(GES_mar_19982021)])
#slope_GES_sep_19982021, intercept_GES_sep_19982021, _, _, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(GES_sep_19982021)], GES_sep_19982021[~np.isnan(GES_sep_19982021)])

plt.figure(figsize=(10, 5))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
plt.scatter(np.arange(1998, 2022), brs_binit, c = [106/256, 153/256, 78/256, 1], linewidth = 2, label='BRS-BINIT', zorder=2, marker='o', s=150, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), brs_bterm, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='BRS-BTERM', zorder=2, marker='o', s=50, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), BRS_decfeb_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-DECFEB', zorder=2, marker='^', s=25)
#plt.scatter(np.arange(1998, 2022), BRS_sepapr_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-SEPAPR', zorder=2, marker='*', s=25)
#plt.scatter(np.arange(1998, 2022), GES_sep_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-SEP', zorder=2, marker='o', s=50, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), GES_mar_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-MAR', zorder=2, marker='o', s=50, alpha=0.5)
plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_binit_19982021 + intercept_BRS_binit_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_apr_19982021 + intercept_BRS_apr_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_mar_19982021 + intercept_GES_mar_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_sep_19982021 + intercept_GES_sep_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.xticks(ticks= np.arange(1998, 2022), labels=['98', '99', '00', '01', '02', '03', '04', '05', '06', '07',
#                                                 '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
#                                                 '18', '19', '20', '21'], fontsize=10)
plt.yticks(fontsize=14)
plt.xticks(np.arange(1998,2023,4),fontsize=14)
#plt.ylim(0.2,1.6)
#plt.ylim(0.4,3.5)
plt.yticks(ticks=[12,15,18,21])#, labels=['1.0', '2.0', '3.0'])
#plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
#plt.legend(fontsize=12, loc=0)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\brs_binit_trendline.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()

#%% BRS BTerm
slope_BRS_bterm_19982021, intercept_BRS_bterm_19982021, _, _, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_bterm)], brs_bterm[~np.isnan(brs_bterm)])
BRS_bterm_regimeshift = [26.8, 26.8,26.8,26.8,26.8,26.8,26.8,26.8,26.8,26.8,26.8,26.8,
                         30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08,]
plt.figure(figsize=(10, 5))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
#plt.scatter(np.arange(1998, 2022), brs_binit, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='BRS-BINIT', zorder=2, marker='o', s=50, alpha=0.5)
plt.scatter(np.arange(1998, 2022), brs_bterm, c = [106/256, 153/256, 78/256, 1], linewidth = 2, label='BRS-BTERM', zorder=2, marker='o', s=150, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), BRS_decfeb_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-DECFEB', zorder=2, marker='^', s=25)
#plt.scatter(np.arange(1998, 2022), BRS_sepapr_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-SEPAPR', zorder=2, marker='*', s=25)
#plt.scatter(np.arange(1998, 2022), GES_sep_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-SEP', zorder=2, marker='o', s=50, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), GES_mar_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-MAR', zorder=2, marker='o', s=50, alpha=0.5)
plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_bterm_19982021 + intercept_BRS_bterm_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
plt.plot(np.arange(1998, 2022), BRS_bterm_regimeshift, c='k', alpha=1, linestyle='--')
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_apr_19982021 + intercept_BRS_apr_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_mar_19982021 + intercept_GES_mar_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_sep_19982021 + intercept_GES_sep_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.xticks(ticks= np.arange(1998, 2022), labels=['98', '99', '00', '01', '02', '03', '04', '05', '06', '07',
#                                                 '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
#                                                 '18', '19', '20', '21'], fontsize=10)
plt.yticks(fontsize=14)
plt.xticks(np.arange(1998,2023,4),fontsize=14)
#plt.ylim(0.2,1.6)
#plt.ylim(0.4,3.5)
plt.yticks(ticks=[20,25,30])
#plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
#plt.legend(fontsize=12, loc=0)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\brs_bterm_trendline.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% BRS BPeak
slope_BRS_bpeak_19982021, intercept_BRS_bpeak_19982021, _, _, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_bpeak)], brs_bpeak[~np.isnan(brs_bpeak)])
#BRS_bpeak_regimeshift = [26.8, 26.8,26.8,26.8,26.8,26.8,26.8,26.8,26.8,26.8,26.8,26.8,
#                         30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08, 30.08,]
plt.figure(figsize=(10, 2.5))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
#plt.scatter(np.arange(1998, 2022), brs_binit, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='BRS-BINIT', zorder=2, marker='o', s=50, alpha=0.5)
plt.scatter(np.arange(1998, 2022), brs_bpeak, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='BRS-BTERM', zorder=2, marker='o', s=50, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), BRS_decfeb_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-DECFEB', zorder=2, marker='^', s=25)
#plt.scatter(np.arange(1998, 2022), BRS_sepapr_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-SEPAPR', zorder=2, marker='*', s=25)
#plt.scatter(np.arange(1998, 2022), GES_sep_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-SEP', zorder=2, marker='o', s=50, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), GES_mar_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-MAR', zorder=2, marker='o', s=50, alpha=0.5)
plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_bpeak_19982021 + intercept_BRS_bpeak_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022), BRS_bterm_regimeshift, c='k', alpha=1, linestyle='--')
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_apr_19982021 + intercept_BRS_apr_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_mar_19982021 + intercept_GES_mar_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_sep_19982021 + intercept_GES_sep_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.xticks(ticks= np.arange(1998, 2022), labels=['98', '99', '00', '01', '02', '03', '04', '05', '06', '07',
#                                                 '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
#                                                 '18', '19', '20', '21'], fontsize=10)
plt.yticks(fontsize=14)
plt.xticks(np.arange(1998,2023,4),fontsize=14)
#plt.ylim(0.2,1.6)
#plt.ylim(0.4,3.5)
#plt.yticks(ticks=[20,25,30])
#plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
#plt.legend(fontsize=12, loc=0)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\brs_bpeak_trendline.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
#%% BRS BDur
slope_BRS_bdur_19982021, intercept_BRS_bdur_19982021, _, _, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_bdur)], brs_bdur[~np.isnan(brs_bdur)])
BRS_bdur_regimeshift = [13.08,13.08,13.08,13.08,13.08,13.08,13.08,13.08,13.08,13.08,13.08,13.08,
                        15.00,15.00,15.00,15.00,15.00,15.00,15.00,15.00,15.00,15.00,15.00,15.00]
plt.figure(figsize=(10, 2.5))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
#plt.scatter(np.arange(1998, 2022), brs_binit, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='BRS-BINIT', zorder=2, marker='o', s=50, alpha=0.5)
plt.scatter(np.arange(1998, 2022), brs_bdur, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='BRS-BTERM', zorder=2, marker='o', s=50, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), BRS_decfeb_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-DECFEB', zorder=2, marker='^', s=25)
#plt.scatter(np.arange(1998, 2022), BRS_sepapr_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-SEPAPR', zorder=2, marker='*', s=25)
#plt.scatter(np.arange(1998, 2022), GES_sep_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-SEP', zorder=2, marker='o', s=50, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), GES_mar_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-MAR', zorder=2, marker='o', s=50, alpha=0.5)
plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_bdur_19982021 + intercept_BRS_bdur_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
plt.plot(np.arange(1998, 2022), BRS_bdur_regimeshift, c='k', alpha=1, linestyle='--')
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_apr_19982021 + intercept_BRS_apr_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_mar_19982021 + intercept_GES_mar_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_sep_19982021 + intercept_GES_sep_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.xticks(ticks= np.arange(1998, 2022), labels=['98', '99', '00', '01', '02', '03', '04', '05', '06', '07',
#                                                 '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
#                                                 '18', '19', '20', '21'], fontsize=10)
plt.yticks(fontsize=14)
plt.xticks(np.arange(1998,2023,4),fontsize=14)
#plt.ylim(0.2,1.6)
#plt.ylim(0.4,3.5)
plt.yticks(ticks=[7,10,13, 16, 19, 22])
#plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
#plt.legend(fontsize=12, loc=0)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\brs_bdur_trendline.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()


#%% BRS BArea
slope_BRS_barea_19982021, intercept_BRS_barea_19982021, _, _, _ = stats.linregress(np.arange(1998, 2022)[~np.isnan(brs_barea)], brs_barea[~np.isnan(brs_barea)])
plt.figure(figsize=(10, 3.33))
#plt.plot(np.arange(1,13),weddell_cluster_mean19972021, color = [43/256, 131/256, 186/256, 1], linewidth = 4, label='1997-2021')
#plt.scatter(np.arange(1998, 2022), brs_binit, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='BRS-BINIT', zorder=2, marker='o', s=50, alpha=0.5)
plt.scatter(np.arange(1998, 2022), brs_barea, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='BRS-BAREA', zorder=2, marker='o', s=50, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), BRS_decfeb_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-DECFEB', zorder=2, marker='^', s=25)
#plt.scatter(np.arange(1998, 2022), BRS_sepapr_19982021, c = [106/256, 153/256, 78/256, 1], linewidth = 1, label='DRS-SEPAPR', zorder=2, marker='*', s=25)
#plt.scatter(np.arange(1998, 2022), GES_sep_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-SEP', zorder=2, marker='o', s=50, alpha=0.5)
#plt.scatter(np.arange(1998, 2022), GES_mar_19982021, c = [41/256, 51/256, 92/256, 1], linewidth = 1, label='GES-MAR', zorder=2, marker='o', s=50, alpha=0.5)
plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_barea_19982021 + intercept_BRS_barea_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_BRS_apr_19982021 + intercept_BRS_apr_19982021, color = [106/256, 153/256, 78/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_mar_19982021 + intercept_GES_mar_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.plot(np.arange(1998, 2022),np.arange(1998, 2022) * slope_GES_sep_19982021 + intercept_GES_sep_19982021, color = [41/256, 51/256, 92/256, 1], linewidth = 4, linestyle='-', alpha=1, zorder=1)
#plt.xticks(ticks= np.arange(1998, 2022), labels=['98', '99', '00', '01', '02', '03', '04', '05', '06', '07',
#                                                 '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
#                                                 '18', '19', '20', '21'], fontsize=10)
plt.yticks(fontsize=14)
plt.xticks(np.arange(1998,2023,4),fontsize=14)
#plt.ylim(0.2,1.6)
#plt.ylim(0.4,3.5)
plt.yticks(ticks=[0.8,1.2, 1.6])#, labels=['1.0', '2.0', '3.0'])
#plt.ylabel('Chl $a$ (mg m$^{-3}$)', fontsize=14)
#plt.legend(fontsize=12, loc=0)
plt.tight_layout()
graphs_dir = 'C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\brs_barea_trendline.png'
plt.savefig(graphs_dir,format = 'png', bbox_inches = 'tight', dpi = 300)
plt.close()
