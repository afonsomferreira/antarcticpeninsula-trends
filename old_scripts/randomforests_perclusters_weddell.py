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
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
def dropcol_importances(rf, X_train, y_train):
    rf_ = clone(rf)
    rf_.random_state = 23
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 23
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I
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
#%% Load Variables
# Load clusters
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\oc4so_chl\\')
fh = np.load('antarcticpeninsula_cluster.npz',allow_pickle = True)
clusters = fh['clusters']
# CHL
fh = np.load('chloc4so_19972021_10km.npz', allow_pickle=True)
lat_chl = fh['lat'][100:]
lon_chl = fh['lon'][30:250]
chl = fh['chl'][100:, 30:250, :]
time_date_chl = fh['time_date']
# Correct values
chl[chl > 50] = 50
# SST
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sst-seaice\\ostia')
### Load data 1998-2020
fh = np.load('sst_19972021_10km.npz', allow_pickle=True)
lat_sst = fh['lat'][100:]
lon_sst = fh['lon'][30:250]
sst = fh['sst'][100:, 30:250, :]
time_date_sst = fh['time_date']
# SEA ICE
### Load data 1998-2020
fh = np.load('seaice_19972021_10km.npz', allow_pickle=True)
lat_seaice = fh['lat'][100:]
lon_seaice = fh['lon'][30:250]
seaice = fh['seaice'][100:, 30:250, :]
seaice = seaice*100
time_date_seaice = fh['time_date']
#PAR
### Load data 1998-2020
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\par\\')
fh = np.load('par_19972022_10km.npz', allow_pickle=True)
lat_par = fh['lat'][100:]
lon_par = fh['lon'][30:250]
par = fh['par'][100:, 30:250, :]
time_date_par = fh['time_date']
# MEI
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\elnino\\')
mei_pd = pd.read_csv('meiv2.csv', sep=';')
mei_monthly_19972021 = mei_pd['MEI2'][8:-8].values
# SAM
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\resources\\sam\\')
sam_pd = pd.read_csv('sam.csv', sep=';')
sam_monthly_19972021 = sam_pd['SAM'][8:-8].values
#%% Create dataframe for Weddell
## CHL
weddell_cluster = chl[clusters == 1,:]
weddell_cluster = np.nanmean(weddell_cluster,0)
weddell_cluster_df_chl = pd.Series(data=weddell_cluster, index=time_date_chl)
weddell_cluster_df_monthly_chl = weddell_cluster_df_chl.resample('M').mean()
weddell_monthly_19972021_chl = weddell_cluster_df_monthly_chl.values[:-8]
weddell_monthly_19972021_index = weddell_cluster_df_monthly_chl.index[:-8]
## SST
weddell_cluster = sst[clusters == 1,:]
weddell_cluster = np.nanmean(weddell_cluster,0)
weddell_cluster_df_sst = pd.Series(data=weddell_cluster, index=time_date_sst)
weddell_cluster_df_monthly_sst = weddell_cluster_df_sst.resample('M').mean()
weddell_monthly_19972021_sst = weddell_cluster_df_monthly_sst.values[8:-8]
## Sea Ice
weddell_cluster = seaice[clusters == 1,:]
weddell_cluster = np.nanmean(weddell_cluster,0)
weddell_cluster_df_seaice = pd.Series(data=weddell_cluster, index=time_date_seaice)
weddell_cluster_df_monthly_seaice = weddell_cluster_df_seaice.resample('M').mean()
weddell_monthly_19972021_seaice = weddell_cluster_df_monthly_seaice.values[8:-8]
## PAR
weddell_cluster = par[clusters == 1,:]
weddell_cluster = np.nanmean(weddell_cluster,0)
weddell_cluster_df_par = pd.Series(data=weddell_cluster, index=time_date_par)
weddell_cluster_df_monthly_par = weddell_cluster_df_par.resample('M').mean()
weddell_monthly_19972021_par = weddell_cluster_df_monthly_par.values[:-17]
## Join all in the same dataframe
dataframe_weddell = pd.DataFrame(data=weddell_monthly_19972021_chl, index= weddell_monthly_19972021_index, columns=['Chla'])
dataframe_weddell['SST'] = weddell_monthly_19972021_sst
#dataframe_weddell['SEAICE'] = weddell_monthly_19972021_seaice
dataframe_weddell['PAR'] = weddell_monthly_19972021_par
dataframe_weddell['MEI'] = mei_monthly_19972021
dataframe_weddell['SAM'] = sam_monthly_19972021
#%%
## Plot Correlation Matrix
corr = dataframe_weddell.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, mask = mask, cmap=cmocean.cm.balance, vmin=-1, vmax=1)
#plt.tight_layout()
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\correlation_matrices\\')
graphs_dir = 'weddell.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Run model
## Check co-correlation between variables
correlated_features = set()
## Keep only non correlated variables
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) >= 0.7:
            colname = corr.columns[i]
            correlated_features.add(colname)
#dataframe_all_original = dataframe_all
dataframe_all = dataframe_weddell.dropna()
#f, ax = plt.subplots(figsize=(9, 6))
#sns.heatmap(correlation_matrix, annot=True, linewidths=.5, ax=ax,
#            vmin=-1,vmax=1,cmap=sns.diverging_palette(220, 20, n=256))
#b, t = plt.ylim() # discover the values for bottom and top
#b += 0.5 # Add 0.5 to the bottom
#t -= 0.5 # Subtract 0.5 from the top
#plt.ylim(b, t) # update the ylim(bottom, top) values
#plt.tight_layout()
### REMOVE RESPONSE VARIABLE ###
features = dataframe_all
X = features.drop(['Chla'],axis=1)
features_list = features.columns
y = dataframe_all['Chla']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500, random_state=0)
regressor.fit(X, y)
y_pred = regressor.predict(X)
from sklearn import metrics
# Metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))
print('Score Train: ', regressor.score(X_train, y_train))
print('Score Full: ', regressor.score(X, y))
#print('Score Test: ', regressor.score(X_test, y_test))
#%%
# Variable importance
from sklearn.inspection import permutation_importance
scoring = ['r2']
r_multi = permutation_importance(regressor, X, y,
                           n_repeats=100,
                           random_state=23, scoring=scoring)
for metric in r_multi:
    print(f"{metric}")
    r = r_multi[metric]
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X.columns[i]:<8}  " 
                  f"{r.importances_mean[i]:.3f}  " 
                  f" +/- {r.importances_std[i]:.3f}")
os.chdir('C:\\Users\\afons\\Documents\\artigos\\antarcticpeninsula-trends-2021\\analysis\\randomforests\\weddell\\')
currentmodel_permut = pd.read_csv('permutation_weddell_v2_withseaice.csv', sep=";")
# Load sardine data
variables_names = currentmodel_permut['Variable'].values
variables_importances = currentmodel_permut['Importance'].values
variables_stds = currentmodel_permut['Std'].values
#%%
fig, axs = plt.subplots(1, 2, figsize=(11,4))
## Plot the actual values
#axs[1].scatter(dataframe_all_original.index, dataframe_all_original['Recruitment'].values, marker='^', c='#000080', label = 'Observed recruitment')
axs[1].plot(dataframe_all.index, dataframe_all['Chla'].values, marker='^', c='#36454F', label = 'Observed Chla', markersize=2, alpha=0.5)
# Plot the predicted values
axs[1].scatter(dataframe_all.index, regressor.predict(X), marker='o', label = 'Predicted Chla', c=[43/256, 131/256, 186/256, 1], s=20, edgecolor='k')
#axs[1].set_xticks(ticks = np.arange(1999, 2021),
#           labels = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08',
#                     '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
#                     '19', '20'], fontsize=9)
axs[1].legend()
axs[1].set_xlabel('Years', fontsize=14)
axs[1].set_ylabel('Chla', fontsize=14)
#axs[1].set_title('Predictions', fontsize=14)
#from partial_dependence import PartialDependenceExplainer
axs[0].barh(variables_names[::-1], variables_importances[::-1], xerr=variables_stds[::-1], facecolor=[43/256, 131/256, 186/256, 1], edgecolor='k')
#axs[0].axhline(y=10.5, c='#36454F', alpha=0.5)
#axs[0].axhspan(10.4, 15, facecolor='#36454F', alpha=0.2)
#axs[0].set_ylim(-1, 15)
axs[0].text(0.6, 0, f"R2={regressor.score(X, y):.2f}", fontsize=14)
axs[0].text(0.6, -.25, f"MAE={metrics.mean_absolute_error(y, y_pred):.2f}", fontsize=14)
#axs[0].axvline(x=np.mean(predictor_importances['Importance']), c='#A50021', linestyle='--')
axs[0].set_xlabel('Variable Importance', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'weddell_randomforestresults_V2withseaice.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()


#%% Plot Figure 7
from sklearn.inspection import plot_partial_dependence
#fig = plt.figure(figsize=(9,9))
plot_partial_dependence(regressor,X, ['PAR'], kind='average', 
                        line_kw={"color": "k", "linewidth": 4}
                        )
#plt.yticks(ticks=[np.log10(800), np.log10(1000), np.log10(1200), np.log10(1400), np.log10(1600), np.log10(1800), np.log10(2000)],
#           labels=[800, 1000, 1200, 1400, 1600, 1800, 2000])
#plt.ylim(2.9,3.2)
#plt.xlabel(fontsize=14)
#plt.ylabel(fontsize=14)
#axs[0].set_xlabel('Fav. Window (n)', fontsize=14)
plt.tight_layout()
#os.chdir('C:\\Users\\afons\\Documents\\artigos\\sardine-2021\\analyses\\Random Forests\\NW')
graphs_dir = 'weddell_randomforest_PAR.png'
plt.savefig(graphs_dir, format='png', bbox_inches='tight', dpi=300)
plt.close()
#%% Testes
plt.scatter(weddell_monthly_19972021_chl, weddell_monthly_19972021_sst)
np.corrcoef(weddell_monthly_19972021_chl, weddell_monthly_19972021_sst)
stats.linregress(weddell_monthly_19972021_chl[~np.isnan(weddell_monthly_19972021_chl)], weddell_monthly_19972021_sst[~np.isnan(weddell_monthly_19972021_chl)])
