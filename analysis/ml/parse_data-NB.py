# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
sys.path.append('../')
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.stats import binned_statistic
from funcs.analysis.analysis import analysis

obj = 'qsos'
ID  = 'uid'
band = 'r'
redshift_bool = True


def reader(n_subarray):
    return pd.read_csv('../../data/surveys/ztf/{}/lc_{}.csv'.format(obj, n_subarray), usecols = [0,1,2,3,4,5], index_col = 0, dtype = {'oid': np.uint64, 
                                                                                                                                           'mag': np.float32, 
                                                                                                                                           'magerr': np.float32, 
                                                                                                                                           'mjd': np.float64, 
                                                                                                                                           'uid': np.uint32})


dr = analysis(ID, obj)

dr.read_in(reader=reader);
dr.band = band

for band in 'gri':
    for 
        dr.df[dr.df['filtercode']==band].drop(columns='filtercode').to_csv('../../data/surveys/ztf/{}/lc_{}.csv'.format(obj,n))

dr.group(read_in=True)

dr.df_grouped

gb = dr.df.head(1000).groupby('uid')

mag = [group.values for _, group in gb['mag']]
magerr = [group.values for _, group in gb['magerr']]
mjd = [group.values for _, group in gb['mjd']]

# ## Cesium-ml

from cesium import datasets
from cesium import featurize

asas = datasets.fetch_asas_training('/disk1/hrb/python/analysis/ml/datasets/asas_training')

asas["classes"].unique()

features_to_use = ["amplitude",
                   "percent_beyond_1_std",
                   "maximum",
                   "max_slope",
                   "median",
                   "median_absolute_deviation",
                   "percent_close_to_median",
                   "minimum",
                   "skew",
                   "std",
                   "weighted_average"]
fset_cesium = featurize.featurize_time_series(times=asas["times"],
                                              values=asas["measurements"],
                                              errors=None,
                                              features_to_use=features_to_use)

fset_cesium

# +
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

train, test = train_test_split(np.arange(len(asas["classes"])), random_state=0)

model_cesium = RandomForestClassifier(n_estimators=128, max_features="auto",
                                      random_state=0)
model_cesium.fit(fset_cesium.iloc[train], asas["classes"][train])

# +
from sklearn.metrics import accuracy_score
preds_cesium = model_cesium.predict(fset_cesium)

print("Built-in cesium features: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_cesium[train], asas["classes"][train]),
          accuracy_score(preds_cesium[test], asas["classes"][test])))
# -



# +
# dr2.df_grouped.to_csv('../../data/surveys/ztf/ztfdr2_gb_uid.csv')
# -

bin_centres

plt.scatter(bin_centres, (tss/counts)**0.5)

plt.scatter(bin_centres,counts)

dr2.plot_series(uids = dr2.df_grouped.sort_values('mag_count',ascending=False).head(400).index,sharex=True)
