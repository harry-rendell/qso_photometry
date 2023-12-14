# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: astro
#     language: python
#     name: python3
# ---

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.plotting.common import savefigs
from module.preprocessing import data_io, parse, binning
from module.preprocessing.binning import construct_T_edges
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

obj = 'qsos'
ID = 'uid' if obj == 'qsos' else 'uid_s'
band = 'r'

kwargs = {'dtypes': cfg.PREPROC.dtdm_dtypes,
          'ID':ID,
          'basepath': cfg.D_DIR + f'merged/{obj}/clean/',
          'nrows':20000000}
lcs = data_io.dispatch_reader(kwargs, multiproc=True, fnames=['lc_000000_005000.csv','lc_005001_010000.csv','lc_010001_015000.csv','lc_020001_025000.csv'])

kwargs = {'dtypes': cfg.PREPROC.dtdm_dtypes,
          'ID':ID,
          'basepath': cfg.D_DIR + f'merged/{obj}/clean/dtdm_{band}',
          'nrows':1000000}
dtdm = data_io.dispatch_reader(kwargs, multiproc=True, fnames=['dtdm_000000_005000.csv','dtdm_005001_010000.csv','dtdm_010001_015000.csv','dtdm_020001_025000.csv'])

dtdm

lcs_r_ztf = lcs[(lcs['band'] == 'r') & (lcs['sid']==11)]


lcs_r_ztf = lcs[(lcs['band'] == 'r') & (lcs['sid']==11)]


lcs_r = lcs[(lcs['band'] == 'r') ]


from module.preprocessing import lightcurve_statistics
a = lightcurve_statistics.groupby_apply_welch_stetson(lcs_r, kwargs={})

a['welch_stetson_j'].hist(bins=100)

a['welch_stetson_k'].hist(bins=100)

from module.plotting.common import plot_series

uids = lcs_r.index.unique()
plot_series(lcs_r, uids=uids[:20])

from tslearn.utils import to_time_series_dataset
X = to_time_series_dataset([lcs_r.loc[uid,'mag'] for uid in lcs_r.index.unique()[:100]])

# +
from tslearn.clustering import KernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X)
sz = X_train.shape[1]

gak_km = KernelKMeans(n_clusters=3,
                      kernel="gak",
                      kernel_params={"sigma": "auto"},
                      n_init=20,
                      verbose=True,
                      n_jobs=4,
                      random_state=1)
y_pred = gak_km.fit_predict(X_train)

plt.figure()
for yi in range(3):
    plt.subplot(3, 1, 1 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()
plt.show()
# -

# # features

features = pd.read_csv(cfg.D_DIR + f'computed/{obj}/features/features_{band}.csv', index_col=ID)

features.columns

# fs = ['f0_1','f0_4','f0_6','f0_8']
fs = [f'f{0}_{i}' for i in range(10)]
# fs = ['f0_1','f0_4','f0_6','f0_8']
features = parse.filter_data(features, bounds={f:features[f].quantile([0.05,0.95]).values for f in fs}, dropna=False) 

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(features[fs].corr(), annot=True, ax=ax)


g = sns.pairplot(features[fs], kind='hist')

savefigs(g, 'big_correlation_of_features', cfg.W_DIR + 'temp/', dpi=600)

fig, axes = plt.subplots(10,1, figsize=(10,50))
for i, ax in enumerate(axes.ravel()):
    features[f'f2_{i}'].hist(bins=50, ax=ax, range=(-0.5,2))

