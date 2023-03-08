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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# obj = 'qsos'
# ID  = 'uid'
obj = 'calibStars'
ID  = 'uid_s'
band = 'r'


class analyse_group():
    def __init__(self, obj, band, ID):
        self.obj  = obj
        self.band = band
        self.ID   = ID
        
    def read(self):
        self.grouped_all  = pd.read_csv('../data/merged/{}/{}_band/grouped_stats_{}_all.csv' .format(self.obj, self.band, self.band), index_col = self.ID)
        self.grouped_sdss = pd.read_csv('../data/merged/{}/{}_band/grouped_stats_{}_sdss.csv'.format(self.obj, self.band, self.band), index_col = self.ID)
        self.grouped_ps   = pd.read_csv('../data/merged/{}/{}_band/grouped_stats_{}_ps.csv'  .format(self.obj, self.band, self.band), index_col = self.ID)
        self.grouped_ztf  = pd.read_csv('../data/merged/{}/{}_band/grouped_stats_{}_ztf.csv' .format(self.obj, self.band, self.band), index_col = self.ID)
        
    def intersection():
        pass


dr = analyse_group(obj, band, ID)
dr.read()

# # Stars

dr.grouped_sdss['magerr_mean'].mean()

dr.grouped_sdss['magerr_mean'].std()

dr.grouped_sdss['mag_mean'].mean()

dr.grouped_sdss['mag_mean'].std()

# # Qsos

dr.grouped_sdss['magerr_mean'].mean()

dr.grouped_sdss['magerr_mean'].std()

dr.grouped_sdss['mag_mean'].mean()

dr.grouped_sdss['mag_mean'].std()

dr.grouped_sdss.hist('mean_g', bins=200)

dr.grouped_sdss['count'].mean()

dr.grouped_ps['count_r'].mean()

dr.grouped_ps['count_r'].max()

dr.grouped_ztf

ztf = dr.grouped_ztf[dr.grouped_ztf['filtercode']=='r']

dr.grouped_ztf[dr.grouped_ztf['filtercode']=='r']['count'].max()

ztf = ztf[~ztf['mean'].isna()]

fig, ax = plt.subplots(1,1, figsize=(20,8))
ax.hist(sdss['mean_r'], bins=150, alpha=0.5)
ax.hist(ps['mean_r'], bins=150, alpha=0.5)
ax.hist(ztf['mean_ps'], bins=150, alpha=0.5)

ztf

df = pd.read_csv('../data/merged/qsos/meta_data/grouped_stats.csv')

df = df[df['mag_mean'] > 15]



fig, ax = plt.subplots(1,1, figsize=(20,10))
df['mag_mean'].hist(bins=150, ax = ax)
plt.xlabel('r band magnitude', fontsize=25)
plt.xticks(fontsize=25)
plt.title('Quasar magnitude distribution', fontsize=25)

(df['mag_mean']<21).sum()


