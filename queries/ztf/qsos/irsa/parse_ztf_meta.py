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
import matplotlib.pyplot as plt

meta_data = pd.read_csv('ztf_meta_data.csv').set_index('uid_01')
meta_data.index.name = 'uid'
meta_data.columns

# ---
# ### Plot pairing distance and magnitude distribution

fig, ax = plt.subplots(2,1, figsize=(25,14))
meta_data['dist_x'].hist(bins=200, ax=ax[0])
ax[0].set(yscale='log', ylabel='Number of sources at given distance', xlabel='distance (arcsec)')
meta_data['medianmag'].hist(bins=200, ax=ax[1], range=(15,25))
ax[1].set(ylabel='Number of sources at given magnitude', xlabel='median magnitude (mag)')

# ---
# ### Remove sources that do not satisfy following constraints
# 1. At least 2 observations
# 2. Within 1" of coordinates

mask = (meta_data['ngoodobsrel']>1) & (meta_data['dist_x']<1)
meta_data_restricted = meta_data[mask]
print('fraction of data removed with constraints above: {:.2f}%'.format((~mask).sum()/len(mask)*100))

# ---
# ### Plot pairing distance and magnitude distribution

fig, ax = plt.subplots(2,1, figsize=(25,14))
meta_data_restricted['dist_x'].hist(bins=200, ax=ax[0])
ax[0].set(yscale='log', ylabel='Number of sources at given distance', xlabel='distance (arcsec)')
meta_data_restricted['medianmag'].hist(bins=200, ax=ax[1], range=(15,25))
ax[1].set(ylabel='Number of sources at given magnitude', xlabel='median magnitude (mag)')

save_oids = False
if save_oids:
    meta_data_restricted['oid'].to_csv('../ztf_oids.csv')

meta_data_restricted['oid'].is_unique

meta_data_restricted[meta_data_restricted['oid'].duplicated(keep=False)].sort_values('oid')


