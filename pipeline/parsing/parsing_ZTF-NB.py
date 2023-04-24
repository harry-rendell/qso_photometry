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
import numpy as np
import os
import sys  
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.preprocessing import parse, data_io, colour_transform

OBJ    = 'qsos'
ID     = 'uid'
BAND   = 'i'
wdir = cfg.USER.W_DIR
SAVE_OIDS = False

meta_data = pd.read_csv(wdir+'python/pipeline/queries/ztf/{}/irsa/ztf_meta_data.csv'.format(OBJ)).set_index('uid_01')
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
# 3. Fainter than 21.2 (limiting mags in gri are 20.8, 20.6 and 19.9 respectively but allow some buffer)

mask = (meta_data['ngoodobsrel']>1) & (meta_data['dist_x']<1) & (meta_data['medianmag']<21.2)
meta_data_restricted = meta_data[mask]
print('fraction of data removed with constraints above: {:.2f}%'.format((~mask).sum()/len(mask)*100))

# ---
# ### Plot pairing distance and magnitude distribution

fig, ax = plt.subplots(2,1, figsize=(25,14))
meta_data_restricted['dist_x'].hist(bins=200, ax=ax[0])
ax[0].set(yscale='log', ylabel='Number of sources at given distance', xlabel='distance (arcsec)')
meta_data_restricted['medianmag'].hist(bins=200, ax=ax[1], range=(15,25))
ax[1].set(ylabel='Number of sources at given magnitude', xlabel='median magnitude (mag)')

if not meta_data_restricted['oid'].is_unique:
    # If we have duplicate oids, display them here
    raise Warning('duplicate oids!')
    display(meta_data_restricted[meta_data_restricted['oid'].duplicated(keep=False)].sort_values('oid'))

if SAVE_OIDS:
    meta_data_restricted['oid'].to_csv(wdir+'python/pipeline/queries/ztf/{}/ztf_oids.csv'.format(OBJ))

# ---
# # Read raw combined data
# ---
# The following cells should be run after scripts/ztf_lc_query.py has been used with ztf_oids above to fetch lightcurves.

# +
# keyword arguments to pass to our reading function
kwargs = {'dtypes': cfg.COLLECTION.ZTF.dtypes,
          'nrows': None,
          'basepath': wdir+'data/surveys/ztf/{}/dr6/{}_band/'.format(OBJ, BAND),
          'usecols': ['oid','mjd','mag','magerr','limitmag','clrcoeff']}

raw_data = data_io.dispatch_reader(kwargs)
raw_data
# -

ztf_oids = pd.read_csv(wdir+'python/pipeline/queries/ztf/qsos/ztf_oids.csv')
raw_data = raw_data.merge(ztf_oids, on='oid', how='inner').set_index(ID).sort_values(['uid','mjd'])

# # Transform to PanSTARRS
# ---

raw_data = colour_transform.transform_ztf_to_ps(raw_data, OBJ, BAND)

# should not have any NA entries
raw_data.isna().sum()

# # Save data
# ---

# +
# Add comment to start of csv file
comment = """# CSV of photometry transformed to PS with no other preprocessing or cleaning.
# mag      : transformed photometry in PanSTARRS photometric system
# mag_orig : original photometry in native ZTF photometric system.\n"""

# keyword arguments to pass to our writing function
kwargs = {'comment':comment,
          'basepath':cfg.USER.W_DIR + 'data/surveys/ztf/{}/unclean/{}_band/'.format(OBJ, BAND)}

chunks = parse.split_into_non_overlapping_chunks(raw_data, 4)
data_io.dispatch_writer(chunks, kwargs)
