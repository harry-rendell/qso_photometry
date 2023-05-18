# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.preprocessing import parse, data_io

OBJ    = 'calibStars'
ID     = 'uid_s'
wdir = cfg.USER.W_DIR
ddir = cfg.USER.D_DIR
SAVE_OIDS = True

read_cols = [ID+'_01','dist_x','ra_01','dec_01','ra','dec','oid','filtercode','ngoodobsrel','medianmag','medianmagerr','minmag','maxmag']
meta_data = Table.read(ddir+'surveys/ztf/{}/ztf_meta_ipac.txt'.format(OBJ), format='ipac', include_names=read_cols).to_pandas().set_index(ID+'_01')
meta_data.index.name = ID

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

valid_uids = pd.read_csv(ddir+'catalogues/{}/valid_uids_superset.csv'.format(OBJ), usecols=[ID], index_col=ID, comment='#')
meta_data_restricted = parse.filter_data(meta_data, bounds = {'ngoodobsrel':(2,np.inf), 'dist_x':(0,1), 'medianmag':(14,21.2)}, valid_uids=valid_uids)

# ---
# ### Plot pairing distance and magnitude distribution

fig, ax = plt.subplots(2,1, figsize=(25,14))
meta_data_restricted['dist_x'].hist(bins=200, ax=ax[0])
ax[0].set(yscale='log', ylabel='Number of sources at given distance', xlabel='distance (arcsec)')
meta_data_restricted['medianmag'].hist(bins=200, ax=ax[1], range=(15,25))
ax[1].set(ylabel='Number of sources at given magnitude', xlabel='median magnitude (mag)')

if not meta_data_restricted['oid'].is_unique:
    # If we have duplicate oids, display them here
    display(meta_data_restricted[meta_data_restricted['oid'].duplicated(keep=False).values].sort_values('oid'))
    raise Warning('duplicate oids!')

if SAVE_OIDS:
    meta_data_restricted['oid'].to_csv(ddir+'surveys/ztf/{}/ztf_oids.csv'.format(OBJ))

# ---
# # Read raw combined data
# ---
# The following cells should be run after scripts/ztf_lc_query.py has been used with ztf_oids above to fetch lightcurves.

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.preprocessing import parse, data_io, colour_transform

OBJ    = 'calibStars'
ID     = 'uid_s'
BAND   = 'r'
wdir = cfg.USER.W_DIR

# +
# may not be needed as merge now already happens during ztf_lc_query
# ztf_oids = pd.read_csv(wdir+'pipeline/queries/ztf/qsos/ztf_oids.csv')

for band in 'gri':
    # keyword arguments to pass to our reading function
    kwargs = {'dtypes': cfg.COLLECTION.ZTF.dtypes,
              'nrows': None,
              'basepath': cfg.USER.D_DIR + 'surveys/ztf/{}/dr6/{}_band/'.format(OBJ, band),
              'ID':ID,
              'usecols': [ID,'oid','mjd','mag','magerr','limitmag','clrcoeff']}
    raw_data = []
    n_files = len([f for f in os.listdir(kwargs['basepath']) if f.startswith('lc_')])
    for i in range(n_files):
        df = data_io.dispatch_reader(kwargs, multiproc=False, i=i)
        # Line below is only needed if data above does not include uid.
#         df = df.merge(ztf_oids, on='oid', how='inner').set_index(ID)
        raw_data.append(df)
        
    raw_data = pd.concat(raw_data).sort_values([ID,'mjd'])
    raw_data = colour_transform.transform_ztf_to_ps(raw_data, OBJ, band)
    
    print('No. NA entries:',raw_data.isna().sum())
    
    # Add comment to start of csv file
    comment = ( "# CSV of photometry transformed to PS with no other preprocessing or cleaning.\n"
                "# mag      : transformed photometry in PanSTARRS photometric system\n"
                "# mag_orig : original photometry in native ZTF photometric system")

    # keyword arguments to pass to our writing function
    kwargs = {'comment':comment,
              'basepath':cfg.USER.D_DIR + 'surveys/ztf/{}/unclean/{}_band/'.format(OBJ, band)}

    chunks = parse.split_into_non_overlapping_chunks(raw_data, 4)
    data_io.dispatch_writer(chunks, kwargs)
# -
raw_data.isna().sum()

# +
# # Number of observations whose recorded mag is brighter than limitmag, for objects with mag < 20.6 (our current limiting magnitude on r)
# test = raw_data[raw_data['mag']<20.6]
# (test['mag']>test['limitmag']).sum()/len(test)*100
