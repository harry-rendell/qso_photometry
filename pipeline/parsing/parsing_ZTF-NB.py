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
from multiprocessing import Pool
import os
import sys
sys.path.append('../')
from funcs.config import cfg
from funcs.preprocessing import parse, colour_transform, data_io

OBJ    = 'qsos'
ID     = 'uid'
BAND   = 'r'
wdir = cfg.USER.W_DIR

# # This should be combined with queries/ztf/qsos/parse_ztf_meta.ipynb, with notes on how data is collected and parsed before saving raw lcs

# +
# TODO:
# Currently, we have ztf_lc_query.py -> combine.sh -> SPLIT_SAVE (below) and then finally transforming those chunks and overwriting the previous result.
# Better to do ztf_lc_query.py to create {}_band/raw_combined/lc_{}.csv with to_csv(mode='a'), transform raw_combined and remove bad data here, then save to dr6/{}_band/lc_{}.csv
# -

# Separate data by band pass and save
SPLIT_SAVE = False
if SPLIT_SAVE:
    save_cols = ['oid','mjd','mag','magerr','filtercode','clrcoeff']
    # This cell takes the combined ztf raw lightcurve files and splits them by filterband
    ztf_oids = pd.read_csv(wdir+'queries/ztf/qsos/ztf_oids.csv')
    # columns are
    # oid,mjd,mag,magerr,filtercode,magzp,clrcoeff,clrcounc,airmass
    for i in range(4):
        df = pd.read_csv(wdir+'data/surveys/ztf/qsos/dr6/raw_lc_{}.csv'.format(i), usecols=save_cols, dtype=cfg.COLLECTION.ZTF.dtypes, nrows=None)
        df = ztf_oids.merge(df, on='oid').sort_values(['uid','mjd'])
        for band in 'gri':
            df.loc[df['filtercode']=='z'+band, ['uid']+save_cols].to_csv(wdir+'data/surveys/ztf/{}/dr6/{}_band/lc_{}.csv'.format(OBJ, band,i), index=False)
        del df

# # Read raw combined data
# ---

# +
# Memory Profiling:
# ----------------
# 12.46GB peak memory to read in data, which is roughly equal to final mem usage.
# # %load_ext memory_profiler
# # %memit data_io.dispatch_reader(kwargs)

# +
# keyword arguments to pass to our reading function
kwargs = {'dtypes': cfg.COLLECTION.ZTF.dtypes,
          'nrows': 10000,
          'basepath': wdir+'data/surveys/ztf/{}/dr6/{}_band/'.format(OBJ, BAND),
          'ID':'uid',
          'usecols': ['uid','mjd','mag','magerr','clrcoeff']}


raw_data = data_io.dispatch_reader(kwargs)
raw_data
# -

# # Transform to PanSTARRS
# ---

ztf_transformed = colour_transform.transform_ztf_to_ps(raw_data, OBJ, BAND)

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

chunks = parse.split_into_non_overlapping_chunks(ztf_transformed, 4)
data_io.dispatch_writer(chunks, kwargs)
