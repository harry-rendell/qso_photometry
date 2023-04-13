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
import sys

uniqueid = 'uid_s'

# +
from multiprocessing import Pool
def reader(n_subarray):
    return pd.read_csv('/disk1/hrb/python/data/surveys/ztf/calibStars/lc_{}.csv'.format(n_subarray))# usecols = [0,1,2,3,4,5], index_col = 0, dtype = {'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, 'uid': np.uint32})

def read_ztf():
    pool = Pool(4)
    df_list = pool.map(reader, [0,1,2,3])
    return pd.concat(df_list, ignore_index=True)

df_ztf = read_ztf()
df_ztf['filtercode'] = df_ztf['filtercode'].str[-1]
# -

df_ztf

df_ztf = df_ztf.set_index('uid_s')
len(df_ztf.index.unique())

np.savetxt('uids.txt',df_ztf.index.unique(), fmt='%i')

# Need to remove >1 arcsec otherwise we get neighbouring objects
df_ztf_meta = pd.read_csv('/disk1/hrb/python/data/surveys/ztf/calibStars/calibStars_ztf.txt', delimiter = '\s+', skiprows = 62, usecols=[1,3,6,12], names=['dist_x',uniqueid,'oid','ngoodobsrel'], index_col=uniqueid)
mask = df_ztf_meta['oid'].duplicated(keep='first')
df_ztf_meta = df_ztf_meta[~mask]
# df_ztf_meta = pd.read_csv('/disk1/hrb/python/data/surveys/ztf/meta_data/ztf_meta.txt', delimiter = '\s+', skiprows = 62, usecols=[1,5,7,15], names=['dist_x',uniqueid,'oid','ngoodobsrel'], index_col=2)
df_ztf_meta = df_ztf_meta[(df_ztf_meta['dist_x'] < 1) & (df_ztf_meta['ngoodobsrel'] > 0)]

oids = pd.read_csv('/disk1/hrb/python/data/surveys/ztf/calibStars/calibStars_oids.txt', names=['oid'])

assert df_ztf_meta['oid'].is_unique, 'oid is not unique'

df_ztf_2 = df_ztf.merge(df_ztf_meta.reset_index(), on = 'oid', how='inner').set_index(uniqueid).sort_values(uniqueid)

df_ztf_2

for i, df_chunk in enumerate(np.array_split(df_ztf_2,4)):
    df_chunk.to_csv('/disk1/hrb/python/data/surveys/ztf/calibStars/lc_{}.csv'.format(i))

python/data/surveys/ztf/qsos/meta_data/ztf_meta.txt
