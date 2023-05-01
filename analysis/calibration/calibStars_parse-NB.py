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

# # Notebook for matching magnitude distributions of stars/qso
# Since we have more stars than we need, choose the ones that best match the mag distribution of the quasars to make a sensible control sample for the qsos

import pandas as pd
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

t = Table.read(cfg.USER.D_DIR + 'surveys/ztf/calibStars/calibStars_ztf.txt', format='ipac', include_names = ['uid_s_01','dist_x', 'oid', 'ngoodobsrel', 'maxmag', 'minmag', 'medianmag', 'filtercode'])

df=t[~t.mask['oid']].to_pandas(index='uid_s_01')
df=df[df['ngoodobsrel']>0]
df['filtercode'] = df['filtercode'].str[-1]

df.loc[:,'mag_ptp']=df['maxmag']-df['minmag']

uids = pd.read_csv('../data/catalogues/calibStars/calibStars_uids_distn_matched.csv',index_col=0,names=['uid'])

df = df.loc[uids[uids.index.isin(df.index.unique())].index] #remove uids that show up in s82 sdss stars that do not appear in ztf

fig, ax = plt.subplots(1,2,figsize=(15,5))
df['mag_ptp'].hist(range = (0,1.75), bins=300, ax=ax[0])
df['mag_ptp'].hist(range = (0,1.75), bins=300, cumulative=True, density=True, ax=ax[1])

scaling_dict = {'g':24000,'r':35500,'i':35500}
fig, ax = plt.subplots(1,3,figsize=(15,5))
for i, band in enumerate('gri'):
    scaling = scaling_dict[band]
    subdf = df[df['filtercode']==band]
    mask =  ~subdf.index.duplicated(keep='first')
    subdf.loc[mask,'medianmag'].hist(range = (15,24), bins=300, ax=ax[i])
#     n, bins = np.histogram(sdss[band + '_mmu'], bins=200,range=(15,24))
    n_qso, bins_qso = pd.read_csv('computed/qso_mag_dist_nbins_{}.csv'.format(band)).values.T
    n_qso = n_qso[:-1]
    ax[i].plot(0.5*(bins_qso[:-1]+bins_qso[1:]),n_qso*scaling, color = 'b', label = 'qsos')

np.savetxt(cfg.USER.D_DIR + 'surveys/ztf/calibStars/calibStars_oids.txt',df['oid'].values, fmt='%i')

df[df['mag_ptp']<0.2]['oid'].to_csv(cfg.USER.D_DIR + 'surveys/ztf/calibStars/calibStars_oids_uids.csv')

np.savetxt(cfg.USER.D_DIR + 'surveys/ztf/calibStars/calibStars_oids.txt',df[df['mag_ptp']<0.25]['oid'].values, fmt='%i')

# Remember - ztf one to one match does NOT work, ztf oid is specific to a single band. We need to do many to one match within 1".
# Run code below to find oids that we are missing.
# Note that when we do a one to many match we may pick up unwanted neighbouring objects. To remove these, sort by dist_x and remove oid duplicates (keep='first').

t = Table.read(cfg.USER.D_DIR + 'surveys/ztf/calibStars/calibStars_ztf.txt', format='ipac', include_names = ['uid_s_01','dist_x', 'oid', 'ngoodobsrel', 'maxmag', 'minmag'])

df=t.to_pandas(index='uid_s_01')
df=df[df['ngoodobsrel']>0]
df.loc[:,'mag_ptp']=df['maxmag']-df['minmag']
fig, ax = plt.subplots(1,2,figsize=(15,5))
df['mag_ptp'].hist(range = (0,1.75), bins=300, ax=ax[0])
df['mag_ptp'].hist(range = (0,1.75), bins=300, cumulative=True, density=True, ax=ax[1])

df = df.sort_values('dist_x')
df = df[~df['oid'].duplicated(keep='first')]

assert df['oid'].is_unique

df[df['mag_ptp']<0.2]['oid'].to_csv(cfg.USER.D_DIR + 'surveys/ztf/calibStars/calibStars_oids_uids.csv')

oids = df[df['mag_ptp']<0.2]['oid'].values

oids_1to1 = np.loadtxt(cfg.USER.D_DIR + 'surveys/ztf/calibStars/calibStars_oids_1to1.txt',dtype=np.uint64)

oids_diff = np.setdiff1d(oids, oids_1to1)

np.savetxt(cfg.USER.D_DIR + 'surveys/ztf/calibStars/calibStars_oids.txt',oids,fmt='%i')
