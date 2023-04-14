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

# +
# Code is working as of Jun 22
# -

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool
import sys
sys.path.append('../')

from module.analysis.obj_survey import obj_survey
from module.preprocessing.parse import intersection, filter_data
from module.analysis.plotting import plot_magerr_hist, plot_mag_dist

# # Merging lcs from unaveraged data

# obj = 'calibStars'
# ID  = 'uid_s'
obj = 'qsos'
ID  = 'uid'

nrows = None
bounds={'mag_ps':(15,25),'magerr':(0,2)}

colors = pd.read_csv('../../data/computed/{}/colors_sdss.csv'.format(obj), index_col=0)

sdss = obj_survey('sdss', obj, ID)
sdss.read_in_raw(nrows, save=False)
sdss.transform_to_ps(colors=colors, color='g-r', system='tonry')
sdss.df = filter_data(sdss.df, bounds)#, bands = 'ugriz') # apply filter after transformation so any bad transformations are removed
# sdss.residual_raw({'g':0.0148, 'r':0.0049, 'i':0.0198, 'z':0.042}) # apply residuals

# +
# def apply_max_z(group):
#     mag = group['mag_ps'].values
#     z = (mag-mag.mean())/mag.std()
#     max_z = max(abs(z))
#     return max_z
# -

ps = obj_survey('ps', obj, ID)
ps.read_in_raw(nrows)
ps.df['mag_ps'] = ps.df['mag']
ps.df = filter_data(ps.df, bounds)

ztf = obj_survey('ztf', obj, ID)
# ztf.pivot()
ztf.read_in_raw(nrows)
ztf.transform_to_ps(colors=colors)
ztf.df = filter_data(ztf.df, bounds) # apply filter after transformation so any bad transformations are removed
# ztf.residual_raw({'g':0.0074, 'r':-0.0099, 'i':0}) # apply residuals

# # group together observations with ∆t < 1 day?

# ## Merge and save lightcurves

sdss.df['catalogue'] = 5
ps  .df['catalogue'] = 7
ztf .df['catalogue'] = 11
cols = ['mjd','mag','mag_ps','magerr','catalogue']
master = pd.concat([sdss.df[cols].reset_index(), ps.df[cols].reset_index(), ztf.df[cols].reset_index()], axis = 0, ignore_index = True).astype({sdss.ID:'int'}).set_index(ID)

master

# Remove objects with a single observation
# faster to do .duplicated() ?
value_counts = master.index.value_counts()
uids = value_counts[value_counts==1].index
master = master[~master.index.isin(uids)]

# Drop nan mag entries
master = master[~master['mag_ps'].isna()]

master = master.sort_values([ID,'mjd'])


# +
# If we are getting 'cannot allocate memory' then do one at a time.
# master = master[master['filtercode']=='r']

# +
# If we want split up bands
def save(args):
    i, chunk = args
    chunk.to_csv('/disk1/hrb/python/data/merged/{}/{}_band/unclean/lc_{}.csv'.format(obj, band, i))

for band in 'giz':
    chunks = np.array_split(master[master['filtercode']==band].drop(columns='filtercode'),4)
    if __name__ == '__main__':
        pool = Pool(4)
        pool.map(save, enumerate(chunks))

# +
# putting all bands in one, and splitting up by uid
# if __name__ == '__main__':
#     pool = Pool(4)
#     chunks = enumerate([master.loc[:200000], master.loc[200001:320000], master.loc[320001:420000], master.loc[420001:]])
#     pool.map(save, chunks)
