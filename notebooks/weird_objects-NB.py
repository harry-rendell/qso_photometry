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
from multiprocessing import Pool
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
from module.config import cfg
from module.preprocessing import data_io, parse, lightcurve_statistics, pairwise
# from module.classes.analysis import analysis
from module.plotting.common import plot_series

# +
OBJ    = 'qsos'
ID     = 'uid'
BAND   = 'r'
wdir   = cfg.W_DIR
nrows  = None
skiprows = None if nrows == None else nrows * 0
SURVEY = 'sdss'

kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
          'nrows': nrows,
          'skiprows': skiprows,
          'basepath': cfg.D_DIR + 'surveys/{}/{}/clean/{}_band'.format(SURVEY, OBJ, BAND), # we should make this path more general so it is consistent between surveys
          'ID':ID}

df = data_io.dispatch_reader(kwargs, multiproc=False, i=2)
# -

redshifts = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr14q/dr14q_redshift.csv').set_index('uid')

df=df.join(redshifts, on='uid')

df['mjd_rf'] = df['mjd']/(1+df['z'])

df

# +
OBJ    = 'calibStars'
ID     = 'uid_s'
BAND   = 'r'
wdir   = cfg.W_DIR
nrows  = 100
skiprows = None if nrows == None else nrows * 0
SURVEY = 'ztf'

kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
          'nrows': nrows,
          'skiprows': skiprows,
          'basepath': cfg.D_DIR + 'surveys/{}/{}/unclean/{}_band'.format(SURVEY, OBJ, BAND), # we should make this path more general so it is consistent between surveys
          'ID':ID}

df = data_io.dispatch_reader(kwargs, multiproc=True)
# -

df['mag'].hist(bins=200)

df['mag'].hist(bins=200)



axes

axes = plot_series(df, uids)

uids = [291224]
axes = plot_series(df, uids)
axes = plot_series(df, uids)
axes.set_xlim([58580.26953125-20,58580.26953125+20])

uids = 121363
axes = plot_series(df, uids)
axes = plot_series(df, uids)
axes.set_xlim([58890,58900])

# ---
# ### Object 1
# 134 ztf observations in one night.
#

lower = 58298.353
upper = 58298.42
dr.plot_series([471006], xlim=[lower,upper])
dr.plot_series([471006])
print('Time window for observations below: {:.2f} hours'.format((upper-lower)*24))

dr.sdss_quick_look([471006])

# position of object
ra, dec = dr.coords.loc[471006].values
print("https://skyserver.sdss.org/dr15/en/tools/chart/navi.aspx?ra={}&dec={}".format(ra, dec))

# ---
# ### Object 2

coords = dr.coords.loc[[1,2]].values

for ra, dec in coords:
    print(ra, dec)

dr.coords.loc[[1,2]]


