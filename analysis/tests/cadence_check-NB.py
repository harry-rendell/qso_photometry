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
from module.classes.analysis import *
# %matplotlib inline

band = 'r'


def reader(n_subarray):
    return pd.read_csv('../data/merged/{}/{}_band/lc_{}.csv'.format(obj, band, n_subarray), nrows=50000, index_col = ID, dtype = {'catalogue': np.uint8, 'mag_ps': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})


obj = 'qsos'
ID  = 'uid'
dr = analysis(ID)
dr.read_in(reader, redshift=False)


def cadence(mjd):
    diff = mjd[1:] - mjd[:-1]
    return min(diff)


min_time_sep = dr.df['mjd'].groupby('uid').apply(cadence)

fig, ax = plt.subplots(1,1, figsize=(20,10))
min_time_sep.hist(bins=200, ax=ax)
ax.set(yscale='log')

min_time_sep


