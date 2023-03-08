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
import pandas as pd 
pd.options.mode.chained_assignment = None
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool
# from profilestats import profile
from scipy.stats import binned_statistic
from funcs.analysis.analysis import *
# %matplotlib inline

def reader(n_subarray):
    return pd.read_csv('../data/merged/{}/r_band/with_ssa/lc_{}.csv'.format(obj,n_subarray), comment='#', index_col = ID, dtype = {'catalogue': np.uint8, 'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})


# -

obj = 'qsos'
ID  = 'uid'
band = 'r'
redshift_bool = True

dr = analysis(ID, obj)
dr.read_in(reader, redshift=redshift_bool)
dr.band = band
dr.group(keys = ['uid'],read_in=True, redshift=redshift_bool, survey = 'all')

uids = dr.df_grouped['mag_std'].sort_values(ascending=False).head(30).index

dr.plot_series(uids)

dr.plot_series([uids[1]], xlim=[56100, 58750])
