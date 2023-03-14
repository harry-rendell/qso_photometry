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

# %load_ext autoreload
# %autoreload 2

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

obj = 'qsos'
ID  = 'uid'
band = 'all'
redshift_bool = True

dr = analysis(ID, obj)
dr.band = band
dr.read_in(redshift=redshift_bool)
# dr.group(keys = ['uid'],read_in=True, redshift=redshift_bool, survey = 'all')

dr.df.index

uids = dr.df_grouped['mag_std'].sort_values(ascending=False).head(30).index

dr.coords.loc[128193].values

dr.df.loc[128193]

dr.plot_series([128193]);


