# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: astro
#     language: python
#     name: python3
# ---

# + language="bash"
# jupytext --to py plot_pairing_radius-NB.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb plot_pairing_radius-NB.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.plotting.common import savefigs
import seaborn as sns

OBJ = 'qsos'
ID  = 'uid' if OBJ == 'qsos' else 'uid_s'
SURVEYS = ['ssa', 'sdss', 'ps', 'ztf']

# # Method 1: Plot histogram of distances for all observations

from module.preprocessing.data_io import dispatch_reader
basepath = os.path.join(cfg.RES_DIR, 'plot_data')
fnames   = [f'distances_{survey}_{OBJ}.csv' for survey in SURVEYS]
# put the result in a dictionary
dfs = {name:df for name, df in zip(SURVEYS, dispatch_reader({'basepath':basepath, 'ID':ID}, concat=False, fnames=fnames))}

# +
print('Max pairing radius in each survey:')
for name, survey in dfs.items():
    print(f'{name}\t{survey.values.max()}')

# print number of unique indices in each survey:
# Note, this could include mismatches as pariring radius is 1"
print('Number of objects in each survey:')
for name, survey in dfs.items():
    print(f'{name}\t{len(survey.index.unique()):,}')
# -

# # Method 2: Plot average distance per uid
# ### Note, this will distances for primary and secondary objects (if present) 

# +
import functools as ft
from scipy.stats import median_abs_deviation as MAD
from multiprocessing import Pool

def process_survey(survey):
    df = pd.read_csv(os.path.join(cfg.RES_DIR, 'plot_data', f'distances_{survey}_{OBJ}.csv'), index_col=ID, nrows=None).squeeze('columns').rename(survey)
    return df.groupby('uid').agg({(f'{survey}_mean', np.mean), (f'{survey}_std', np.std), (f'{survey}_MAD', MAD)})

def groupby_and_concat_distances():
    # Create a Pool with the number of available CPU cores
    num_cores = 4
    with Pool(processes=num_cores) as pool:
        dist_stats = pool.map(process_survey, SURVEYS)

    df_inner = ft.reduce(lambda left, right: pd.merge(left, right, on=ID, how='inner'), dist_stats).sort_index()
    df_outer = ft.reduce(lambda left, right: pd.merge(left, right, on=ID, how='outer'), dist_stats).sort_index()
    return df_inner, df_outer



# -

df_inner, df_outer = groupby_and_concat_distances()

# plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')
plt.style.use('default')
def distance_histplot(df):
    n_bins = 50
    alpha = 0.7
    binrange = (0,2)
    stat = 'count'
    line_kws = {'linewidth': 0.1}
    fig, ax = plt.subplots(1,1, figsize=(10,7))
    for survey in ['ssa','ztf','ps','sdss']:
        ax = sns.histplot(data=df, x=survey+'_mean', stat=stat, bins=n_bins, label=survey.upper(), alpha=alpha, binrange=binrange, **line_kws)
    ax.set(yscale='log', xlabel='Pairing radius (arcsec)')
    ax.legend()


distance_histplot(df_inner)
distance_histplot(df_outer)


