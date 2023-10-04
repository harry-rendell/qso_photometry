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
# jupytext --to py plot_venn-NB.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb plot_venn-NB.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.plotting.common import savefigs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn3_unweighted, venn3_circles
import seaborn as sns

obj = 'qsos'
ID  = 'uid' if obj == 'qsos' else 'uid_s'
band = 'r'

if obj == 'qsos':   
    redshift = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr14q/dr14q_redshift.csv').set_index(ID)
sdss = {band:pd.read_csv(cfg.D_DIR + 'surveys/sdss/{}/clean/{}_band/grouped.csv'.format(obj,band)       , index_col=0) for band in 'gri'}
ps   = {band:pd.read_csv(cfg.D_DIR + 'surveys/ps/{}/clean/{}_band/grouped.csv'.format(obj,band)         , index_col=0) for band in 'gri'}
ztf  = {band:pd.read_csv(cfg.D_DIR + 'surveys/ztf/{}/clean/{}_band/grouped.csv'.format(obj,band)        , index_col=0) for band in 'gri'}
ssa  = {band:pd.read_csv(cfg.D_DIR + 'surveys/ssa/{}/clean/{}_band/grouped.csv'.format(obj,band), index_col=0) for band in 'gri'}
tot  = {band:pd.read_csv(cfg.D_DIR + 'merged/{}/clean/grouped_{}.csv'.format(obj,band)                  , index_col=0) for band in 'gri'}
surveys = {'ssa':ssa, 'sdss':sdss, 'ps':ps, 'ztf':ztf}

SAVE_FIGS = False

sets = pd.read_csv(cfg.D_DIR + 'catalogues/{}/sets/clean_{}.csv'.format(obj,band), index_col=ID, comment='#')

sets['sdss'].sum()

[sets.index[sets[s.lower()].values] for s in ['SDSS', 'PS', 'ZTF']]

[set(sdss[band].index), set(ps[band].index), set(ztf[band].index)]

# +
fig, ax = plt.subplots(1,1, figsize=(10,8))
total = len(sdss[band].index.union(ztf[band].index.union(ps[band].index)))

v1 = venn3_unweighted(
    [set(sdss[band].index), set(ps[band].index), set(ztf[band].index)],
    set_labels=['SDSS','PS','ZTF'],
    set_colors=['#FBC599','#DB5375','#75B2E3','#9C6AD2'],
    subset_label_formatter=lambda x: f"{(x/total):1.0%}",
    ax=ax,
    alpha=1
)

venn3_circles([1]*7, ax=ax, lw=0.5)
if SAVE_FIGS:
    savefigs(fig, 'SURVEY-DATA-venn_diagram', 'chap2')

# +
fig, ax = plt.subplots(1,1, figsize=(10,8))
surveys = ['sdss', 'ps', 'ztf']
total = np.any([sets[s].values for s in surveys], axis=0).sum()

v1 = venn3_unweighted(
    [set(sets.index[sets[s].values]) for s in surveys],
    set_labels=surveys,
    set_colors=['#FBC599','#DB5375','#75B2E3','#9C6AD2'],
    subset_label_formatter=lambda x: f"{(x/total):1.0%}",
    ax=ax,
    alpha=1
)

venn3_circles([1]*7, ax=ax, lw=0.5)
if SAVE_FIGS:
    savefigs(fig, 'SURVEY-DATA-venn_diagram', 'chap2')
# -


