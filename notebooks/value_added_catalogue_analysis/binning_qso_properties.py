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
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.preprocessing import parse, data_io, binning
from module.classes.dtdm import dtdm_raw_analysis
from module.classes.analysis import analysis
from module.plotting.common import savefigs

OBJ    = 'qsos'
ID     = 'uid' if OBJ == 'qsos' else 'uid_s'
BAND   = 'r' 
SAVE_FIGS = True

dr = analysis(ID, OBJ, 'r')
dr.read_vac(catalogue_name='dr16q_vac')
dr.vac = parse.filter_data(dr.vac, cfg.PREPROC.VAC_BOUNDS, dropna=False)
# Note, bad nEdd entries are set to 0 (exactly) so we can remove those.
dr.vac['nEdd'] = dr.vac['nEdd'].where((dr.vac['nEdd']!=0).values)
dr.vac.describe()

# +
plt.style.use(cfg.FIG.STYLE_DIR + 'style1.mplstyle')
key = 'Lbol'
bounds_z = np.array([-3.5,-1.5,-1,-0.5,0,0.5,1,1.5,3.5])
hist_kwargs = {'bins':250, 'alpha':1}
ax_kwargs = {'xlabel':r'Bolometric luminosity / erg s$^{-1}$',
             'ylabel':r'Number of Quasars'}

bounds_tuple, z_score_val, bounds_values, mean, std, fig = binning.calculate_bins_and_z_scores(dr.vac[key], key, bounds = bounds_z, plot=True, hist_kwargs=hist_kwargs, ax_kwargs=ax_kwargs)
if SAVE_FIGS:
    savefigs(fig, f'ENSEMBLE-PROPERTIES-vac_bounds_{OBJ}_{key}', 'chap3')


# +
plt.style.use(cfg.FIG.STYLE_DIR + 'style1.mplstyle')
key = 'MBH'
bounds_z = np.array([-3.5,-1.5,-1,-0.5,0,0.5,1,1.5,3.5])
hist_kwargs = {'bins':250, 'alpha':1}
ax_kwargs = {'xlabel':r'Black hole mass / $\log_{10}(M_{\mathrm{BH}}/M_\odot)$',
             'ylabel':r'Number of Quasars'}

bounds_tuple, z_score_val, bounds_values, mean, std, fig = binning.calculate_bins_and_z_scores(dr.vac[key], key, bounds = bounds_z, plot=True, hist_kwargs=hist_kwargs, ax_kwargs=ax_kwargs)
if SAVE_FIGS:
    savefigs(fig, f'ENSEMBLE-PROPERTIES-vac_bounds_{OBJ}_{key}', 'chap3')


# +
plt.style.use(cfg.FIG.STYLE_DIR + 'style1.mplstyle')
key = 'nEdd'
bounds_z = np.array([-3.5,-1.5,-1,-0.5,0,0.5,1,1.5,3.5])
hist_kwargs = {'bins':250, 'alpha':1}
ax_kwargs = {'xlabel':r'Eddington ratio / $\log_{10}(n_{\mathrm{Edd}})$',
             'ylabel':r'Number of Quasars'}

bounds_tuple, z_score_val, bounds_values, mean, std, fig = binning.calculate_bins_and_z_scores(dr.vac[key], key, bounds = bounds_z, plot=True, hist_kwargs=hist_kwargs, ax_kwargs=ax_kwargs)
if SAVE_FIGS:
    savefigs(fig, f'ENSEMBLE-PROPERTIES-vac_bounds_{OBJ}_{key}', 'chap3') 

