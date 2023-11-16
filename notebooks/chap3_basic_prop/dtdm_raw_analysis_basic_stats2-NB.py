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
# jupytext --to py dtdm_raw_analysis_basic_stats2-NB.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb dtdm_raw_analysis_basic_stats-NB.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.classes.dtdm import dtdm_raw_analysis
from module.plotting.common import savefigs
import matplotlib.pyplot as plt
import numpy as np

# # Notebook for Chapter 3: Basic ensemble properties
# - ∆m distributions
#     - Fitting with exp/gauss/lorentz
#
# - Drift
#     - all survey data
#     - inner only (contradicting above!)
# - Skewness
#     - interpretation?
# - Kurtosis
#     - interpret and compare to gaussian
#     
#

# # Configurables

SAVE_FIGS = False
PLOT_FITS = True

# # Read in data

# +
# from module.assets import load_grouped_tot
# mag_med = load_grouped_tot('qsos', 'r', usecols=['mag_med']).squeeze()
# (mag_med<20.5).sum()/len(mag_med)

# +
# Set up the dtdm analysis object and load the data

# QSOS
# name = 'log_40'
name = 'log_20_inner'
dtdm_qsos_r_inner = dtdm_raw_analysis('qsos', 'r', 'qsos')
dtdm_qsos_r_inner.read_pooled_stats(name, key='all')

dtdm_qsos_g_inner = dtdm_raw_analysis('qsos', 'g', 'qsos')
dtdm_qsos_g_inner.read_pooled_stats(name, key='all')

dtdm_qsos_i_inner = dtdm_raw_analysis('qsos', 'i', 'qsos')
dtdm_qsos_i_inner.read_pooled_stats(name, key='all')

# STARS
# name = 'log_30'
name = 'log_20_inner'
dtdm_star_r_inner = dtdm_raw_analysis('calibStars', 'r', 'calibStars')
dtdm_star_r_inner.read_pooled_stats(name, key='all')

dtdm_star_g_inner = dtdm_raw_analysis('calibStars', 'g', 'calibStars')
dtdm_star_g_inner.read_pooled_stats(name, key='all')

dtdm_star_i_inner = dtdm_raw_analysis('calibStars', 'i', 'calibStars')
dtdm_star_i_inner.read_pooled_stats(name, key='all')

# Show the features that we can plot
print(dtdm_qsos_r_inner.pooled_stats.keys())

# +
# Set up the dtdm analysis object and load the data

# QSOS
# name = 'log_40'
name = 'log_30'
# name = 'log_20_inner'
dtdm_qsos_r = dtdm_raw_analysis('qsos', 'r', 'qsos')
dtdm_qsos_r.read_pooled_stats(name, key='all')

dtdm_qsos_g = dtdm_raw_analysis('qsos', 'g', 'qsos')
dtdm_qsos_g.read_pooled_stats(name, key='all')

dtdm_qsos_i = dtdm_raw_analysis('qsos', 'i', 'qsos')
dtdm_qsos_i.read_pooled_stats(name, key='all')

name = 'log_30_mag_max_20.5'
# name = 'log_20_inner'
dtdm_qsos_r_bright = dtdm_raw_analysis('qsos', 'r', 'qsos')
dtdm_qsos_r_bright.read_pooled_stats(name, key='all')

dtdm_qsos_g_bright = dtdm_raw_analysis('qsos', 'g', 'qsos')
dtdm_qsos_g_bright.read_pooled_stats(name, key='all')

dtdm_qsos_i_bright = dtdm_raw_analysis('qsos', 'i', 'qsos')
dtdm_qsos_i_bright.read_pooled_stats(name, key='all')

# STARS
name = 'log_30'
# name = 'log_20_inner'
dtdm_star_r = dtdm_raw_analysis('calibStars', 'r', 'calibStars')
dtdm_star_r.read_pooled_stats(name, key='all')

dtdm_star_g = dtdm_raw_analysis('calibStars', 'g', 'calibStars')
dtdm_star_g.read_pooled_stats(name, key='all')

dtdm_star_i = dtdm_raw_analysis('calibStars', 'i', 'calibStars')
dtdm_star_i.read_pooled_stats(name, key='all')

# Show the features that we can plot
print(dtdm_qsos_r.pooled_stats.keys())
# -

# # Plot ensemble drift for all survey data

# Set plotting style
plt.style.use('default')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['mean weighted a']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Mean offset (mag)'}

# +
fig, ax = plt.subplots(3,1, figsize=(10,12), sharex=True)
plt.style.use('default')
plt.style.use(cfg.RES_DIR + 'stylelib/paired_bands.mplstyle')
ylim = (-0.6,0.4) # wide view
# ylim = (-0.1,0.1) # narrow view
# g
kwargs['ylabel'] = r'mean $\Delta g$ (mag)'
ax[0].set_prop_cycle(color=['33a02c', 'b2df8a'])
dtdm_qsos_g.plot_comparison_data(ax[0], 'caplar')
dtdm_qsos_g.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Quasars', ylim=ylim, xlim=(1,3e4), **kwargs)
dtdm_star_g.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Stars', legend_loc='upper left', xlabel='')

# r
kwargs['ylabel'] = r'mean $\Delta r$ (mag)'
ax[1].set_prop_cycle(color=['e31a1c', 'F98583'])
dtdm_qsos_r.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Quasars', ylim=ylim, xlim=(1,3e4), **kwargs)
dtdm_star_r.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Stars', legend_loc='upper left', xlabel='')
# i
kwargs['ylabel'] = r'mean $\Delta i$ (mag)'
ax[2].set_prop_cycle(color=['6a3d9a', 'C299D6'])
dtdm_qsos_i.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Quasars', ylim=ylim, xlim=(1,3e4), **kwargs)
dtdm_star_i.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Stars', legend_loc='upper left')

plt.subplots_adjust(hspace=0.05)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_gri_wide', 'chap3')
# -

# # Plot ensemble drift for inner only

# Set plotting style
plt.style.use('paired_inverse')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['mean weighted b']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Drift'}

# +
fig, ax = plt.subplots(3,1, figsize=(10,12), sharex=True)
plt.style.use('default')
plt.style.use(cfg.RES_DIR + 'stylelib/paired_bands.mplstyle')
# ylim = (-0.1,0.3) # wide view
ylim = (-0.1,0.1) # narrow view
# g
kwargs['ylabel'] = r'mean $\Delta g$ (mag)'
ax[0].set_prop_cycle(color=['33a02c', 'b2df8a'])
dtdm_qsos_g_inner.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_g_inner.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Stars', legend_loc='upper left', xlabel='')
# r
kwargs['ylabel'] = r'mean $\Delta r$ (mag)'
ax[1].set_prop_cycle(color=['e31a1c', 'F98583'])
dtdm_qsos_r_inner.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_r_inner.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Stars', legend_loc='upper left', xlabel='')
# i
kwargs['ylabel'] = r'mean $\Delta i$ (mag)'
ax[2].set_prop_cycle(color=['6a3d9a', 'C299D6'])
dtdm_qsos_i_inner.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_i_inner.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Stars', legend_loc='upper left')

plt.subplots_adjust(hspace=0.05)
# if SAVE_FIGS:
savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_inner_gri', 'chap3')
# -

# # Compare between inner and all survey data

# Set plotting style
plt.style.use('default')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['mean weighted a']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Mean offset (mag)'}

# ## $r$ band

dtdm_qsos_r.pooled_stats.keys()

# +
plt.style.use('paired_inverse')
# plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')
fig, ax = dtdm_qsos_r.plot_stats(plotting_keys, figax=None, label='Quasars', ylim=(-0.15,0.35), xlim=(1,3e4), **kwargs)
fig, ax = dtdm_qsos_r_inner.plot_stats(plotting_keys, figax=(fig,ax), label='Quasars, inner')

fig, ax = dtdm_star_r.plot_stats(plotting_keys, figax=(fig, ax), label='Stars')
fig, ax = dtdm_star_r_inner.plot_stats(plotting_keys, figax=(fig, ax), label='Stars, inner')
if SAVE_FIGS:
    # savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_r', 'chap3')
    savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_inner_comparison_r', 'chap3')
# -

# ## $g$ band

# +
plt.style.use('paired_inverse')
# plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')

fig, ax = dtdm_qsos_g.plot_stats(plotting_keys, figax=None, label='Quasars', ylim=(-0.15,1.1), xlim=(1,3e4), **kwargs)
fig, ax = dtdm_qsos_g_inner.plot_stats(plotting_keys, figax=(fig,ax), label='Quasars, inner')
# fig, ax = dtdm_qsos_g_bright.plot_stats(plotting_keys, figax=(fig,ax), label='Quasars, brightest')

fig, ax = dtdm_star_g.plot_stats(plotting_keys, figax=(fig, ax), label='Stars')
fig, ax = dtdm_star_g_inner.plot_stats(plotting_keys, figax=(fig, ax), label='Stars, inner')

if SAVE_FIGS:
    savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_g', 'chap3')
    savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_inner_comparison_g', 'chap3')
# -

# ## $i$ band

# +
plt.style.use('paired_inverse')

fig, ax = dtdm_qsos_i.plot_stats(plotting_keys, figax=None, label='Quasars', ylim=(-0.2,0.8), xlim=(1,3e4), **kwargs)
fig, ax = dtdm_qsos_i_inner.plot_stats(plotting_keys, figax=(fig,ax), label='Quasars (inner)')
fig, ax = dtdm_qsos_i_bright.plot_stats(plotting_keys, figax=(fig,ax), label='Quasars, brightest')

fig, ax = dtdm_star_i.plot_stats(plotting_keys, figax=(fig, ax), label='Stars')
fig, ax = dtdm_star_i_inner.plot_stats(plotting_keys, figax=(fig, ax), label='Stars (inner)')

if SAVE_FIGS:
    savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_i', 'chap3')
    # savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_inner_comparison_i', 'chap3')
# -

# # Plot kurtosis

# Set plotting style
plt.style.use('paired_inverse')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['kurtosis']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Kurtosis'}

# +
fig, ax = plt.subplots(3,1, figsize=(10,12), sharex=True)
for ax_ in ax:
    ax_.axhline(3, color='k', linestyle=(0,(5,3)), lw=0.7)
plt.style.use('default')
plt.style.use(cfg.RES_DIR + 'stylelib/paired_bands.mplstyle')
ylim = (0,9) # wide view
# ylim = (-0.1,0.1) # narrow view
# g
kwargs['ylabel'] = r'Kurtosis'
ax[0].set_prop_cycle(color=['33a02c', 'b2df8a'])
dtdm_qsos_g.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_g.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Stars', legend_loc='upper right', xlabel='')
# r
kwargs['ylabel'] = r'Kurtosis'
ax[1].set_prop_cycle(color=['e31a1c', 'F98583'])
dtdm_qsos_r.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_r.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Stars', legend_loc='upper right', xlabel='')
# i
kwargs['ylabel'] = r'Kurtosis'
ax[2].set_prop_cycle(color=['6a3d9a', 'C299D6'])
dtdm_qsos_i.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Quasars', ylim=(0,25), xlim=(1,2e4), **kwargs)
dtdm_star_i.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Stars', legend_loc='upper right')

plt.subplots_adjust(hspace=0.05)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/KURTOSIS-ENSEMBLE-qsos_and_stars_gri', 'chap3')
# -

# # Plot skewness

# Set plotting style
plt.style.use('paired_inverse')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['skewness']
kwargs = {'xscale':'log',
          'yscale':'linear'}

# +
fig, ax = plt.subplots(3,1, figsize=(10,12), sharex=True)
plt.style.use('default')
plt.style.use(cfg.RES_DIR + 'stylelib/paired_bands.mplstyle')
# ylim = (0,15) # wide view
ylim = (-1.5,0.6) # narrow view
# g
kwargs['ylabel'] = r'Skewness'
ax[0].set_prop_cycle(color=['33a02c', 'b2df8a'])
dtdm_qsos_g.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_g.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Stars', legend_loc='lower left', xlabel='')
# r
kwargs['ylabel'] = r'Skewness'
ax[1].set_prop_cycle(color=['e31a1c', 'F98583'])
dtdm_qsos_r.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_r.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Stars', legend_loc='lower left', xlabel='')
# i
kwargs['ylabel'] = r'Skewness'
ax[2].set_prop_cycle(color=['6a3d9a', 'C299D6'])
dtdm_qsos_i.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_i.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Stars', legend_loc='lower left')

plt.subplots_adjust(hspace=0.05)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/SKEWNESS-ENSEMBLE-qsos_and_stars_gri', 'chap3')

# +
fig, ax = plt.subplots(3,1, figsize=(10,12), sharex=True)
plt.style.use('default')
plt.style.use(cfg.RES_DIR + 'stylelib/paired_bands.mplstyle')
# ylim = (0,15) # wide view
ylim = (-0.6,0.6) # narrow view
# g
kwargs['ylabel'] = r'Skewness'
ax[0].set_prop_cycle(color=['33a02c', 'b2df8a'])
dtdm_qsos_g_inner.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_g_inner.plot_stats(plotting_keys, figax=(fig, ax[0]), label='Stars', legend_loc='upper left', xlabel='')
# r
kwargs['ylabel'] = r'Skewness'
ax[1].set_prop_cycle(color=['e31a1c', 'F98583'])
dtdm_qsos_r_inner.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_r_inner.plot_stats(plotting_keys, figax=(fig, ax[1]), label='Stars', legend_loc='upper left', xlabel='')
# i
kwargs['ylabel'] = r'Skewness'
ax[2].set_prop_cycle(color=['6a3d9a', 'C299D6'])
dtdm_qsos_i_inner.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Quasars', ylim=ylim, xlim=(1,2e4), **kwargs)
dtdm_star_i_inner.plot_stats(plotting_keys, figax=(fig, ax[2]), label='Stars', legend_loc='upper left')

plt.subplots_adjust(hspace=0.05)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/SKEWNESS-ENSEMBLE-qsos_and_stars_inner_gri', 'chap3')
