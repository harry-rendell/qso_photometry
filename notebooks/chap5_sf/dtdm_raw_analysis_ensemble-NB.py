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

# + language="bash"
# jupytext --to py dtdm_raw_analysis_ensemble-NB.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb dtdm_raw_analysis_ensemble-NB.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.classes.dtdm import dtdm_raw_analysis
from module.plotting.common import savefigs
from module.modelling.models import bkn_pow, bkn_pow_smooth

# from module.modelling.models import broken_power_law


# +
# Set up the dtdm analysis object and load the data

# QSOS
name = 'log_40'
dtdm_qsos_r = dtdm_raw_analysis('qsos', 'r', 'qsos')
dtdm_qsos_r.read_pooled_stats(name, key='all')

dtdm_qsos_g = dtdm_raw_analysis('qsos', 'g', 'qsos')
dtdm_qsos_g.read_pooled_stats(name, key='all')

dtdm_qsos_i = dtdm_raw_analysis('qsos', 'i', 'qsos')
dtdm_qsos_i.read_pooled_stats('log', key='all')

# STARS
name = 'log_30'
dtdm_star_r = dtdm_raw_analysis('calibStars', 'r', 'calibStars')
dtdm_star_r.read_pooled_stats(name, key='all')

dtdm_star_g = dtdm_raw_analysis('calibStars', 'g', 'calibStars')
dtdm_star_g.read_pooled_stats(name, key='all')

dtdm_star_i = dtdm_raw_analysis('calibStars', 'i', 'calibStars')
dtdm_star_i.read_pooled_stats(name, key='all')

# Show the features that we can plot
print(dtdm_qsos_r.pooled_stats.keys())
# -

SAVE_FIGS = False

# # Plot SF

# Set plotting style
plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['SF cwf a']
kwargs = {'xscale':'log',
          'ylabel': 'Structure Function'
          }

# ## $r$ band

# +
fig, ax = dtdm_qsos_r.plot_stats(plotting_keys, figax=None, label='Quasars', yscale='linear', ylim=(-0.25,1.2), **kwargs)

# Fit power law
coefficient, exponent = dtdm_qsos_r.fit_stats(plotting_keys[0], 'power_law', ax=ax, value_range=[0,4], x_fit_bounds=[1,4], n_model_points=100)

# Fit broken power law
e=0.2
fit_kwargs = {'bounds':[(0.3-e, 0.3+e), (1e2, 1e4), (0.2, 0.8), (0.2, 0.8)],
          'x0':[0.3, 1e2, 0.3, 0.3]}
least_sq_kwargs = {'loss':'cauchy'}
dtdm_qsos_r.fit_stats(plotting_keys[0], 'broken_power_law_minimize', ax=ax, least_sq_kwargs=least_sq_kwargs, **fit_kwargs)

fig, ax = dtdm_star_r.plot_stats(plotting_keys, figax=(fig, ax), label='Stars', yscale='linear', ylim=(-0.25,1.2), **kwargs)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/SF-ENSEMBLE-qsos_and_stars_r', 'chap4')

# +
fig, ax = dtdm_qsos_r.plot_stats(plotting_keys, figax=None, label='Quasars', yscale='log', ylim=(1e-2,1.5), **kwargs)

# Fit power law
coefficient, exponent = dtdm_qsos_r.fit_stats(plotting_keys[0], 'power_law', ax=ax, value_range=[0,4], x_fit_bounds=[1,4], n_model_points=100)

# Fit broken power law
e=0.2
fit_kwargs = {'bounds':[(0.3-e, 0.3+e), (1e2, 1e4), (0.2, 0.8), (0.2, 0.8)],
          'x0':[0.3, 1e2, 0.3, 0.3]}
least_sq_kwargs = {'loss':'cauchy'}
dtdm_qsos_r.fit_stats(plotting_keys[0], 'broken_power_law_minimize', ax=ax, least_sq_kwargs=least_sq_kwargs, **fit_kwargs)

fig, ax = dtdm_star_r.plot_stats(plotting_keys, figax=(fig, ax), **kwargs)
# -

# ## $g$ band

# +
fig, ax = dtdm_qsos_g.plot_stats(plotting_keys, figax=None, label='Quasars', yscale='linear', ylim=(-0.25,1.2), **kwargs)

# Fit power law
coefficient, exponent = dtdm_qsos_r.fit_stats(plotting_keys[0], 'power_law', ax=ax, value_range=[0,4], x_fit_bounds=[1,4], n_model_points=100)

# Fit broken power law
e=0.2
fit_kwargs = {'bounds':[(0.3-e, 0.3+e), (1e2, 1e4), (0.2, 0.8), (0.2, 0.8)],
              'x0':[0.3, 1e2, 0.3, 0.3]}
least_sq_kwargs = {'loss':'cauchy'}
dtdm_qsos_g.fit_stats(plotting_keys[0], 'broken_power_law_minimize', ax=ax, least_sq_kwargs=least_sq_kwargs, **fit_kwargs)

fig, ax = dtdm_star_g.plot_stats(plotting_keys, figax=(fig, ax), label='Stars', yscale='linear', ylim=(-0.25,1.2),**kwargs)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/SF-ENSEMBLE-qsos_and_stars_g', 'chap4')

# +
fig, ax = dtdm_qsos_g.plot_stats(plotting_keys, figax=None, label='Quasars', yscale='log', ylim=(1e-2,1.5), **kwargs)
# Fit power law
coefficient, exponent = dtdm_qsos_g.fit_stats(plotting_keys[0], 'power_law', ax=ax, value_range=[0,4], x_fit_bounds=[1,4], n_model_points=100) # value_range and x_fit_bounds are log(mjd)

# Fit broken power law
e=0.2
fit_kwargs = {'bounds':[(0.3-e, 0.3+e), (1e2, 1e4), (0.2, 0.8), (0.2, 0.8)],
              'x0':[0.3, 1e2, 0.3, 0.3]}
least_sq_kwargs = {'loss':'cauchy'}
dtdm_qsos_g.fit_stats(plotting_keys[0], 'broken_power_law_minimize', ax=ax, least_sq_kwargs=least_sq_kwargs, **fit_kwargs)

# Plot stars
fig, ax = dtdm_star_g.plot_stats(plotting_keys, figax=(fig, ax), label='Stars', yscale='log', ylim=(1e-2,1.5))
# -

# ## $i$ band

# +
fig, ax = dtdm_qsos_i.plot_stats(plotting_keys, figax=None, label='Quasars', yscale='linear', ylim=(-0.25,1.2), **kwargs)
# Fit power law
coefficient, exponent = dtdm_qsos_g.fit_stats(plotting_keys[0], 'power_law', ax=ax, value_range=[0,4], x_fit_bounds=[1,4], n_model_points=100) # value_range and x_fit_bounds are log(mjd)

# Fit broken power law
e=0.2
fit_kwargs = {'bounds':[(0.3-e, 0.3+e), (1e2, 1e4), (0.2, 0.8), (0.2, 0.8)],
              'x0':[0.3, 1e2, 0.3, 0.3]}
least_sq_kwargs = {'loss':'cauchy'}
dtdm_qsos_i.fit_stats(plotting_keys[0], 'broken_power_law_minimize', ax=ax, least_sq_kwargs=least_sq_kwargs, **fit_kwargs)

fig, ax = dtdm_star_i.plot_stats(plotting_keys, figax=(fig, ax), label='Stars', yscale='linear', ylim=(-0.25,1.2), **kwargs)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/SF-ENSEMBLE-qsos_and_stars_i', 'chap4')

# +
fig, ax = dtdm_qsos_i.plot_stats(plotting_keys, figax=None, label='Quasars', yscale='log', ylim=(1e-2,1.5), **kwargs)
# Fit power law
coefficient, exponent = dtdm_qsos_g.fit_stats(plotting_keys[0], 'power_law', ax=ax, value_range=[0,4], x_fit_bounds=[1,4], n_model_points=100) # value_range and x_fit_bounds are log(mjd)

# Fit broken power law
e=0.2
fit_kwargs = {'bounds':[(0.3-e, 0.3+e), (1e2, 1e4), (0.2, 0.8), (0.2, 0.8)],
              'x0':[0.3, 1e2, 0.3, 0.3]}
least_sq_kwargs = {'loss':'cauchy'}
dtdm_qsos_i.fit_stats(plotting_keys[0], 'broken_power_law_minimize', ax=ax, least_sq_kwargs=least_sq_kwargs, **fit_kwargs)

fig, ax = dtdm_star_i.plot_stats(plotting_keys, figax=(fig, ax), label='Stars', yscale='log', ylim=(1e-2,1.5), **kwargs)
# -

# # Plot SF asymmetry

# Set plotting style
plt.style.use(cfg.FIG.STYLE_DIR + 'paired.mplstyle')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['SF cwf p', 'SF cwf n']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Structure Function',
          'ylim':(-0.15,1.5)}

# ## $r$ band

# +
kwargs['yscale'] = 'linear'
kwargs['ylim'] = (-0.5,1)


kwargs['label'] = 'Quasars'
fig, ax = dtdm_qsos_r.plot_stats(plotting_keys, figax=None, **kwargs)
kwargs['label'] = 'Stars'
fig, ax = dtdm_star_r.plot_stats(plotting_keys, figax=(fig, ax), **kwargs)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/SF-ENSEMBLE_ASYM-qsos_and_stars_r', 'chap4')

# compare drift with SF asym
fig, ax = dtdm_qsos_r.plot_stats(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_qsos_r.plot_stats(['mean weighted b'], figax=(fig,ax), **kwargs)
ax.axhline(y=0, lw=0.6, color='k', ls='--')
if SAVE_FIGS:
    savefigs(fig, 'ensemble/SF-ENSEMBLE_ASYM-qsos_and_stars_r_drift_comparison', 'chap4')
# -

# ## $g$ band

kwargs['ylim'] = (-0.15, 1.5)
fig, ax = dtdm_qsos_g.plot_stats(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_star_g.plot_stats(plotting_keys, figax=(fig, ax), **kwargs)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/SF-ENSEMBLE_ASYM-qsos_and_stars_g', 'chap4')

# ## $i$ band

fig, ax = dtdm_qsos_i.plot_stats(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_star_i.plot_stats(plotting_keys, figax=(fig, ax), **kwargs)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/SF-ENSEMBLE_ASYM-qsos_and_stars_i', 'chap4')

# # Plot ensemble drift

# Set plotting style
plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['mean weighted b']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Drift',
          'ylim':(-0.5,0.5)}

# ## $r$ band

fig, ax = dtdm_qsos_r.plot_stats(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_star_r.plot_stats(plotting_keys, figax=(fig, ax), **kwargs)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_r', 'chap4')

# ## $g$ band

fig, ax = dtdm_qsos_g.plot_stats(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_star_g.plot_stats(plotting_keys, figax=(fig, ax), **kwargs)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_g', 'chap4')

# ## $i$ band

fig, ax = dtdm_qsos_i.plot_stats(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_star_i.plot_stats(plotting_keys, figax=(fig, ax), **kwargs)
if SAVE_FIGS:
    savefigs(fig, 'ensemble/DRIFT-ENSEMBLE-qsos_and_stars_i', 'chap4')
