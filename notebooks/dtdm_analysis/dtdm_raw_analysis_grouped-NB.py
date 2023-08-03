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

import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.classes.dtdm import dtdm_raw_analysis
from module.plotting.common import savefigs

# +
# Set up the dtdm analysis object and load the data
name = 'log'
dtdm_qsos_lbol_r = dtdm_raw_analysis('qsos', 'r', 'qsos')
dtdm_qsos_lbol_r.read_pooled_stats(name, key='Lbol')

dtdm_qsos_mbh_r = dtdm_raw_analysis('qsos', 'r', 'qsos')
dtdm_qsos_mbh_r.read_pooled_stats(name, key='MBH')

dtdm_qsos_nedd_r = dtdm_raw_analysis('qsos', 'r', 'qsos')
dtdm_qsos_nedd_r.read_pooled_stats(name, key='nEdd')

dtdm_qsos_lbol_g = dtdm_raw_analysis('qsos', 'g', 'qsos')
dtdm_qsos_lbol_g.read_pooled_stats(name, key='Lbol')

dtdm_qsos_mbh_g = dtdm_raw_analysis('qsos', 'g', 'qsos')
dtdm_qsos_mbh_g.read_pooled_stats(name, key='MBH')

dtdm_qsos_nedd_g = dtdm_raw_analysis('qsos', 'g', 'qsos')
dtdm_qsos_nedd_g.read_pooled_stats(name, key='nEdd')

dtdm_qsos_lbol_i = dtdm_raw_analysis('qsos', 'i', 'qsos')
dtdm_qsos_lbol_i.read_pooled_stats(name, key='Lbol')

dtdm_qsos_mbh_i = dtdm_raw_analysis('qsos', 'i', 'qsos')
dtdm_qsos_mbh_i.read_pooled_stats(name, key='MBH')

dtdm_qsos_nedd_i = dtdm_raw_analysis('qsos', 'i', 'qsos')
dtdm_qsos_nedd_i.read_pooled_stats(name, key='nEdd')

# Show the features that we can plot
print(dtdm_qsos_lbol_r.pooled_stats.keys())
# -

SAVE_FIGS=True

# # Plot SF

# Set plotting style
plt.style.use(cfg.FIG.STYLE_DIR + 'dtdm_raw_analysis.mplstyle')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['SF cwf a']
kwargs = {'xscale':'log',
          'yscale':'log',
          'ylabel':'Structure Function',
          'ylim':(1e-2,1.3)}
fill_between=True

# ## $r$ band

fig1, ax1 = dtdm_qsos_lbol_r.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig2, ax2 = dtdm_qsos_mbh_r.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig3, ax3 = dtdm_qsos_nedd_r.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
if SAVE_FIGS:
    savefigs(fig1, 'grouped/SF-GROUPED-r_lbol', 'chap4')
    savefigs(fig2, 'grouped/SF-GROUPED-r_mbh', 'chap4')
    savefigs(fig3, 'grouped/SF-GROUPED-r_nedd', 'chap4')

# ## $g$ band

fig1, ax1 = dtdm_qsos_lbol_g.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig2, ax2 = dtdm_qsos_mbh_g.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig3, ax3 = dtdm_qsos_nedd_g.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
if SAVE_FIGS:
    savefigs(fig1, 'grouped/SF-GROUPED-g_lbol', 'chap4')
    savefigs(fig2, 'grouped/SF-GROUPED-g_mbh', 'chap4')
    savefigs(fig3, 'grouped/SF-GROUPED-g_nedd', 'chap4')

# ## $i$ band

fig1, ax1 = dtdm_qsos_lbol_i.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig2, ax2 = dtdm_qsos_mbh_i.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig3, ax3 = dtdm_qsos_nedd_i.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
if SAVE_FIGS:
    savefigs(fig1, 'grouped/SF-GROUPED-i_lbol', 'chap4')
    savefigs(fig2, 'grouped/SF-GROUPED-i_mbh', 'chap4')
    savefigs(fig3, 'grouped/SF-GROUPED-i_nedd', 'chap4')

# # Plot mean drift

# +
# Set plotting style
plt.style.use(cfg.FIG.STYLE_DIR + 'dtdm_raw_analysis.mplstyle')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['mean weighted b']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Drift',
          'ylim':(-0.4,0.4)}

fill_between=True
# -

# ## $r$ band

fig1, ax1 = dtdm_qsos_lbol_r.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig2, ax2 = dtdm_qsos_mbh_r.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig3, ax3 = dtdm_qsos_nedd_r.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
if SAVE_FIGS:
    savefigs(fig1, 'grouped/DRIFT-GROUPED-r_lbol', 'chap4')
    savefigs(fig2, 'grouped/DRIFT-GROUPED-r_mbh', 'chap4')
    savefigs(fig3, 'grouped/DRIFT-GROUPED-r_nedd', 'chap4')

# ## $g$ band

fig1, ax1 = dtdm_qsos_lbol_g.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig2, ax2 = dtdm_qsos_mbh_g.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig3, ax3 = dtdm_qsos_nedd_g.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
if SAVE_FIGS:
    savefigs(fig1, 'grouped/DRIFT-GROUPED-g_lbol', 'chap4')
    savefigs(fig2, 'grouped/DRIFT-GROUPED-g_mbh', 'chap4')
    savefigs(fig3, 'grouped/DRIFT-GROUPED-g_nedd', 'chap4')

# ## $i$ band

fig1, ax1 = dtdm_qsos_lbol_i.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig2, ax2 = dtdm_qsos_mbh_i.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
fig3, ax3 = dtdm_qsos_nedd_i.plot_stats_property(plotting_keys, figax=None, fill_between=fill_between, **kwargs)
if SAVE_FIGS:
    savefigs(fig1, 'grouped/DRIFT-GROUPED-i_lbol', 'chap4')
    savefigs(fig2, 'grouped/DRIFT-GROUPED-i_mbh', 'chap4')
    savefigs(fig3, 'grouped/DRIFT-GROUPED-i_nedd', 'chap4')

# # Plot kurtosis

# Set plotting style
plt.style.use(cfg.FIG.STYLE_DIR + 'dtdm_raw_analysis.mplstyle')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['kurtosis']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Kurtosis',
          'ylim':(0,20)}

# ## $r$ band

fig, ax = dtdm_qsos_lbol_r.plot_stats_property(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_qsos_mbh_r.plot_stats_property(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_qsos_nedd_r.plot_stats_property(plotting_keys, figax=None, **kwargs)

# ## $g$ band

fig, ax = dtdm_qsos_lbol_g.plot_stats_property(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_qsos_mbh_g.plot_stats_property(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_qsos_nedd_g.plot_stats_property(plotting_keys, figax=None, **kwargs)

# ## $i$ band

kwargs['ylim'] = (0,30)
fig, ax = dtdm_qsos_lbol_i.plot_stats_property(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_qsos_mbh_i.plot_stats_property(plotting_keys, figax=None, **kwargs)
fig, ax = dtdm_qsos_nedd_i.plot_stats_property(plotting_keys, figax=None, **kwargs)
