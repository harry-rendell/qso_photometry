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

# +
# Set up the dtdm analysis object and load the data

# STARS
name = 'log_30'
dtdm_star_r = dtdm_raw_analysis('calibStars', 'r', 'calibStars')
dtdm_star_r.read_pooled_stats(name, key='all')

dtdm_star_g = dtdm_raw_analysis('calibStars', 'g', 'calibStars')
dtdm_star_g.read_pooled_stats(name, key='all')

dtdm_star_i = dtdm_raw_analysis('calibStars', 'i', 'calibStars')
dtdm_star_i.read_pooled_stats(name, key='all')

# Show the features that we can plot
print(dtdm_star_r.pooled_stats.keys())
# -

# # Plot SF

# Set plotting style
plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')
# Available keys:
# ['SF cwf p', 'mean weighted b', 'n', 'SF cwf a', 'kurtosis', 'skewness', 'SF cwf b', 'mean weighted a', 'SF cwf n']
plotting_keys = ['SF cwf a']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Structure Function',
          'ylim':(-1.5e-1,1.5e-1)}

kwargs['label'] = 'stars, $r$ band'
fig, ax = dtdm_star_r.plot_stats(plotting_keys, figax=None, **kwargs)
kwargs['label'] = 'stars, $g$ band'
fig, ax = dtdm_star_g.plot_stats(plotting_keys, figax=(fig,ax), **kwargs)
kwargs['label'] = 'stars, $i$ band'
fig, ax = dtdm_star_i.plot_stats(plotting_keys, figax=(fig,ax), **kwargs)

# # Plot SF asymmetry in stars

plotting_keys = ['SF cwf p','SF cwf n']
kwargs = {'xscale':'log',
          'yscale':'linear',
          'ylabel': 'Structure Function',
          'ylim':(-1e-1,2e-1)}

# ## $r$ band

fig, ax = dtdm_star_r.plot_stats(plotting_keys, figax=None, **kwargs)

# ## $g$ band

fig, ax = dtdm_star_g.plot_stats(plotting_keys, figax=None, **kwargs)

# ## $i$ band

fig, ax = dtdm_star_i.plot_stats(plotting_keys, figax=None, **kwargs)


