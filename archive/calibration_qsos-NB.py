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

# # Assessing the magnitude distributions of the qsos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool
font = {'size'   : 18}
matplotlib.rc('font', **font)
import sys
sys.path.append('../')

from module.analysis.obj_survey import obj_survey
from module.preprocessing.parse import intersection, filter_data
from module.analysis.plotting import plot_magerr_hist, plot_mag_dist

# # Using raw (unaveraged) data

nrows = None
bounds={'mag':(15,25),'magerr':(0,2)}

sdss = obj_survey('sdss', 'qsos', 'uid')
sdss.read_in_raw(nrows)


sdss.df

sdss.df = filter_data(sdss.df, bounds, bands = 'ugriz')

ps = obj_survey('ps', 'qsos', 'uid')
ps.read_in_raw(nrows)
ps.df = filter_data(ps.df, bounds)

ztf = obj_survey('ztf', 'qsos', 'uid')
# ztf.pivot()
ztf.read_in_raw(nrows)
ztf.df = filter_data(ztf.df, bounds)

colors = pd.read_csv('/disk1/hrb/python/data/computed/qsos/colors_sdss.csv', index_col=0)

sss = pd.read_csv('/disk1/hrb/python/data/surveys/supercosmos/qsos/ssa_secondary.csv')

sss

# # Saving groupby frames

# + jupyter={"outputs_hidden": true} active=""
# sdss.pivot(read_in=False, magerr=2)

# + jupyter={"outputs_hidden": true} active=""
# ztf.transform_to_ps(colors=colors)
# ztf.pivot(read_in=False, magerr=2)

# + jupyter={"outputs_hidden": true} active=""
# ps.pivot(read_in=False, magerr=2)

# + active=""
# ######################################################
# -

# takes ~30s, could reduce but need to parallelise
sdss.df, ps.df, ztf.df = intersection(sdss.df,ps.df,ztf.df)

ztf.df

# +
fig, axes = plot_magerr_hist([sdss, ps, ztf], bands='gri', quantiles=[0.09,0.10,0.11,0.12], show_lines=0, savename = None, magerr=0.8)
for ax in axes:
    ax.set(xlim=[0,0.3])

# fig.savefig('../plots/magerr_hist_qsos.pdf', bbox_inches='tight')

# -

# | Band | Max magerr |  SDSS |  PS   | ZTF   |
# |------|------------|-------|-------|-------|
# |  g   |    0.05    | 66.4% | 38.6% | 11.7% |
# |  r   |    0.05    | 61.5% | 39.6% | 14.1% |
# |  i   |    0.05    | 51.7% | 36.7% | 11.5% |
# ||||||
# |  g   |    0.10    | 91.9% | 69.9% | 41.0% |
# |  r   |    0.10    | 88.4% | 70.3% | 42.3% |
# |  i   |    0.10    | 80.7% | 67.4% | 43.0% |
# ||||||
# |  g   |    0.15    | 97.2% | 88.0% | 67.4% |
# |  r   |    0.15    | 96.2% | 87.8% | 68.6% |
# |  i   |    0.15    | 92.7% | 86.1% | 76.4% |
# ||||||
# |  g   |    0.20    | 98.9% | 97.8% | 90.0% |
# |  r   |    0.20    | 98.7% | 97.7% | 90.6% |
# |  i   |    0.20    | 97.1% | 97.3% | 99.0% |
# ||||||
# |  g   |    0.25    | 99.4% | 99.8% | 99.0% |
# |  r   |    0.25    | 99.4% | 99.8% | 98.7% |
# |  i   |    0.25    | 98.6% | 99.8% | 99.8% |
#

# | band | max magerr |  SDSS |  PS   | ZTF   |
# |------|------------|-------|-------|-------|
# |  g   |     80%    | 0.0657| 0.1234| 0.1751|
# |  r   |     80%    | 0.0736| 0.1234| 0.1731|
# |  i   |     80%    | 0.0935| 0.1313| 0.1552|
# |  z   |     80%    | 0.2289| 0.1512|   -   |
#

import seaborn as sns
sdss.correlate_mag_magerr_hist_sns('g', 2e0,1e6, save=0)
sdss.correlate_mag_magerr_hist_sns('r', 2e0,1e6, save=0)
sdss.correlate_mag_magerr_hist_sns('i', 2e0,1e6, save=0)

# # Using averaged data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib 
font = {'size' : 18}
matplotlib.rc('font', **font)
import sys
sys.path.append('../')
path = '/disk1/hrb/python/'

from module.analysis.obj_survey import obj_survey#, reader
from module.preprocessing.parse import intersection, filter_data
from module.analysis.plotting import plot_magerr_hist, plot_mag_dist

# colors = pd.read_csv('computed/colors_ps_last.csv', index_col=0)
colors = pd.read_csv(path + 'data/computed/qsos/colors_sdss.csv', index_col=0)

# +
#if any observation has a value outside these bounds, that observation is NaN'd
bounds_generic  = {'mean':(15,25), 'meanerr':(0,2)} 
bounds_specific_fn = lambda x: {'magerr_max_'+band:value for band,value in zip('griz',zip([0,0,0,0],x))}
bands = 'griz'
magerr = 2
pop = 80

print('SDSS')
sdss = obj_survey('sdss', 'qsos', 'uid')
sdss.pivot(magerr=magerr)
# for b in bands:
#     sdss.df_pivot['mean_sdss_'+b] = sdss.df_pivot['mean_'+b]
sdss.df_pivot = filter_data(sdss.df_pivot, bounds_generic, bounds_specific_fn([0.066, 0.074, 0.094, 0.229]), bands) #80
sdss.transform_avg_to_ps(colors, 'mean_gr', 'griz', system='tonry')
sdss.residual({'g':0.0148, 'r':0.0049, 'i':0.0198, 'z':0.042})
# colors = compute_colors(sdss)
# sdss.df_pivot = filter_data(sdss.df_pivot, {'mean_gi':(0.4,2.7)}) 

print('\nPS')
ps = obj_survey('ps', 'qsos', 'uid')
ps.pivot(magerr=magerr)
ps.df_pivot = filter_data(ps.df_pivot, bounds_generic, bounds_specific_fn([0.123, 0.123, 0.131, 0.151]), bands) #80
# duplicating rows to keep the naing convention consistent
for b in bands:
    ps.df_pivot['mean_ps_'+b] = ps.df_pivot['mean_'+b]

bands='gri'
print('\nZTF')
ztf = obj_survey('ztf', 'qsos', 'uid')
ztf.pivot(magerr=magerr)
ztf.df_pivot = filter_data(ztf.df_pivot, bounds_generic, bounds_specific_fn([0.175, 0.173, 0.155]), bands) #80
ztf.residual({'g':0.0074, 'r':-0.0099, 'i':0})
# -

sdss.df_pivot, ps.df_pivot, ztf.df_pivot = intersection(sdss.df_pivot, ps.df_pivot, ztf.df_pivot)
# sdss.df_pivot, ps.df_pivot  = intersection(sdss.df_pivot, ps.df_pivot)
# sdss.df_pivot, ztf.df_pivot = intersection(sdss.df_pivot, ztf.df_pivot)
# ztf.df_pivot , ps.df_pivot  = intersection(ztf.df_pivot , ps.df_pivot)

ps .df_pivot = ps .df_pivot.join(colors, on=ps.ID)
ztf.df_pivot = ztf.df_pivot.join(colors, on=ztf.ID)

# +
# ztf.df_pivot_g = ztf.df_pivot['mean_g'].dropna()
# sdss.df_pivot_g = sdss.df_pivot['mean_g'].dropna()
# ps.df_pivot_g = ps.df_pivot['mean_g'].dropna(
# sdss.df_pivot_g, ps.df_pivot_g, ztf.df_pivot_g = intersection(sdss.df_pivot_g,ps.df_pivot_g,ztf.df_pivot_g)
# -

ax = plot_mag_dist(sdss, ps, ztf, inner=True, save=1)

mag_bounds = [16,19,21]
system='_ps'
bins=101

sdss_mag_g = sdss.df_pivot['mean_g']
sdss.calculate_pop_mask('g', bounds=mag_bounds)

ps.calculate_pop_mask('g', bounds=mag_bounds)
ps.calculate_offset(sdss, system=system, bands='griz')


ztf.calculate_pop_mask('g', bounds=mag_bounds)
ztf.calculate_offset(sdss, system=system, bands='gri')
ztf.calculate_offset(ps, system=system, bands='gri')


axes = ps.plot_offset_distrib(sdss, bands='griz', scale='linear', save=1, range=(-1,1), density=True, bins=bins)

axes = ztf.plot_offset_distrib(ps, bands='gri', scale='linear', save=1, range=(-1,1), density=True, bins=bins)

axes = ztf.plot_offset_distrib(sdss, bands='gri', scale='linear', save=1, range=(-1,1), density=True, bins=bins)

# #### Cells below plots a 2D histogram of the offset vs colour (try combine this into a grid of subplots)

# +
cmap = 'jet'

# ps.correlate_offset_color_hist_sns(sdss,'g', 'mean_gr', 1e0,1e5)
# ps.correlate_ofset_color_hist_sns(sdss,'g', 'mean_ri', 1e0,1e5)

# ps.correlate_offset_color_hist_sns(sdss,'r', 'mean_gr', 1e0,1e5)
# ps.correlate_offset_color_hist_sns(sdss,'r', 'mean_ri', 1e0,1e5)

# ps.correlate_offset_sns(sdss, 'i', 'mean_gr', 1e0,1e5, yrange=(-0.1,0.1))
# ps.correlate_offset_sns(sdss,'i', 'mean_ri', 1e0,1e5)

# ps.correlate_offset_sns(sdss,'z', 'mean_gr', 1e0,1e5)
# ps.correlate_offset_sns(sdss,'z', 'mean_ri', 1e0,1e5)
# xrange = (-0.5,2)
xrange = (16,21)
ps.correlate_offset_sns(sdss, 'g', 'mean_g', pop, 1e1,3e3, xrange=xrange, yrange=(-1,1), save=1, cmap=cmap)
ps.correlate_offset_sns(sdss, 'r', 'mean_r', pop, 1e1,3e3, xrange=xrange, yrange=(-1,1), save=1, cmap=cmap)
ps.correlate_offset_sns(sdss, 'i', 'mean_i', pop, 1e1,3e3, xrange=xrange, yrange=(-1,1), save=1, cmap=cmap)
# -

cmap = 'jet'
ztf.correlate_offset_sns(ps, 'g', 'mean_gr', pop, 1e1,3e3, xrange=xrange, yrange=(-1,1), save=1, cmap=cmap)
ztf.correlate_offset_sns(ps, 'r', 'mean_ri', pop, 1e1,3e3, xrange=xrange, yrange=(-1,1), save=1, cmap=cmap)
ztf.correlate_offset_sns(ps, 'i', 'mean_iz', pop, 1e1,3e3, xrange=xrange, yrange=(-1,1), save=1, cmap=cmap)

cmap = 'jet'
ztf.correlate_offset_sns(sdss, 'g', 'mean_gr', pop, 1e1,3e3, xrange=xrange, yrange=(-1,1), save=1, cmap=cmap)
ztf.correlate_offset_sns(sdss, 'r', 'mean_ri', pop, 1e1,3e3, xrange=xrange, yrange=(-1,1), save=1, cmap=cmap)
ztf.correlate_offset_sns(sdss, 'i', 'mean_iz', pop, 1e1,3e3, xrange=xrange, yrange=(-1,1), save=1, cmap=cmap)
