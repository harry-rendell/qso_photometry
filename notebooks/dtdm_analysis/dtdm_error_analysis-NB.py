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

# +
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

# import pandas as pd 
# import numpy as np
# from astropy.table import Table
# import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('font', **{'size':16})
# import matplotlib.cm as cmap
# from multiprocessing import Pool
# # from profilestats import profile
# from time import time
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# pd.options.mode.chained_assignment = None
# from module.preprocessing.dtdm import dtdm_raw_analysis
# -

# # Splitting by property

dtdm_qsos_lbol = dtdm_raw_analysis('qsos', 'uid', 'r', 'qsos')
# dtdm_qsos_lbol.calculate_stats_looped_key(26, 'log', 'Lbol', save=True)
dtdm_qsos_lbol.read_pooled_stats('log', key='Lbol')
# dtdm_qsos_lbol.pooled_stats.keys()

# +
########try linear mjd spacing next
# -

dtdm_qsos_civ = dtdm_raw_analysis('qsos', 'uid', 'r', 'qsos')
# dtdm_qsos_civ.calculate_stats_looped_key(26, 'log2', 'MBH_CIV', save=True)
dtdm_qsos_civ.read_pooled_stats('log', key='MBH')
# dtdm_qsos_lbol.pooled_stats.keys()

dtdm_qsos_nedd = dtdm_raw_analysis('qsos', 'uid', 'r', 'qsos')
# dtdm_qsos_nedd.calculate_stats_looped_key(26, 'log', 'nEdd', save=True)
dtdm_qsos_nedd.read_pooled_stats('log', key='nEdd')
# dtdm_qsos_lbol.pooled_stats.keys()

# +
# To find max dt
# dtdm_qsos_civ.read_key('MBH_CIV')
# uids = pd.concat(dtdm_qsos_civ.groups).index

# dts = []
# for i in range(26):
#     dtdm_qsos_civ.read(i)
#     dt = dtdm_qsos_civ.df[dtdm_qsos_civ.df.index.isin(uids)]['dt'].max()
#     dts.append(dt)

# +
# fig, ax = dtdm_qsos_nedd.plot_stats_property(['SF cwf a'], figax=None, xscale='log', yscale='linear', ylim=(0, 0.45), ylabel='Structure Function$^2$')
# fig.savefig('plots/qso_nedd_lin.pdf', bbox_inches='tight')
# -

dtdm_qsos_lbol.pooled_stats.keys()

fig, ax = dtdm_qsos_civ.plot_stats_property(['SF cwf a'], figax=None, xscale='log', yscale='log', xlim=(0,20000), ylim=(1e-2, 1e0 ), ylabel='Structure Function$^2$')
# fig.savefig('plots/qso_mbh_log.pdf', bbox_inches='tight')

fig, ax = dtdm_qsos_lbol.plot_stats_property(['SF cwf a'], figax=None, xscale='log', yscale='linear', ylim=(0, 1), ylabel='Structure Function$^2$')
# fig.savefig('plots/qso_lbol_lin.pdf', bbox_inches='tight')

# # Skewness

fig, ax = dtdm_qsos_civ.plot_stats_property(['skewness'], figax=None, xscale='log', yscale='linear', xlim=(0,10000), ylim=(-1, 1 ), ylabel='Skewness')

fig, ax = dtdm_qsos_civ.plot_stats_property(['kurtosis'], figax=None, xscale='log', yscale='linear', xlim=(0,10000), ylim=(-2,20), ylabel='Skewness')

# ### Drift

# +
dtdm_qsos_civ = dtdm_raw_analysis('qsos', 'uid', 'r', 'qsos')
dtdm_qsos_civ.read_pooled_stats('log_inner', key=None)

dtdm_star = dtdm_raw_analysis('calibStars', 'uid_s', 'r', 'stars')
dtdm_star.read_pooled_stats('log_inner')

print(dtdm_star.pooled_stats.keys())
print(dtdm_qsos_civ.pooled_stats.keys())

# -

fig, ax = dtdm_qsos_civ.plot_stats(['mean weighted a'], figax=None, xscale='log', yscale='linear', xlim=(0,10000), ylim=(-0.01, 0.18), ylabel='Drift (magnitude)')
dtdm_star.plot_stats(['mean weighted a'], figax=(fig,ax), color='r')
# fig.savefig('plots/qso_mbh_log.pdf', bbox_inches='tight')

# # Entire population

names = ['n', 'mean weighted a', 'mean weighted b', 'SF cwf a', 'SF cwf b', 'SF cwf c']

# ## Stars

# Initialise 
dtdm_star = dtdm_raw_analysis('calibStars', 'uid_s', 'r', 'stars')

# +
# # max dts within a survey
# dts = []
# for i in range(50):
#     dtdm_star.read(i)
#     dt = (dtdm_star.df['dt'][np.sqrt(dtdm_star.df['cat'])%1==0]).max()
#     dts.append(dt)
# max(dts)

# +
# Compute
# dtdm_star.calculate_stats_looped(46, 'log_inner', save=True, inner=True, max_t=4027)
# -

# Read
dtdm_star.read_pooled_stats('log_inner')
dtdm_star.pooled_stats.keys()

# +
fig, ax = dtdm_star.plot_stats(['SF cwf p'], figax=None, xscale='log', yscale='linear', ylim=(-0.01, 0.21), xlim=(0.9,3e4), ylabel='Structure Function (mag)', color='r')
dtdm_star.plot_stats(['SF cwf n'], figax=(fig,ax), xscale='log', yscale='linear', color='b')

# fig.savefig('plots/SF_asym.pdf', bbox_inches='tight')
# -

# ## Qsos

# Initialise
dtdm_qsos = dtdm_raw_analysis('qsos', 'uid', 'r', 'qsos')

# +
# dts = []
# for i in range(26):
#     dtdm_qsos.read(i)
#     dt = (dtdm_qsos.df['dt'][np.sqrt(dtdm_qsos.df['cat'])%1==0]).max()
#     dts.append(dt)
# max(dts)

# + active=""
# # Compute
# dtdm_qsos.calculate_stats_looped(26, 'log_inner', max_t=3467, save=True, inner=True)
# -

# Reading
dtdm_qsos.read_pooled_stats('log')

dtdm_qsos.pooled_stats.keys()

# # Moments

# mean weighted a and b are the same except in their errors. a is using 1/errors^2 and b uses var(dm)
fig, ax = dtdm_qsos.plot_stats(['mean weighted b'], figax=None, xscale='log', yscale='linear', ylim=(-0.4, 0.4), xlim=(0.9,3e4), ylabel='Mean', color='r')
ax.legend()


fig, ax = dtdm_qsos.plot_stats(['mean weighted b'], figax=None, xscale='log', yscale='linear', ylim=(-0.4, 0.4), xlim=(1084.05, 3650), ylabel='Mean', color='r')


fig, ax = dtdm_qsos.plot_stats(['skewness'], figax=None, xscale='log', yscale='linear', ylim=(-0.5, 0.5), xlim=(0.9,3e4), ylabel='Skewness', color='r')


fig, ax = dtdm_qsos.plot_stats(['kurtosis'], figax=None, xscale='log', yscale='linear', ylim=(-1, 6.5), xlim=(0.9,3e4), ylabel='Excess kurtosis', color='r')


# # Asymmetry

# +
fig, ax = dtdm_qsos.plot_stats(['SF cwf p'], figax=None, xscale='log', yscale='linear', ylim=(-0.01, 0.6), xlim=(0.9,3e4), ylabel='Structure Function (mag)', macleod=True, fit=True, color='r')
dtdm_qsos.plot_stats(['SF cwf n'], figax=(fig,ax), color='b')

# dtdm_star.plot_stats(['SF cwf p'], figax=(fig,ax), color='r')
# dtdm_star.plot_stats(['SF cwf n'], figax=(fig,ax), color='b')


# fig.savefig('plots/SF_asym.pdf', bbox_inches='tight')

# +
fig, ax = dtdm_qsos.plot_stats(['SF cwf p'], figax=None, xscale='log', yscale='linear', ylim=(-0.01, 0.6), xlim=(0.9,3e4), ylabel='Structure Function (mag)', macleod=True, fit=True, color='r')
dtdm_qsos.plot_stats(['SF cwf n'], figax=(fig,ax), xscale='log', yscale='linear', color='b', label='john')

# fig.savefig('plots/SF_asym.pdf', bbox_inches='tight')

# + active=""
# # Saving 
# np.savetxt('Ensemble_structure_function_qsos.csv',np.vstack([dtdm_qsos.mjd_centres,dtdm_qsos.pooled_stats['SF a'].T**0.5]).T,fmt='%.8e',delimiter=',')

# +
# keys = sorted([key for key in dtdm_qsos_log.pooled_stats.keys() if key.startswith('SF')])
# n = len(keys)
# # fig, ax = plt.subplots(n, 1, figsize=(18,5*n))
# for i, key in enumerate(keys):
# #     ax[i].hist(dtdm_qsos_log.pooled_stats[key][:,1], label=key)
# #     ax[i].legend()
#     print(key,'\t\t\t',dtdm_qsos_log.pooled_stats[key][:,1].mean())
# -

# Squareroot our SF, since we save SF^2
for dtdm_survey in [dtdm_qsos]:
    for name in ['SF cwf a','SF cwf b', 'SF cwf n', 'SF cwf p']:
        dtdm_survey.pooled_stats[name][:,1] = dtdm_survey.pooled_stats[name][:,1]**2
        dtdm_survey.pooled_stats[name][np.isnan(dtdm_survey.pooled_stats[name])]=0

dtdm_star.read(4)

dtdm_star.plot_dm2_de2_hist(figax=None, bins=100, xlim=[-1,1])

keys = ['SF cwf a']
fig, ax = dtdm_qsos.plot_stats(keys, None, xscale='log', yscale='linear', ylim=(-0.01, 0.6), xlim=(0.9,3e4), ylabel='Structure Function (mag)', macleod=True, fit=True, color='r')
fig, ax = dtdm_star.plot_stats(keys, (fig,ax), color='g')
# fig, ax = dtdm_star.plot_stats(['SF cwf c'], (fig,ax), color='r')
# fig, ax = dtdm_star2.plot_stats(['SF corrected weighted fixed'], (fig,ax), color='b')
# dtdm_star_log3.plot_stats(keys, (fig,ax))
ax.axhline(y=0, color='k', ls='--')
# fig.savefig('plots/sf_largeerrors_corrected_plate_errors.pdf', bbox_inches='tight')

dtdm_star_log = dtdm_raw_analysis('calibStars', 'uid_s', 'r', 'name1')
dtdm_star_log.read_pooled_stats(log_or_lin='log')
dtdm_star_log2 = dtdm_raw_analysis('calibStars', 'uid_s', 'r', 'name2')
dtdm_star_log2.read_pooled_stats(log_or_lin='log2')
dtdm_star_log3 = dtdm_raw_analysis('calibStars', 'uid_s', 'r', 'name3')
dtdm_star_log3.read_pooled_stats(log_or_lin='log3')
dtdm_qsos_log = dtdm_raw_analysis('qsos', 'uid', 'r', 'name4')
dtdm_qsos_log.read_pooled_stats(log_or_lin='log')
# dtdm_qsos_log2 = dtdm_raw_analysis('qsos', 'uid', 'r')
# dtdm_qsos_log2.read_pooled_stats(log_or_lin='log2')
# dtdm_star_log3 = dtdm_raw_analysis('calibStars', 'uid_s', 'r')
# dtdm_star_log3.read_pooled_stats(log_or_lin='log3')

dtdm_star_log.pooled_stats.keys()

dtdm_star_log.pooled_stats['n']

keys = ['SF cwf a']
fig, ax = dtdm_star_log2.plot_stats(keys,None, xscale='log', yscale='linear', ylim=(-0.01, 0.6), xlim=(0.9,3e4), ylabel='Structure Function$^2$', macleod=True)
fig, ax = dtdm_star_log.plot_stats(keys, (fig,ax))
fig, ax = dtdm_qsos_log.plot_stats(keys, (fig,ax))
dtdm_star_log3.plot_stats(keys, (fig,ax))
# fig.savefig('plots/QSOS_SF_corrected_fixed_comparison_ylinear.pdf', bbox_inches='tight')

keys = ['SF corrected weighted fixed']
fig, ax = dtdm_qsos_log.plot_stats(keys,None, xscale='log', yscale='log', ylim=(3e-4, 0.4), ylabel='Structure Function$^2$', macleod=True)
fig, ax = dtdm_star_log.plot_stats(keys, (fig,ax))
# dtdm_star_log3.plot_stats(keys, (fig,ax))
# fig.savefig('plots/QSOS_SF_corrected_fixed_comparison_3.pdf', bbox_inches='tight')

keys = ['SF corrected weighted fixed']
fig, ax = dtdm_qsos_log.plot_stats(keys,None, xscale='log', yscale='log', ylim=(3e-4, 0.4), ylabel='Structure Function$^2$', macleod=True)
fig, ax = dtdm_star_log.plot_stats(keys, (fig,ax))
# dtdm_star_log3.plot_stats(keys, (fig,ax))
# fig.savefig('plots/QSOS_SF_corrected_fixed_comparison_ylinear.pdf', bbox_inches='tight')

dtdm_star_log3.pooled_stats['SF corrected weighted']

dtdm_qsos_log.read_key('Lbol')

dtdm_qsos_log.groups[-1].loc[20005]

dtdm_qsos_log.pooled_stats['SF 2'][:,1] /= 2

keys = ['SF corrected wighte', 'SF 2']
fig, ax = dtdm_qsos_log.plot_stats(keys,None, macleod=True, xscale='log', yscale='linear', ylim=(0, 1.1), ylabel='Structure Function$^2$')
dtdm_star_log.plot_stats(keys, (fig,ax))
# fig.savefig('plots/QSOS_SF_corrected_fixed_comparison.pdf', bbox_inches='tight')

keys = ['SF corrected weighted', 'SF corrected weighted fixed']
fig, ax = dtdm_qsos_log.plot_stats(keys,None, xscale='log', yscale='log', ylim=(3e-2, 1e0), ylabel='Structure Function$^2$', macleod=True)
dtdm_star_log.plot_stats(keys, (fig,ax))
fig.savefig('plots/QSOS_SF_corrected_fixed_comparison_ylinear.pdf', bbox_inches='tight')

keys = ['SF 1', 'SF 2', 'SF 3']
fig, ax = dtdm_qsos_log.plot_stats(keys,None, xscale='log', yscale='log', ylim=(3e-2, 1), ylabel='Structure Function$^2$')
# dtdm_star.plot_stats(keys, (fig,ax))
# fig.savefig('plots/QSOS_SF_corrected_comparison.pdf', bbox_inches='tight')

keys = ['SF', 'SF corrected']
fig, ax = dtdm_qsos.plot_stats(keys,None, xscale='log', yscale='log', ylim=(3e-2, 1), ylabel='Structure Function$^2$')
# dtdm_star.plot_stats(keys, (fig,ax))
fig.savefig('plots/QSOS_SF_corrected_comparison.pdf', bbox_inches='tight')

keys = ['SF corrected weighted']
fig, ax = dtdm_qsos_lin.plot_stats(keys,None, ylim=(3e-3, 1), ylabel='Structure Function$^2$', macleod=True)
dtdm_qsos_log.plot_stats(keys, (fig, ax))
dtdm_star_log.plot_stats(keys, (fig, ax))
# dtdm_star.plot_stats(keys, (fig,ax))
# fig.savefig('plots/QSOS_SF_corrected_comparison_lin.pdf', bbox_inches='tight')

# Note that SF is really SF^2, and error bars are variance not std
keys = ['SF corrected weighted', 'SF corrected']
fig, ax = dtdm_qsos.plot_stats(keys,None, xscale='log', yscale='log', ylabel='Structure Function$^2$', ylim=(2e-3, 7e-1))
dtdm_star.plot_stats(keys, (fig,ax))
fig.savefig('plots/QSOS_STARS_SF_corrected_weighted_comparison.pdf', bbox_inches='tight')

dtdm_star.plot_stats('all',None)

keys = ['mean weighted']
fig, ax = dtdm_qsos.plot_stats(keys,None,xscale='log',ylim=(-0.23,0.45))
dtdm_star.plot_stats(keys, (fig,ax), xscale='log')
ax.axhline(y=0, lw=0.5, color='k')

dtdm_star = dtdm_raw_analysis('calibStars', 'uid_s', 'r')
# dtdm_star.read(10)
