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

# ### Code below is for reading in + analysis

import pandas as pd 
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'size':18})
import matplotlib.cm as cmap
# %matplotlib inline
from multiprocessing import Pool
# from profilestats import profile
from scipy.stats import binned_statistic, skew, iqr
from scipy.optimize import curve_fit
from funcs.analysis.analysis import *
from funcs.preprocessing.binning import bin_data  
from os import listdir
import os
import time

from funcs.preprocessing.dtdm import dtdm, dtdm_key

# +
config = {'obj':'qsos','ID':'uid'  ,'t_max':23576,'n_bins_t':200,'n_bins_m':200,'n_t_chunk':19, 'width':2, 'steepness':0.005}

width   = config['width']
steepness = config['steepness']
obj = config['obj']
ID  = config['ID']
t_max = config['t_max']
n_bins_t = config['n_bins_t']
n_bins_m = config['n_bins_m']
n_t_chunk = config['n_t_chunk']

all_q = dtdm(obj, 'all', 'all qsos', 200, 200, t_max, n_t_chunk, steepness, width);
# sep = dtdm(obj, 'sep', 200, 200, t_max, n_t_chunk, 0.005, 1);s
sdss_sdss_q = dtdm(obj, 'sdss_sdss', 'sdss-sdss qsos', 200, 200, t_max, n_t_chunk, steepness, width); 
ps_ps_q = dtdm(obj, 'ps_ps', 'ps-ps qsos', 200, 200, t_max, n_t_chunk, steepness, width); 
ztf_ztf_q = dtdm(obj, 'ztf_ztf', 'ztf-ztf qsos', 200, 200, t_max, n_t_chunk, steepness, width);

sdss_ps_q = dtdm(obj, 'sdss_ps', 'sdss-ps qsos', 200, 200, t_max, n_t_chunk, steepness, width);
sdss_ztf_q = dtdm(obj, 'sdss_ztf', 'sdss-ztf qsos', 200, 200, t_max, n_t_chunk, steepness, width); 
ps_ztf_q = dtdm(obj, 'ps_ztf', 'ps-ztf qsos', 200, 200, t_max, n_t_chunk, steepness, width);

# survey_list_q = [all_q, sdss_ps_q, sdss_ztf_q, ps_ztf_q, sdss_sdss_q, ps_ps_q, ztf_ztf_q]
survey_list_q = [all_q]

# +
# If we are using a key
config = {'obj':'qsos','ID':'uid'  ,'t_max':23576,'n_bins_t':200,'n_bins_m':200, 'n_bins_m2':248, 'n_t_chunk':19, 'width':2, 'steepness':0.005, 'leftmost_bin':-0.244}

width   = config['width']
steepness = config['steepness']
obj = config['obj']
ID  = config['ID']
t_max = config['t_max']
n_bins_t = config['n_bins_t']
n_bins_m = config['n_bins_m']
n_bins_m2 = config['n_bins_m2']
n_t_chunk = config['n_t_chunk']
leftmost_bin = config['leftmost_bin']
all_q = dtdm_key(obj, 'all', 'all qsos', 'Lbol', n_bins_t, n_bins_m, n_bins_m2, t_max, n_t_chunk, steepness, width, leftmost_bin);
survey_list_q = [all_q]


# +
config = {'obj':'calibStars','ID':'uid_s','t_max':7772,'n_bins_t':200,'n_bins_m':200,'n_t_chunk':19, 'width':1, 'steepness':0.005}

width   = config['width']
steepness = config['steepness']
obj = config['obj']
ID  = config['ID']
t_max = config['t_max']
n_bins_t = config['n_bins_t']
n_bins_m = config['n_bins_m']
n_t_chunk = config['n_t_chunk']

all_s = dtdm(obj, 'all', 'all stars', 200, 200, t_max, n_t_chunk, steepness, width);
# sep = dtdm(obj, 'sep', 200, 200, t_max, n_t_chunk, 0.005, 1);
sdss_sdss_s = dtdm(obj, 'sdss_sdss' ,'sdss-sdss stars', 200, 200, t_max, n_t_chunk, steepness, width); 
ps_ps_s = dtdm(obj, 'ps_ps', 'ps-ps stars', 200, 200, t_max, n_t_chunk, steepness, width); 
ztf_ztf_s = dtdm(obj, 'ztf_ztf', 'ztf-ztf stars', 200, 200, t_max, n_t_chunk, steepness, width);

sdss_ps_s = dtdm(obj, 'sdss_ps', 'sdss-ps stars', 200, 200, t_max, n_t_chunk, steepness, width);
sdss_ztf_s = dtdm(obj, 'sdss_ztf', 'sdss-ztf stars', 200, 200, t_max, n_t_chunk, steepness, width); 
ps_ztf_s = dtdm(obj, 'ps_ztf', 'ps-ztf stars', 200, 200, t_max, n_t_chunk, steepness, width);

survey_list_s = [all_s, sdss_ps_s, sdss_ztf_s, ps_ztf_s, sdss_sdss_s, ps_ps_s, ztf_ztf_s]
# -

print('qsos:')
all_q.stats(verbose=True)
# print('stars:')
# all_s.stats(verbose=True)

fig, ax = plt.subplots(1,1, figsize=(20,10))
for i,surv in enumerate(survey_list_q):
    figax, SF = surv.plot_sf_dm2_de2(figax=(fig,ax))
ax.legend()

all_q.dm2_de2_binned

fig, ax = plt.subplots(1,1, figsize=(20,10))
# for surv in survey_list_s[4:5]:
#     ax, SF = surv.plot_sf_ensemble_asym(ax=ax)
for i,surv in enumerate(survey_list_q):
    figax, SF_n, SF_p = surv.plot_sf_ensemble_asym(figax=(fig,ax), color='brgkcmy'[i])
ax.legend()
# fig.savefig('')

# +
# #Testing IQR with brute force
# full_dist_list = []
# for d, bins in zip(dms_binned,all_.m_bin_centres):
#     full_dist_list.append(np.full(shape=d, fill_value=bins))
# full_dist = np.concatenate(full_dist_list)
# iqr(full_dist)
# -

# This should be the value we get for bin 0


fig, ax = plt.subplots(1,1, figsize=(15,8))
for surv in survey_list_s[:1]:
    ax, SF = surv.plot_sf_ensemble_iqr(ax=ax)
for surv in survey_list_q[:1]:
    ax, SF = surv.plot_sf_ensemble_iqr(ax=ax)
ax.legend()
ax.set(title='Structure function using IQR method')
# fig.savefig('/disk1/hrb/python/analysis/{}/plots/{}_SF.jpg'.format(surv.obj,surv.obj), dpi=300, bbox_inches='tight')

all_q.hist_dm(2)

# +
# QSOS
fig, ax = plt.subplots(1,1, figsize=(15,8))
# for surv in survey_list_s[4:5]:
#     ax, SF = surv.plot_sf_ensemble(ax=ax)
for surv in survey_list_q[:1]:
    (fig,ax), SF = surv.plot_sf_ensemble(figax=(fig,ax))
ax.legend()
ax.set(ylim = [0.1,1], yticks=[0.1,0.5,0.4,0.7,1])
# ax.get_yaxis().set_major_formatter()
def power_law(x,a,b):
    return b * x ** a

from scipy.optimize import curve_fit

popt, pcov = curve_fit(power_law, all_q.t_bin_chunk_centres[3:15], SF[3:15])

x = np.logspace(1,4.5, 19)
y = power_law(x, *popt)
ax.text(0.04, 0.92, 'slope: {:.2f}'.format(popt[0]), transform=ax.transAxes)
ax.plot(x,y, lw=0.8)

print('best fit by power law with exponent: {}'.format(popt[0]))
fig.savefig('/disk1/hrb/python/analysis/{}/plots/{}_SF_fit.jpg'.format(surv.obj,surv.obj), dpi=300, bbox_inches='tight')
# -

# QSOS
fig, ax = plt.subplots(1,1, figsize=(23,10))
# for surv in survey_list_s[:1]:
#     surv.plot_means(ax, ls='-')
for surv in survey_list_q[:1]:
    surv.plot_means(ax, ls='-')
ax.legend()
ax.set(xlabel='∆t (days)', ylabel='∆m (mags)', ylim=[-0.15,0.3])
ax.axhline(y=0, lw=1, ls='--', color='k')
fig.savefig('/disk1/hrb/python/analysis/{}/plots/{}_drift_mean.jpg'.format(obj,obj), dpi=300, bbox_inches='tight')

# QSOS
fig, ax = plt.subplots(1,1, figsize=(23,10))
for surv in survey_list_q[:1]:
    surv.plot_modes(ax, ls='-')
# for surv in survey_list_s[:1]:
#     surv.plot_modes(ax, ls='-')
ax.legend()
ax.set(xlabel='∆t (days)', ylabel='∆m (mags)', ylim=[-0.15,0.05])
ax.axhline(y=0, lw=1, ls='--', color='k')
fig.savefig('/disk1/hrb/python/analysis/{}/plots/{}_drift_mode.jpg'.format(obj,obj), dpi=300, bbox_inches='tight')

# +
# for a in obj_list[:1]:
#     a.hist_dm(window_width=0.1, overlay_gaussian=True, overlay_lorentzian=False, save=False)

# +
# redefining dcat, because ZTF-PS is equivalent to PS-ZTF
# for a,b in zip([1,2,5],[3,6,7]):
#     dcat_result[:,a] = dcat_result[:,a] + dcat_result[:,b]
# dcat_result = dcat_result[:,(0,4,8,1,2,5)]

# +
# fig, axes = plt.subplots(2,3, figsize = (16,8))
# labels    = ['SDSS-SDSS', 'PS-PS', 'ZTF-ZTF', 'SDSS-PS', 'SDSS-ZTF', 'PS-ZTF']
# for i, ax in enumerate(axes.ravel()):
#     ax.hist(t_bin_chunk[:-1], weights = dcat_result[:,i], alpha=0.5, label=labels[i], edgecolor='black', lw=1.2)
#     ax.set(yscale='log', xlabel='∆t (days)')
#     ax.legend()   
# fig.savefig('dt_distribution.pdf', bbox_inches='tight')

# +
# Asymmetry of SF
# fig, ax = plt.subplots(1,1,figsize = (16,8))
# plot_sf_ensemble(dts_binned_tot_sep[:,:100], dms_binned_tot_sep[:,:100], ax)
# plot_sf_ensemble(dts_binned_tot_sep[:,100:], dms_binned_tot_sep[:,100:], ax)

# +
# splitting up by redshift
# key = 'redshift' 
# bounds, z_score, bounds_values, ax = dr.bounds(key, bounds = np.array([-5,-1,-0.5,0,0.5,1,5]))
# uids = dr.properties['mjd_ptp_rf'][(bounds[i]<z_score)&(z_score<bounds[i+1])].sort_values(ascending=False).head(100000).index

# +
config = {'obj':'qsos','ID':'uid','t_max':6751,'n_bins_t':200,'n_bins_m':200,'n_t_chunk':19, 'width':2, 'steepness':0.005}
# config = {'obj':'calibStars','ID':'uid_s','t_max':7772,'n_bins_t':200,'n_bins_m':200,'n_t_chunk':19, 'width':1, 'steepness':0.005}

width   = config['width']
steepness = config['steepness']
obj = config['obj']
ID  = config['ID']
t_max = config['t_max']
n_bins_t = config['n_bins_t']
n_bins_m = config['n_bins_m']
n_t_chunk = config['n_t_chunk']
key = 'Lbol'

all_q = dtdm_key(obj, 'all', 'all qsos', key, 200, 200, t_max, n_t_chunk, steepness, width);
# -

all_q.stats()

all_q.means[0]

fig, ax = all_q.plot_sf_ensemble()
fig.savefig('/disk1/hrb/python/analysis/{}/plots/SF_{}_{}.jpg'.format(obj,obj,key), bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(15,10))
all_q.plot_means(ax)
ax.legend()

fig, ax = plt.subplots(1,1, figsize=(15,10))
all_q.plot_modes(ax)
ax.legend()
