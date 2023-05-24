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

# ### Code below is for reading in + analysis

import pandas as pd 
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
font = {'size' : 18}
matplotlib.rc('font', **font)
# %matplotlib inline
from multiprocessing import Pool
# from profilestats import profile
from scipy.stats import binned_statistic, skew, iqr
from scipy.optimize import curve_fit
from module.analysis.analysis import *
from module.preprocessing.binning import bin_data
from os import listdir
import os
import time

from module.preprocessing.dtdm import dtdm, dtdm_key

# +
config = {'obj':'qsos','ID':'uid','t_max':23576,'n_bins_t':200,'n_bins_m':200, 'n_bins_m2':248, 
          'n_t_chunk':20, 'width':2, 'steepness':0.005, 'leftmost_bin':-0.244, 'subset':'all'}

# Must ensure that this list is exactly the same as the keys in the config dictionary above
obj, ID, t_max, n_bins_t, n_bins_m, n_bins_m2, n_t_chunk, width, steepness, leftmost_bin, subset = config.values()

# all_q = dtdm_key(obj, 'all', 'all qsos', 'Lbol', 200, 200, t_max, n_t_chunk, steepness, width);
all_q = dtdm(obj, 'all', 'all qsos', n_bins_t, n_bins_m, n_bins_m2, t_max, n_t_chunk, steepness, width, leftmost_bin, subset, True);
# sep = dtdm(obj, 'sep', 200, 200, t_max, n_t_chunk, 0.005, 1);

# ssa_sdss_q = dtdm(obj, 'ssa_sdss', 'ssa-sdss qsos', 200, 200, t_max, n_t_chunk, steepness, width); 
# ssa_ps_q = dtdm(obj, 'ssa_ps', 'ssa-ps qsos', 200, 200, t_max, n_t_chunk, steepness, width); 
# ssa_ztf_q = dtdm(obj, 'ssa_ztf', 'ssa-ztf qsos', 200, 200, t_max, n_t_chunk, steepness, width);

# sdss_sdss_q = dtdm(obj, 'sdss_sdss', 'sdss-sdss qsos', 200, 200, t_max, n_t_chunk, steepness, width); 
# ps_ps_q = dtdm(obj, 'ps_ps', 'ps-ps qsos', 200, 200, t_max, n_t_chunk, steepness, width); 
# ztf_ztf_q = dtdm(obj, 'ztf_ztf', 'ztf-ztf qsos', 200, 200, t_max, n_t_chunk, steepness, width);

# sdss_ps_q = dtdm(obj, 'sdss_ps', 'sdss-ps qsos', 200, 200, t_max, n_t_chunk, steepness, width);
# sdss_ztf_q = dtdm(obj, 'sdss_ztf', 'sdss-ztf qsos', 200, 200, t_max, n_t_chunk, steepness, width); 
# ps_ztf_q = dtdm(obj, 'ps_ztf', 'ps-ztf qsos', 200, 200, t_max, n_t_chunk, steepness, width);

# survey_list_q = [all_q, sdss_ps_q, sdss_ztf_q, ps_ztf_q, sdss_sdss_q, ps_ps_q, ztf_ztf_q]
survey_list_q = [all_q]

# +
config = {'obj':'calibStars','ID':'uid_s','t_max':25200,'n_bins_t':200,'n_bins_m':200, 'n_bins_m2':235,
          'n_t_chunk':20, 'width':1, 'steepness':0.005, 'leftmost_bin':-0.21, 'subset':'all'}

# Must ensure that this list is exactly the same as the keys in the config dictionary above
obj, ID, t_max, n_bins_t, n_bins_m, n_bins_m2, n_t_chunk, width, steepness, leftmost_bin, subset = config.values()

all_s = dtdm(obj, 'all', 'all stars', n_bins_t, n_bins_m, n_bins_m2, t_max, n_t_chunk, steepness, width, leftmost_bin, subset, True);
# sep = dtdm(obj, 'sep', 200, 200, t_max, n_t_chunk, 0.005, 1);
# sdss_sdss_s = dtdm(obj, 'sdss_sdss' ,'sdss-sdss stars', 200, 200, t_max, n_t_chunk, steepness, width, subset=subset); 
# ps_ps_s = dtdm(obj, 'ps_ps', 'ps-ps stars', 200, 200, t_max, n_t_chunk, steepness, width, subset=subset); 
# ztf_ztf_s = dtdm(obj, 'ztf_ztf', 'ztf-ztf stars', 200, 200, t_max, n_t_chunk, steepness, width, subset=subset);

# sdss_ps_s = dtdm(obj, 'sdss_ps', 'sdss-ps stars', 200, 200, t_max, n_t_chunk, steepness, width, subset=subset);
# sdss_ztf_s = dtdm(obj, 'sdss_ztf', 'sdss-ztf stars', 200, 200, t_max, n_t_chunk, steepness, width, subset=subset); 
# ps_ztf_s = dtdm(obj, 'ps_ztf', 'ps-ztf stars', 200, 200, t_max, n_t_chunk, steepness, width, subset=subset);

survey_list_s = [all_s]#, sdss_ps_s, sdss_ztf_s, ps_ztf_s, sdss_sdss_s, ps_ps_s, ztf_ztf_s]
# -

# # ∆σ analysis

fig, ax, r2s = all_q.hist_dm(1.5, overlay_gaussian=True, overlay_lorentzian=True, overlay_exponential=False, overlay_diff=False, colors=['royalblue','k','k','red'], save=False)

fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.plot(r2s[0], label='gaussian')
ax.plot(r2s[1], label='exponential')
ax.legend()

fig, ax, r2s = all_s.hist_dm(1, overlay_gaussian=True, overlay_lorentzian=False, overlay_exponential=True, overlay_diff=False, colors=['orangered','k','k','royalblue'], save=False)


fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.plot(r2s[0], label='gaussian')
ax.plot(r2s[1], label='exponential')
ax.legend()

self = all_q
i = 0
n=1
cmap = plt.cm.cool
m,_,_= ax.hist(self.m_bin_edges[:-1], self.m_bin_edges[::n], weights = self.dms_binned[i], alpha = 1, density=True, label = self.t_dict[i], color = cmap(i/20.0));
plt.plot(self.m_bin_centres,m)

# + active=""
# lin_bins = np.linspace(self.m_bin_edges[10],self.m_bin_edges[-10], 101)
# print(lin_bins)
# digitized = np.digitize(self.m_bin_centres, lin_bins)
# bin_counts = [m[digitized == i].sum() for i in range(1, len(lin_bins))]\
# plt.plot(bin_counts)

# + active=""
# fig, ax = sdss_ps_s.hist_dm(0.8, False, False, False)
# -

all_s.hist_de(0.5, False, False, False)

# + jupyter={"outputs_hidden": true}
sdss_ps_s.hist_de(0.5, False, False, False)

# + jupyter={"outputs_hidden": true}
ztf_ztf_s.hist_de(0.5, False, False, False)
# -

sdss_sdss_q.hist_de(0.5, False, False, False)

# Histogram showing the number of bin counts for (∆m, ∆t) pairs in different time intervals ∆t
all_q.hist_dt_stacked(False, False, False)

print('qsos:')
all_q.stats(verbose=True)
print('stars:')
all_s.stats(verbose=True)

fig, ax = plt.subplots(1,1, figsize=(15,8))
# for surv in survey_list_s[4:5]:
#     ax, SF = surv.plot_sf_ensemble_asym(ax=ax)
for i,surv in enumerate(survey_list_q):
    (fig,ax), SF_n, SF_p = surv.plot_sf_ensemble_asym(figax=(fig,ax), color='brgkcmy'[i])
ax.legend()

# +
# #Testing IQR with brute force
# full_dist_list = []
# for d, bins in zip(dms_binned,all_.m_bin_centres):
#     full_dist_list.append(np.full(shape=d, fill_value=bins))
# full_dist = np.concatenate(full_dist_list)
# iqr(full_dist)
# -

fig, ax = plt.subplots(1,1, figsize=(15,8))
for surv in survey_list_s[:1]:
    ax, SF = surv.plot_sf_ensemble_iqr(ax, xscale='log', yscale='linear')
for surv in survey_list_q[:1]:
    ax, SF = surv.plot_sf_ensemble_iqr(ax, xscale='log', yscale='linear')
ax.legend()
ax.set(title='Structure function using IQR method')
# fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_SF.jpg'.format(surv.obj,surv.obj), dpi=300, bbox_inches='tight')

# +
fig, ax = plt.subplots(1,1, figsize=(15,8))
for surv in survey_list_q[:1]:
    (fig,ax), SF = surv.plot_sf_ensemble(figax=(fig,ax))

for surv in survey_list_q[:1]:
    (fig,ax), SF = surv.plot_sf_ensemble_corrected(figax=(fig,ax))

# +
# QSOS
fig, ax = plt.subplots(1,1, figsize=(15,8))
# for surv in survey_list_s[4:5]:
#     ax, SF = surv.plot_sf_ensemble(ax=ax)
for surv in survey_list_q[:1]:
    (fig,ax), SF = surv.plot_sf_ensemble(figax=(fig,ax))
for surv in survey_list_s[:1]:
    (fig,ax), SF = surv.plot_sf_ensemble(figax=(fig,ax))
ax.legend()
ax.set(ylim = [0.01,1], yticks=[0.1,0.5,0.4,0.7,1])
# ax.get_yaxis().set_major_formatter()
def power_law(x,a,b):
    return b * x ** a

from scipy.optimize import curve_fit

popt, pcov = curve_fit(power_law, all_q.t_bin_chunk_centres[3:15], SF[3:15])

x = np.logspace(1,4, 19)
y = power_law(x, *popt)
ax.text(0.04, 0.92, 'slope: {:.2f}'.format(popt[0]), transform=ax.transAxes)
# ax.plot(x,y)

print('best fit by power law with exponent: {}'.format(popt[0]))
# fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_SF_fit.jpg'.format(surv.obj,surv.obj), dpi=300, bbox_inches='tight')
# -

# QSOS
fig, ax = plt.subplots(1,1, figsize=(23,10))
for surv in survey_list_s[:1]:
    surv.plot_means(ax, ls='-')
for surv in survey_list_q[:1]:
    surv.plot_means(ax, ls='-')
ax.legend()
ax.set(xlabel='∆t (days)', ylabel='∆m (mags)', ylim=[-0.15,0.3])
ax.axhline(y=0, lw=1, ls='--', color='k')
# fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_drift_mean.jpg'.format(obj,obj), dpi=300, bbox_inches='tight')

# QSOS
fig, ax = plt.subplots(1,1, figsize=(23,10))
for surv in survey_list_q[:1]:
    surv.plot_modes(ax, ls='-')
for surv in survey_list_s[:1]:
    surv.plot_modes(ax, ls='-')
ax.legend()
ax.set(xlabel='∆t (days)', ylabel='∆m (mags)', ylim=[-0.15,0.1])
ax.axhline(y=0, lw=1, ls='--', color='k')
# fig.savefig(cfg.W_DIR + 'analysis/{}/plots/{}_drift_mode.jpg'.format(obj,obj), dpi=300, bbox_inches='tight')

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
config = {'obj':'qsos','ID':'uid','t_max':6751,'n_bins_t':200,'n_bins_m':200, 'n_bins_m2':235,'n_t_chunk':19, 'width':2, 'steepness':0.005, 'leftmost_bin':-0.21}
# config = {'obj':'calibStars','ID':'uid_s','t_max':7772,'n_bins_t':200,'n_bins_m':200,'n_t_chunk':19, 'width':1, 'steepness':0.005, 'leftmost_bin':-0.21}
# config = {'obj':'calibStars','ID':'uid_s','t_max':25200,'n_bins_t':200,'n_bins_m':200, 'n_bins_m2':235,'n_t_chunk':20, 'width':1, 'steepness':0.005, 'leftmost_bin':-0.21}

width   = config['width']
steepness = config['steepness']
obj = config['obj']
ID  = config['ID']
t_max = config['t_max']
n_bins_t = config['n_bins_t']
n_bins_m = config['n_bins_m']
n_bins_m2 = config['n_bins_m2']
n_t_chunk = config['n_t_chunk']
key = 'Lbol'
leftmost_bin = config['leftmost_bin']

all_q = dtdm_key(obj, 'all', 'all qsos', key, 200, 200, n_bins_m2, t_max, n_t_chunk, steepness, width, leftmost_bin);
# -

all_q.stats()

all_q.means[0]

fig, ax = all_q.plot_sf_ensemble()
# fig.savefig(cfg.W_DIR + 'analysis/{}/plots/SF_{}_{}.jpg'.format(obj,obj,key), bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(15,10))
all_q.plot_means(ax)
ax.legend()

fig, ax = plt.subplots(1,1, figsize=(15,10))
all_q.plot_modes(ax)
ax.legend()
