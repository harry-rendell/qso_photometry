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

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.plotting.common import savefigs
from module.preprocessing import parse, color_transform
from module.assets import load_vac
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

obj = 'qsos'
ID = 'uid' if obj == 'qsos' else 'uid_s'
band = 'r'

# # Check correlations of skewfits

# +
# Load skewfit data
vac = load_vac('qsos', usecols=['z','Lbol'])
skewfits = []
for band in 'gri':
    s = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/obs/{band}_ssa_sdss_ps_ztf_30.csv", index_col=ID)
    s['band'] = band
    vac['wavelength'] = color_transform.calculate_wavelength(band, vac['z'])
    s = s.join(vac, on=ID)
    skewfits.append(s)
skewfits = pd.concat(skewfits).dropna().sort_index()
skewfits = parse.filter_data(skewfits, bounds={'a':(0,0.01),'loc':(2,5),'scale':(0.1,1), 'z':(0.2,5), 'tau16':(0,5), 'tau50':(1,5), 'tau84':(1,6)}, verbose=True)

# Plot skewfit correlations
for band in 'gri':
    fig, ax = plt.subplots(figsize=(13,10))
    # sns.heatmap(skewfits[skewfits['band']==band].drop(columns='band'))
    sns.heatmap(skewfits[skewfits['band']==band].drop(columns='band').corr()*100, annot=True, ax=ax, fmt='.1f', cmap='coolwarm', cbar_kws={'label': 'Correlation (%)'})
# -

# # Correlations of SF_per_qso with VAC
#

# +
n_bins = 10
bands = 'gri'

vac = load_vac('qsos', usecols=['z','Lbol','MBH','nEdd'])
sf = []
for band in bands:
    s = pd.read_csv(cfg.D_DIR + f'computed/{obj}/features/{band}/SF_{n_bins}_bins.csv', index_col=ID)
    skewfits = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/obs/{band}_ssa_sdss_ps_ztf_30.csv", index_col=ID)
    skewfits = parse.filter_data(skewfits, bounds={'a':(0,0.01),'loc':(2,5),'scale':(0.1,1), 'z':(0.2,5), 'tau16':(0,5), 'tau50':(1,5), 'tau84':(1,6)}, verbose=True)
    s = s.join(skewfits, on=ID, how='inner')
    s['band'] = band
    vac['wavelength'] = color_transform.calculate_wavelength(band, vac['z'])
    s = s.join(vac, on=ID, how='left')
    sf.append(s)
sf = pd.concat(sf).sort_index()
# -

# # Correlations of grouped

# Plot skewfit correlations
coi = [col for col in sf.columns if (col.startswith('SF2_w') or ((not col.startswith('SF2_cw')) and (not col.startswith('n_') and not col.startswith('dm_var_'))))]
for band in 'r':
    fig, ax = plt.subplots(figsize=(19,15))
    # sns.heatmap(skewfits[skewfits['band']==band].drop(columns='band'))
    sns.heatmap((sf[sf['band']==band][coi].drop(columns='band')).corr()*100, annot=True, ax=ax, fmt='.1f', cmap='coolwarm', cbar_kws={'label': 'Correlation (%)'})

# # Interpretation from plot above:
# Note that we can plot the correlation of many quantities with SF as a function of time!
# - SF decorrelates with time, could plot this as a function of time lag (kinda like an ACF)
# - Lbol most strongly correlates with 5<∆t<10 then 73<∆t<190. Correlation decreases with timelag
# - 

# # could we create the plots below for all correlation pairs with |r| > 50% excluding trivial pairs

fig, ax = plt.subplots(1,2, figsize=(20,10))
bins=20
threshold=1
x = 'sig16'
y = 'SF2_w_3576_10045'
sns.histplot(data=sf[[x,y]].dropna(), x=x, y=y, bins=bins, cmap='Spectral_r', thresh=threshold, ax=ax[0], log_scale=(False, True))
sns.scatterplot(data=sf[[x,y]].dropna(), x=x, y=y, ax=ax[1])
ax[1].set(yscale='log')
# sns.histplot(data=sf, x=x, y=y, bins=bins, cmap='Spectral_r', thresh=threshold, ax=ax)
