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
# jupytext --to py calculate_ssa_errors.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb calculate_ssa_errors.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.classes.dtdm import dtdm_raw_analysis
from module.preprocessing import data_io
from module.assets import load_grouped_tot

OBJ = 'calibStars'
band = 'r'
ID = 'uid' if OBJ == 'qsos' else 'uid_s'

a = dtdm_raw_analysis(OBJ, band, 'ssa_ps', phot_str='dsid_21_49')
a.read_all()

mag_med = load_grouped_tot(OBJ, band, usecols=['mag_med'])
df = a.df.join(mag_med, on=ID, how='inner').dropna()
ps_mask = df['dsid']==49
ssa_ps_mask = df['dsid']==21

fig, ax = plt.subplots(1,1, figsize=(15,10))
sns.histplot(data=a.df.sample(frac=0.2), x='dm', hue='dsid', bins=100, stat='density', ax=ax, binrange=(-1,1))

# Create masks so we are comparing the same objects in each survey
intersection = ssa_ps_mask.index[ssa_ps_mask.values].unique().intersection(ps_mask.index[ps_mask.values].unique())
ps_mask[~ps_mask.index.isin(intersection)] = False
ssa_ps_mask[~ssa_ps_mask.index.isin(intersection)] = False

# +
# sns.jointplot(data=a.df[ps_mask].sample(frac=0.1), x='dm', y='mag_med', kind='hex', )
fig, axes = plt.subplots(1,2, figsize=(12,6))
plt.style.use(cfg.RES_DIR + 'stylelib/sf_ensemble.mplstyle')
lims = [(16.5,21),(-1,1)]
for ax in axes:
    ax.grid(visible=True, which='both', alpha=0.5)
    ax.grid(visible=True, which='minor', alpha=0.2)
    ax.set(xlim=lims[0], ylim=lims[1])

sns.histplot(data=df.loc[ps_mask    ].sample(frac=0.1), x='mag_med', y='dm', binrange=lims, cmap='Spectral_r', ax=axes[0], thresh=4)
axes[0].set(ylabel=r'$\Delta m (\mathrm{PS-PS})$ [mag]', xlabel='Median magnitude [mag]')
sns.histplot(data=df.loc[ssa_ps_mask].sample(frac=0.1), x='mag_med', y='dm', binrange=lims, cmap='Spectral_r', ax=axes[1], thresh=4)
axes[1].set(ylabel=r'$\Delta m (\mathrm{SSA-PS})$ [mag]', xlabel='Median magnitude [mag]')
# -

mag_edges = np.arange(18.5, 20, 0.25)
mag_centres = (mag_edges[1:] + mag_edges[:-1]) / 2
n_bins = len(mag_edges) - 1

mag_mask = df['mag_med'].between(mag_edges[0], mag_edges[-1])
ps_mask_ = ps_mask[mag_mask]
ssa_ps_mask_ = ssa_ps_mask[mag_mask]

df_masked = df[mag_mask].copy()
df_masked['bin_idx'] = pd.cut(df_masked['mag_med'], mag_edges, labels=False)

# +
ps_var = np.zeros(n_bins)
ssa_ps_var = np.zeros(n_bins)
def var_iqr(series):
    q1, q2 = series.quantile([0.25,0.75]).values
    return 0.549081*(q2-q1)**2
for i in range(n_bins):
    # ps_var[i]     = df_masked[(df_masked['bin_idx']==i) & ps_mask_    ]['dm'].var()
    # ssa_ps_var[i] = df_masked[(df_masked['bin_idx']==i) & ssa_ps_mask_]['dm'].var()
    ps_var[i]     = var_iqr(df_masked[(df_masked['bin_idx']==i) & ps_mask_    ]['dm'])
    ssa_ps_var[i] = var_iqr(df_masked[(df_masked['bin_idx']==i) & ssa_ps_mask_]['dm'])

ps_std = ps_var**0.5
ssa_ps_std = ssa_ps_var**0.5
ssa_std = (ssa_ps_var-ps_var)**0.5
# -

# fit with exponential
from scipy.optimize import curve_fit
def exp(x, a, b, c, d):
    return a * np.exp(b * (x-d)) + c


# +
plt.style.use(cfg.RES_DIR + 'stylelib/sf_ensemble.mplstyle')

fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.grid(visible=True, which='both', alpha=0.6)
ax.grid(visible=True, which='minor', alpha=0.2)

popt_ps, pcov = curve_fit(exp, mag_centres, ps_std, p0=[0.1, 1.2, 0.1, 18])
ax.plot(mag_centres, exp(mag_centres, *popt_ps), 'b-', label=rf'${popt_ps[0]:.2f}\mathrm{{exp}}(x-{popt_ps[3]:.2f})+{popt_ps[2]:.2f}$')
popt_ssa_ps, pcov = curve_fit(exp, mag_centres, ssa_ps_std, p0=[0.1, 1.2, 0.1, 18])
ax.plot(mag_centres, exp(mag_centres, *popt_ssa_ps), 'r-', label=rf'${popt_ssa_ps[0]:.2f}\mathrm{{exp}}(x-{popt_ssa_ps[3]:.2f})+{popt_ssa_ps[2]:.2f}$')
popt_ssa, pcov = curve_fit(exp, mag_centres, ssa_std, p0=[0.1, 1.2, 0.1, 18])
ax.plot(mag_centres, exp(mag_centres, *popt_ssa), 'k-', label=rf'${popt_ssa[0]:.2f}\mathrm{{exp}}(x-{popt_ssa[3]:.2f})+{popt_ssa[2]:.2f}$')

ax.plot(mag_centres, ps_std, label='PS', marker='.')
ax.plot(mag_centres, ssa_ps_std, label='SSA_PS', marker='.')
ax.plot(mag_centres, ssa_std+0.05, label='SSA', marker='.')
ax.legend()
ax.set(xlabel='Median magnitude [mag]', ylabel=r'$\sigma_{\Delta m}$ [mag]')

ax.plot(mag_centres, 0.06751*mag_centres - 1.08)


# +
# sns.jointplot(data=a.df[ps_mask].sample(frac=0.1), x='dm', y='mag_med', kind='hex', )
fig, axes = plt.subplots(1,2, figsize=(12,6))
plt.style.use(cfg.RES_DIR + 'stylelib/sf_ensemble.mplstyle')
lims = [(16.5,21),(-1,1)]
for ax in axes:
    ax.grid(visible=True, which='both', alpha=0.5)
    ax.grid(visible=True, which='minor', alpha=0.2)
    ax.set(xlim=lims[0], ylim=lims[1])

sns.histplot(data=df.loc[ps_mask    ].sample(frac=0.1), x='mag_med', y='dm', binrange=lims, cmap='Spectral_r', ax=axes[0], thresh=4)
axes[0].set(ylabel=r'$\Delta m (\mathrm{PS-PS})$ [mag]', xlabel='Median magnitude [mag]')
axes[0].plot(mag_centres, ps_std, label='PS', marker='.')
axes[0].plot(mag_centres,-ps_std, label='PS', marker='.')
sns.histplot(data=df.loc[ssa_ps_mask].sample(frac=0.1), x='mag_med', y='dm', binrange=lims, cmap='Spectral_r', ax=axes[1], thresh=4)
axes[1].set(ylabel=r'$\Delta m (\mathrm{SSA-PS})$ [mag]', xlabel='Median magnitude [mag]')
axes[1].plot(mag_centres, -0.05+ssa_ps_std, label='SSA_PS', marker='.')
axes[1].plot(mag_centres, -0.05-ssa_ps_std, label='SSA_PS', marker='.')

x = np.linspace(16, 21, 100)
axes[0].plot(x, exp(x, *popt_ps), 'b-')
axes[0].plot(x, -exp(x, *popt_ps), 'b-')
axes[1].plot(x, -0.05+exp(x, *popt_ssa_ps), 'r-')
axes[1].plot(x, -0.05-exp(x, *popt_ssa_ps), 'r-')
# -



exp(-100, *popt_ssa)+0.05
