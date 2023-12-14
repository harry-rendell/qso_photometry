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
from module.plotting.plot_vac_groups import plot_groups_lambda_lbol
from module.preprocessing import parse, binning, color_transform
from module.assets import load_vac
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

obj = 'qsos'
ID = 'uid' if obj == 'qsos' else 'uid_s'

# +
n_bins = 10
bands = 'gri'

vac = load_vac('qsos', usecols=['z','Lbol'])
sf = []
for band in bands:
    s = pd.read_csv(cfg.D_DIR + f'computed/{obj}/features/{band}/SF_{n_bins}_bins.csv', index_col=ID)
    s['band'] = band
    vac['wavelength'] = color_transform.calculate_wavelength(band, vac['z'])
    s = s.join(vac, on=ID)
    sf.append(s)
sf = pd.concat(sf).sort_index()
# -

sf.columns

# +
# kwargs = {'n_l':2, 'n_L':11, 'l_low':1000, 'l_high':5000, 'L_low':45.2, 'L_high':47.2}
kwargs = {'n_l':2, 'n_L':11, 'l_low':1000, 'l_high':5000, 'L_low':44.7, 'L_high':47.2}
mask_dict_wide, l_edges_w, L_edges_w = binning.create_mask_lambda_lbol(sf, threshold=1000, verbose=False, gap=0, return_edges=True, **kwargs)
fig = plot_groups_lambda_lbol(sf, mask_dict_wide, **kwargs)

# kwargs = {'n_l':11, 'n_L':11, 'l_low':1000, 'l_high':5000, 'L_low':45.2, 'L_high':47.2}
kwargs = {'n_l':11, 'n_L':11, 'l_low':1000, 'l_high':5000, 'L_low':44.7, 'L_high':47.2}
mask_dict_narrow, l_edges_n, L_edges_n = binning.create_mask_lambda_lbol(sf, threshold=1000, verbose=False, gap=0, return_edges=True, **kwargs)
fig = plot_groups_lambda_lbol(sf, mask_dict_narrow, **kwargs)


# -

def plot_sfs(df, mask_dict, statistic, ax, plot_kwargs={'lw':1}, title=None):
    import re

    sort_key = lambda x: (int(re.search(r'(\d+)_(\d+)$', x).group(1)) , int(re.search(r'(\d+)_(\d+)$', x).group(1))) 
    sorted_cols = sorted([col for col in df.columns if col.startswith(statistic)], key=sort_key)
    sorted_n = sorted([col for col in df.columns if col.startswith('n_')], key=sort_key)
    mjd_centres = [(int(item.split('_')[2]) + int(item.split('_')[3])) / 2 for item in sorted_cols]
    plt.style.use(cfg.RES_DIR + 'stylelib/gradient.mplstyle')
    for key, mask in mask_dict.items():
        SF2 = df[mask][sorted_cols].mean().values
        SF = np.where(SF2 > 0, SF2, np.nan)**0.5
        n_points_per_bin = df[mask][sorted_cols].notna().sum().values
        ax.plot(mjd_centres, SF, label=key, **plot_kwargs)
        # ax.scatter(mjd_centres, SF, s=n_points_per_bin/n_points_per_bin.mean()*30)

    ax.set(xlabel='∆t', ylabel='SF', title=title, xscale='log', yscale='log', xlim=(1e0, 3e4), ylim=(2e-2, 3e0))
    ax.legend(bbox_to_anchor=(1.15, 1))
    ax.grid(visible=True, which='both', alpha=0.6)
    ax.grid(visible=True, which='minor', alpha=0.2)     


# +
stat = 'SF2_cw'
## overlay ensemble on each plot
N_WAVE = 9
fig, ax = plt.subplots(N_WAVE+1,1, figsize=(12,N_WAVE*6), sharex=True)
for i in range(N_WAVE):
    plot_sfs(sf, mask_dict_wide, stat, ax=ax[i+1], plot_kwargs={'lw':0.5, 'color':'k', 'ls':'--'})
    plot_sfs(sf, {key:value for key, value in mask_dict_narrow.items() if key[0] == i}, stat, ax=ax[i+1], title=rf'{l_edges_n[i]}Å < $\lambda$ < {l_edges_n[i+1]}Å')

plot_sfs(sf, mask_dict_wide, stat, ax=ax[0], title=rf'{l_edges_w[0]:.0f} Å < $\lambda$ < {l_edges_w[1]:.0f} Å')


# +
stat = 'SF2_w'
## overlay ensemble on each plot
N_WAVE = 9
fig, ax = plt.subplots(N_WAVE+1,1, figsize=(12,N_WAVE*6), sharex=True)
for i in range(N_WAVE):
    plot_sfs(sf, mask_dict_wide, stat, ax=ax[i+1], plot_kwargs={'lw':0.5, 'color':'k', 'ls':'--'})
    plot_sfs(sf, {key:value for key, value in mask_dict_narrow.items() if key[0] == i}, stat, ax=ax[i+1], title=rf'{l_edges_n[i]}Å < $\lambda$ < {l_edges_n[i+1]}Å')

plot_sfs(sf, mask_dict_wide, stat, ax=ax[0], title=rf'{l_edges_w[0]:.0f} Å < $\lambda$ < {l_edges_w[1]:.0f} Å')

# -

fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
plot_sfs(sf, {key:value for key, value in mask_dict_narrow.items() if key[0] == i}, stat, ax=ax[0], title=rf'{l_edges_n[i]}Å < $\lambda$ < {l_edges_n[i+1]}Å')
plot_sfs(sf, mask_dict_wide, stat, ax=ax[1], title=rf'{l_edges_w[0]:.0f} Å < $\lambda$ < {l_edges_w[1]:.0f} Å')







vac = pd.read_csv(os.path.join(cfg.D_DIR,'catalogues/qsos/dr16q/dr16q_vac_shen_matched.csv'), index_col=ID)
vac = parse.filter_data(vac, cfg.PREPROC.VAC_BOUNDS, dropna=False)

features = features.join(vac[['Lbol','MBH']], how='left', on='uid')

mjd_edges = binning.construct_T_edges(t_max=cfg.PREPROC.MAX_DT['REST']['qsos'][band], n_edges=10+1)
mjd_centres = (mjd_edges[1:] + mjd_edges[:-1])/2

features

# +
# g = sns.jointplot(data=features, x='f1_6', y='Lbol', kind='hist')

# plt.style.use('paired')
fig, ax = plt.subplots(1,1, figsize=(10,10))
bins=80
threshold=1
x = 'f0_3'
sns.histplot(data=features, x=x, y='Lbol', bins=bins, cmap='Spectral_r', thresh=threshold, ax=ax, log_scale=(True, False), binrange=((-2.5,0),(44,48)))
# ax.set(xlabel=x, ylabel='Lbol', xlim=(10**-2.5,1e4), ylim=(1e-3,10**1.5))
ax.set_facecolor(plt.get_cmap('Spectral_r')(0))
plt.grid(visible=False, which='both')


# -

from sklearn.decomposition import PCA
from sklearn import datasets, manifold

# tsne_keys = ['dm','dt','uid']
# X = df[['Lbol','MBH','nEdd']+tsne_keys]
# X = parse.filter_data(X, {stat:X[stat].quantile([0,1]).values for stat in tsne_keys}, dropna=True)
fs = [f'f{0}_{i}' for i in range(2,6)]
t_sne = manifold.TSNE(
    n_components=2,
    perplexity=30,
    init="random",
    n_iter=250,
    random_state=1)
Y = t_sne.fit_transform(features[fs].dropna().values)

# +
# X['tsne1'] = S_t_sne[:,0]
# X['tsne2'] = S_t_sne[:,1]

# color_key = 'uid'
# X = X.join(vac[color_key], on=ID)
# colors = (X[color_key]-X[color_key].min())/(X[color_key].max()-X[color_key].min())
# plt.scatter(X['tsne1'], X['tsne2'], s=2, color=plt.cm.jet(colors))

plt.scatter(Y[:,0], Y[:,1], s=0.01)
# -

# Do PCA on the features and plot the first two components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(features[fs].dropna().values)
X_pca = pca.transform(features[fs].dropna().values)

# +
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(X_pca[:,0], X_pca[:,1], s=0.01)
ax.set(xlim=(-1,1), ylim=(-1,1))

fig, ax = plt.subplots(1,1, figsize=(10,10))
bins=100
from matplotlib.colors import LogNorm
sns.histplot(x=X_pca[:,0], y=X_pca[:,1], bins=bins, cmap='Spectral_r', ax=ax, binrange=((-1,1),(-1,1)), thresh=1)
ax.set(xlabel='PCA 1', ylabel='PCA 2', xlim=(-1,1), ylim=(-1,1))
ax.set_facecolor(plt.get_cmap('Spectral_r')(0))
plt.grid(visible=False, which='both')
# -



bounds_z = np.array([-3.5,-1,0,1,3.5])
groups, bounds_values = binning.calculate_groups(vac['Lbol'], bounds = bounds_z)
n_groups = len(groups)
for i, group in enumerate(groups):
    df_fit.loc[df_fit.index.isin(group),'group'] = i
df_fit['group'].value_counts()
