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

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.plotting.common import savefigs
from module.preprocessing import data_io, parse, binning
from module.preprocessing.binning import construct_T_edges
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

obj = 'qsos'
ID = 'uid' if obj == 'qsos' else 'uid_s'
band = 'r'
kwargs = {'dtypes': cfg.PREPROC.dtdm_dtypes,
          'ID':ID,
          'basepath': cfg.D_DIR + f'merged/{obj}/clean/dtdm_{band}',
          'nrows':30000000}
dtdm = data_io.dispatch_reader(kwargs, multiproc=True, fnames=['dtdm_000000_005000.csv','dtdm_005001_010000.csv','dtdm_010001_015000.csv','dtdm_020001_025000.csv'])

print(f'memory use of dtdm: {sys.getsizeof(dtdm)/1e9:.2f} GB')

# ideas:
# 1. compute ∆m std or ∆m^2-∆e^2 std for a bunch of quasars (with sufficient observations) and correlate the hell out of it
# 2. compute z-score of ∆m within ensemble distribution (better comparison with total population) and correlate that
# 3. 

T_edges = np.array([10, dtdm['dt'].max()/5, dtdm['dt'].max()])
# print ∆t edges:
for i in range(len(T_edges)-1):
    print(f'group {i}: {T_edges[i]:.2f} < ∆t < {T_edges[i+1]:.2f}')

from scipy.stats import median_abs_deviation as MAD
results = []
def calculate_ensemble_stats(dtdm):
    for i in [0,1]:
        mask = (T_edges[i] <= dtdm['dt']) & (dtdm['dt'] < T_edges[i+1])
        results.append(dtdm[mask.values]['dm'].groupby('uid').agg({(f'mean_{i}', np.mean), (f'std_{i}', np.std), (f'MAD_{i}', MAD), (f'counts_{i}','count')}))
    return results


results = calculate_ensemble_stats(dtdm)
df = results[0].join(results[1], on='uid')
df['total_counts'] = df['counts_0'] + df['counts_1']

vac = pd.read_csv(os.path.join(cfg.D_DIR,'catalogues/qsos/dr16q/dr16q_vac_shen_matched.csv'), index_col=ID)
# vac = vac.rename(columns={'z':'redshift_vac'});
vac = parse.filter_data(vac, cfg.PREPROC.VAC_BOUNDS, dropna=False)

koi = ['Lbol','MBH','nEdd','z']
df = df.join(vac[koi], on=ID)

fig, ax = plt.subplots(figsize=(10,10))
stat='mean'
sns.scatterplot(df, x=f'{stat}_0', y=f'{stat}_1', hue='total_counts', ax=ax)

fig, ax = plt.subplots(figsize=(10,10))
stat='std'
sns.scatterplot(df, x=f'{stat}_0', y=f'{stat}_1', hue='total_counts', ax=ax)

bounds_z = np.array([-3.5,-1,0,1,3.5])
groups, bounds_values = binning.calculate_groups(vac['Lbol'], bounds = bounds_z)
n_groups = len(groups)

for i, group in enumerate(groups):
    df.loc[df.index.isin(group),'group'] = i
df['group'].value_counts()

# +
stat='std'
from matplotlib.colors import LogNorm

# g = sns.jointplot(data=df, x=f'{stat}_0', y=f'{stat}_1', kind='hue', height=10, xlim=[0,0.8], ylim=[0,0.8], color='b', joint_kws={'gridsize':50, 'fill':True, 'levels':10, 'thresh':0.1}, )
g = sns.jointplot(data=df, x=f'{stat}_0', y=f'{stat}_1', hue='group', kind='scatter', height=10, xlim=[0,0.8], ylim=[0,0.8], color='b')
# g = sns.JointGrid(data=df, x=f'{stat}_0', y=f'{stat}_1', xlim=[0,1], ylim=[0,1], height=8)
# g = g.plot_joint(plt.hexbin, norm=LogNorm(vmin=1, vmax=1e1), cmap='jet')
# -

from sklearn.decomposition import PCA
from sklearn import datasets, manifold

tsne_keys = ['dm','dt','uid']
X = dtdm.reset_index()[tsne_keys].head(100000)
# X = df[['Lbol','MBH','nEdd']+tsne_keys]
# X = parse.filter_data(X, {stat:X[stat].quantile([0,1]).values for stat in tsne_keys}, dropna=True)
t_sne = manifold.TSNE(
    n_components=2,
    perplexity=30,
    init="random",
    n_iter=250,
    random_state=1)
Y = t_sne.fit_transform(X[tsne_keys].values)

X

# +
# X['tsne1'] = S_t_sne[:,0]
# X['tsne2'] = S_t_sne[:,1]

color_key = 'uid'
# X = X.join(vac[color_key], on=ID)
colors = (X[color_key]-X[color_key].min())/(X[color_key].max()-X[color_key].min())
# plt.scatter(X['tsne1'], X['tsne2'], s=2, color=plt.cm.jet(colors))

plt.scatter(Y[:,0], Y[:,1], s=2, color=plt.cm.jet(colors+0.2))

# +
tsne_keys = ['MAD_0','MAD_1']
X = df[['Lbol','MBH','nEdd']+tsne_keys]
X = parse.filter_data(X, {stat:X[stat].quantile([0,1]).values for stat in tsne_keys}, dropna=True)
t_sne = manifold.TSNE(
    n_components=2,
    perplexity=30,
    init="random",
    n_iter=250,
    random_state=1)
S_t_sne = t_sne.fit_transform(X[tsne_keys].values)

X['tsne1'] = S_t_sne[:,0]
X['tsne2'] = S_t_sne[:,1]

color_key = 'Lbol'
colors = (X[color_key]-X[color_key].min())/(X[color_key].max()-X[color_key].min())
plt.scatter(X['tsne1'], X['tsne2'], s=2, color=plt.cm.jet(colors))
# -

pca = PCA(n_components=2)
y = pca.fit_transform(X[tsne_keys].values)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(y[:,0], y[:,1], s=2, color=plt.cm.jet(colors))

pca = PCA(n_components=2)
y = pca.fit_transform(X[tsne_keys].values)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(X[tsne_keys[0]],X[tsne_keys[1]], s=2, color=plt.cm.jet(colors))

# +
stat='MAD'
from matplotlib.colors import LogNorm

g = sns.jointplot(data=df, x=f'{stat}_1', y='Lbol', kind='scatter', palette='Spectral', height=10, xlim=[0,0.5], ylim=[44,48], color='b')#, joint_kws={'gridsize':50, 'fill':False, 'levels':5, 'thresh':0.1})
# g = g.jointplot(data=df.tail(1000), x=f'{stat}_0', y=f'{stat}_1', kind='kde', height=10, xlim=[0,0.8], ylim=[0,0.8], color='r', joint_kws={'gridsize':50, 'fill':False, 'levels':10, 'thresh':0.2}, )
# g = sns.JointGrid(data=df, x=f'{stat}_0', y=f'{stat}_1', xlim=[0,1], ylim=[0,1], height=8)
# g = g.plot_joint(plt.hexbin, norm=LogNorm(vmin=1, vmax=1e1), cmap='jet')

# +
stat='std'
from matplotlib.colors import LogNorm

g = sns.jointplot(data=df, x=f'{stat}_0', y=f'{stat}_1', kind='kde', hue='group', palette='Spectral', height=10, xlim=[0,0.5], ylim=[0,0.5], color='b', joint_kws={'gridsize':50, 'fill':False, 'levels':5, 'thresh':0.1})
# g = g.jointplot(data=df.tail(1000), x=f'{stat}_0', y=f'{stat}_1', kind='kde', height=10, xlim=[0,0.8], ylim=[0,0.8], color='r', joint_kws={'gridsize':50, 'fill':False, 'levels':10, 'thresh':0.2}, )
# g = sns.JointGrid(data=df, x=f'{stat}_0', y=f'{stat}_1', xlim=[0,1], ylim=[0,1], height=8)
# g = g.plot_joint(plt.hexbin, norm=LogNorm(vmin=1, vmax=1e1), cmap='jet')
# -

fig, ax = plt.subplots(figsize=(10,10))
stat='MAD'
sns.scatterplot(df, x=f'{stat}_0', y=f'{stat}_1', hue='total_counts', ax=ax)

# # Plot individual ∆m histograms for quasars for different time lags

uids = dtdm.index.value_counts().index
fig, axes = plt.subplots(15,2, figsize=(18,50))
for j, ax in enumerate(axes.ravel()):
    ax.text(0.02,0.9, f'uid: {uids[j]}', transform=ax.transAxes)
    x = dtdm.loc[uids[j]]
    ypos = 0.8
    for i in [2,4,6,8]:
        mask = (T_edges[i] <= x['dt']) & (x['dt'] < T_edges[i+1])
        if uids[j] in x[mask].index:
            x[mask]['dm'].hist(bins=50, ax=ax, label=f'{T_edges[i]}<∆t<{T_edges[i+1]}', alpha=0.5, density=True, range=(-1.5,1.5))
            ax.text(0.02,ypos, f'N = {mask.sum()}', transform=ax.transAxes)
            ax.set(xlim=[-1.5,1.5])
            ax.legend(loc = 'upper right')
        ypos -= 0.1


