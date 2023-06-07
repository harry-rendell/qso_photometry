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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
path = cfg.W_DIR
plt.style.use(cfg.FIG.STYLE_DIR + 'style1.mplstyle')

df = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr14q/dr14q_uid_mi_coords.csv', index_col='uid')

df = df[df['mi']!=9999]
df


# ## M_i vs z

# +
def sns_correlate(df, vmin, vmax, save=False):
    from matplotlib.colors import LogNorm
    xname = 'z'
    yname = 'mi'
    
    data = df[[xname,yname]]
    
#     xbounds, ybounds = dr.properties[[xname,yname]].quantile(q=[0.001,0.999]).values.T
    xbounds, ybounds = (0,5), (-30,-20)
    
    data = data[((xbounds[0] < data[xname])&(data[xname] < xbounds[1])) & ((ybounds[0] < data[yname])&(data[yname] < ybounds[1]))]
    
    bounds={xname:xbounds, yname:ybounds}
    g = sns.JointGrid(x=xname, y=yname, data=data, xlim=bounds[xname], ylim=bounds[yname], height=6)
    g = g.plot_joint(plt.hexbin, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='Blues')
    g.ax_marg_x.hist(data[xname], bins=200, color='royalblue')
    g.ax_marg_y.hist(data[yname], bins=200, orientation='horizontal', density = True, color='royalblue')
    g.ax_joint.invert_yaxis()
    
    g.ax_joint.set(xlabel='Redshift', ylabel=r'M$_\mathrm{i}[\mathrm{z}=2]$')
#     g.ax_marg_y.hist(data[yname], bins=200, orientation='horizontal', cumulative=True, alpha=0.2, color='k')#, yticks=[1e-3, 1e-1, 1])
    # Could show 95, 99, 99.9% intervals on magerr histplot? Will need to do np.stats.quantile and ax.axvline
#     q = [0.85,0.95,0.99]
#     quantiles = zip(np.quantile(data[yname],q),q)
#     for q_val,q in quantiles:
#         g.ax_marg_y.axhline(y=q_val, lw=2, ls='--', color='k')
#         g.ax_marg_y.text(y=q_val+0.003,x=0.8, s=f'{q*100:.0f}%: {q_val:.2f}', fontdict={'size':12}, horizontalalignment='center')
#     g.ax_marg_y.set(xscale='log')

#     plt.suptitle(self.obj + ' ' + band, x=0.1, y=0.95)
    if save:
        g.savefig(path+'analysis/plots/Mi_z.pdf')


# -

sns_correlate(df, 3e0, 1e3, True)

# ---

# ## Redshift

fig, ax = plt.subplots(1,1, figsize=(9,3))
ax.hist(df['z'], bins=150, range=(0,5), color='royalblue');
ax.set(ylabel='Number of Quasars', xlabel='Redshift', xlim=[0,5])
# fig.savefig(path+'analysis/plots/redshift.pdf', bbox_inches='tight')

# ---

# ## mag dist

import pandas as pd 
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)
# %matplotlib inline
from multiprocessing import Pool
# from profilestats import profile
from scipy.stats import binned_statistic
from module.classes.analysis import *

# obj = 'calibStars'
# ID  = 'uid_s'
obj = 'qsos'
ID = 'uid'
band = 'r'

dr = analysis(ID, obj, band)

dr.read_in(redshift=True)
dr.band = 'r'
dr.group(keys = ['uid'],read_in=True, survey='all')

dr.df_grouped

fig, ax = plt.subplots(1,1, figsize=(9,3))
ax.hist(dr.df_grouped['mag_mean'], bins=150, range=(16,23), color='royalblue');
ax.set(ylabel='Number of Quasars', xlabel=r'$r$-band magnitude', xlim=[16,23])
# fig.savefig(path+'analysis/plots/qso_mag_dist.pdf', bbox_inches='tight')
