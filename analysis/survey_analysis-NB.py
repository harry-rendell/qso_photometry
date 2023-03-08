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
from funcs.analysis.analysis import *

# obj = 'calibStars'
# ID  = 'uid_s'
obj = 'qsos'
ID = 'uid'
band = 'r'


# +
def reader(n_subarray):
    return pd.read_csv(wdir+'data/merged/{}/{}_band/lc_{}.csv'.format(obj,band,n_subarray), comment='#', nrows=None, index_col = ID, dtype = {'catalogue': np.uint8, 'mag_ps': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})



# -

# band = 'r'
dr = analysis(ID, obj)

dr.read_in(reader, redshift=True)
# dr.group(keys = ['uid'],read_in=True)

dr.df

dr.band = band

dr.group(keys = ['uid'],read_in=True)

dr.df_grouped

# +
# What does this do
# dr.df = dr.df[dr.df.index.duplicated(keep=False)]
# -

fig, ax = plt.subplots(1,2, figsize=(10,6))
dr.df_grouped[['mjd_ptp_rf','mjd_ptp']].hist(bins=200, ax=ax)
print('max ∆t for observer frame: {:.2f}'.format(dr.df_grouped['mjd_ptp'].max()))
print('max ∆t for rest     frame: {:.2f}'.format(dr.df_grouped['mjd_ptp_rf'].max()))

fig, ax = plt.subplots(1,1,figsize=(10,10))
dr.df_grouped['mag_count'].hist(bins=350, ax=ax)
ax.set(xlim=[0,1000], xlabel='number of observations for given quasar')

# +
# dr.df_grouped.to_csv('/disk1/hrb/python/data/merged/meta_data/df_gb_uid_r.csv')
# -

'{:,}'.format(int((count*(count-1)/2).sum()))

dr.df['catalogue'].unique()

dr.merge_with_catalogue(catalogue = 'dr12_vac', remove_outliers=False, prop_range_any={'MBH_CIV':(0,100)})

# ### L-z plot
# ---

dr.properties['redshift'].hist(bins=100)

x, y = dr.properties[['redshift','MBH_CIV']].values.T
plt.hist2d(x,y, bins=100);

fig, ax = plt.subplots(1,1, figsize=(10,7))
dr.properties.loc[dr.properties['MBH_CIV']==-9.999,'redshift'].hist(bins=100, alpha=0.7, ax=ax, label='undefined mass')
dr.properties.loc[dr.properties['MBH_CIV']!=-9.999,'redshift'].hist(bins=100, alpha=0.7, ax=ax, label='defined mass')
ax.legend()
ax.set(xlabel='Redshift', ylabel='Number of Quasars')

dr.properties['redshift'].hist(bins=100)

# +
import seaborn as sns
def sns_correlate(self, band, vmin, vmax, save=False):
    from matplotlib.colors import LogNorm
    xname = 'redshift'
    yname = 'Lbol'
    
    data = self.properties[[xname,yname]]
    
#     xbounds, ybounds = dr.properties[[xname,yname]].quantile(q=[0.001,0.999]).values.T
    xbounds, ybounds = (1,4), (45,48)
    bounds={xname:xbounds, yname:ybounds}
    g = sns.JointGrid(x=xname, y=yname, data=data, xlim=bounds[xname], ylim=bounds[yname], height=9)
    g = g.plot_joint(plt.hexbin, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='jet')
    g.ax_marg_x.hist(data[xname], bins=300)
    g.ax_marg_y.hist(data[yname], bins=300, orientation='horizontal', density = True)
#     g.ax_marg_y.hist(data[yname], bins=200, orientation='horizontal', cumulative=True, alpha=0.2, color='k')#, yticks=[1e-3, 1e-1, 1])
    # Could show 95, 99, 99.9% intervals on magerr histplot? Will need to do np.stats.quantile and ax.axvline
#     q = [0.85,0.95,0.99]
#     quantiles = zip(np.quantile(data[yname],q),q)
#     for q_val,q in quantiles:
#         g.ax_marg_y.axhline(y=q_val, lw=2, ls='--', color='k')
#         g.ax_marg_y.text(y=q_val+0.003,x=0.8, s=f'{q*100:.0f}%: {q_val:.2f}', fontdict={'size':12}, horizontalalignment='center')
#     g.ax_marg_y.set(xscale='log')

    plt.suptitle(self.obj + ' ' + band, x=0.1, y=0.95)
    if save:
        g.savefig('{}/plots/mag_magerr_{}_{}.pdf'.format(self.obj, band))


# -

sns_correlate(dr, band, 1e0, 1e3, save=Falsez

# ---

plt.hist(dr.properties['MBH_CIV'], bins=100)
len(dr.properties)

dr.merge_with_catalogue(catalogue = 'dr14_vac', remove_outliers=True, prop_range_any={'MBH_CIV':(6,12)})

plt.hist(dr.properties['MBH_CIV'], bins=100)
len(dr.properties)

# +
# dr.merge_with_catalogue(catalogue = 'dr12', remove_outliers=True, prop_range_any={'MBH_CIV':(5,13)})

# +
# grouped = pd.read_csv('/disk1/hrb/python/data/surveys/ztf/meta_data/ztfdr2_gb_uid_{}.csv'.format('r'),index_col = 0)
# test = grouped.join(df_z, on = 'uid', how = 'left')
# test.to_csv('/disk1/hrb/python/data/surveys/ztf/meta_data/ztfdr2_gb_uid_{}.csv'.format('r'))

# +
# #need to group by and apply this fn.
# def slope(group):
#     if len(group) > 1:
#         x = group['mjd']
#         y = group['mag']
#         return ((x-x.mean())*(y-y.mean())).sum()/((x-x.mean())**2).sum()
#     else:
#         return np.NaN

# slopes = dr.df.groupby('uid').apply(slope)
# slopes.name = 'slope'
# slopes_df = dr.df_grouped.join(slopes, how = 'inner', on='uid')

# +
# dr.plot_series(dr.df_grouped['slope'][dr.df_grouped['mag_count']>10].sort_values().head(10).index)
# -

# ### Group uids by MBH distribution

dr.properties[(dr.properties['Lbol'])

key = 'Lbol' 
bounds, z_score, bounds_values, mean, std, ax = dr.bounds(key, bounds = np.array([-5,-1,-0.5,0,0.5,1,5]))

test = dr.properties[(7 < dr.properties[key]) & (dr.properties[key] < 10.5)]

n, bin_edges, _ = plt.hist(test[key], bins=68, range=(7.1,10.5), density=True)

bin_edges

np.savetxt('bin_edges_mbh_civ.csv',bin_edges, fmt='%.2f')

np.savetxt('prob_mbh_civ.csv',n)


# +
def loc_uids(self, lower, upper):
    lower = int(lower)
    upper = int(upper)
    uids = self.df.index.unique()[:]
    boolean = ((lower<uids)&(uids<=upper))
    uids = uids[boolean]
    return uids

def savedtdm(sub_uids):
    df = pd.DataFrame(columns=['uid','dt','dm','cat'])
    print('computing: {} to {}\n'.format(min(sub_uids), max(sub_uids)))
    for batch in np.array_split(sub_uids,4):
        df_batch = dr.save_dtdm_rf(sub_uids)
        df = df.append(df_batch)
    print('saving:    {} to {}\n'.format(min(sub_uids), max(sub_uids)))
    df.to_csv('/disk1/hrb/python/analysis/qsos/computed/dtdm/raw/dtdm_raw_{}_{:06d}_to_{:06d}.csv'.format(dr.band,min(sub_uids),max(sub_uids)),index=False)

# if __name__ == '__main__': 
#     p = Pool(4)
#     p.map(savedtdm, np.array_split(loc_uids(dr, 0, 16e5),4))



# -

def calc_moments(bins,weights):
    k = np.array([3,4])
    x = bins*weights
    z = (x-x.mean(axis=1)[:,np.newaxis])/x.std(axis=1)[:,np.newaxis]
    return x.mean(axis=1), (z**4).mean(axis = 1) - 3
# 1st and second moment are unstandardised (else they would be 0,1 respectively). 3rd and 4th moments are standardised.


dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = dr.calc_dtdm(uids=None, n_bins_t = 1000, n_bins_m = 200, t_max=3020, t_spacing = 'log', m_spacing = 'log', read_in = 1, key = 'Lbol')

# +
import matplotlib
matplotlib.rc('font', **{'size': 13})
matplotlib.rc('text', usetex=False)

fig, ax, fig2, axes2, fig3, axes3 = dr.plot_sf_moments(key, bounds, ztf=False)
plt.figure(figsize=(15,8))
axes2[0].set(ylim=[-0.1,0.045])
axes2[1].set(ylim=[-0.7,0.5])

ax.set(xlabel = '∆t (days)')
# ax.set(ylim = [1e-1,5e-1])
axes2[1].legend(loc='upper center', 
             bbox_to_anchor=(1.1, 1.5),fancybox=False, shadow=False)
fig .savefig('/disk1/hrb/python/analysis/qsos/plots/dtdm/SF_{}.pdf'.format(key), bbox_inches='tight')
fig2.savefig('/disk1/hrb/python/analysis/qsos/plots/dtdm/mean_kurtosis_{}.pdf'.format(key), bbox_inches='tight')


# +
# Testing two SF definitions.
# Fits a gaussian to ∆m distribution and plots it as a function of ∆t
def gaussian(x,peak,offset):
    sigma = (2*np.pi)**-0.5*1/peak
    return peak*np.exp( -( (x-offset)**2/(2*sigma**2) ) )

from scipy.optimize import curve_fit
stds_fit = np.zeros(19)
stds     = np.zeros(19)
dms_binned_norm = np.zeros((19,200))
for i in range(19):
    m,_= np.histogram(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[i], density=True);
    popt, _ = curve_fit(gaussian, m_bin_edges[:-1], m, p0 = [m.max(),m_bin_edges[:-1][m.argmax()]])
    stds[i] = (2*np.pi)**-0.5*1/popt[0]
    dms_binned_norm[i] = m
SF_1 = (((m_bin_centres**2)*dms_binned_norm).sum(axis=1)/dms_binned_norm.sum(axis=1))**0.5
SF_2 = (((m_bin_centres**2)*dms_binned).sum(axis=1)/dms_binned.sum(axis=1))**0.5

#Find the mathematical difference between the two definitions above

fig, ax = plt.subplots(1,1,figsize = (14,8))
ax.plot(t_bin_chunk_centres,SF_1, label = 'sf_1', lw = 0.5, marker = 'o')
ax.plot(t_bin_chunk_centres,SF_2, label = 'sf_2', lw = 0.5, marker = 'o')
# ax.scatter(t_bin_chunk_centres,stds, label = 'stds')
# ax.set(xscale='log',yscale='log',xticks = [0,1,2,3]);
ax.legend()

# +
########REST
dms_binned, dts_binned, t_bin_edges, t_bin_chunk, t_bin_chunk_centres, m_bin_edges, m_bin_centres, t_dict = dr.calc_dtdm(uids=None, n_bins_t = 1000, n_bins_m = 200, t_max=7600, t_spacing = 'log', m_spacing = 'log', read_in = 1, key = 'Lbol')
cmap = plt.cm.cool
skip = 1
if skip > 1:
    fig, axes = plt.subplots((19//skip + 1),1,figsize = (15,3*(19//skip + 1)))
else:
    fig, axes = plt.subplots(19,1,figsize = (15,3*19))

for i, ax in enumerate(axes):
    n=1
    m,_,_=ax.hist(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[i], alpha = 1, density=True, label = t_dict[i], color = cmap(i/20.0));
    ax.legend()
    ax.axvline(x=0, lw = 1, color = 'k', ls = '-')
    ax.set(xlim = [-2,2])
    x = np.linspace(-2,2,1000)
#     ax.plot(x,gaussian(x,m.max(),m_bin_edges[:-1:n][m.argmax()]))
#     popt, _ = curve_fit(gaussian, m_bin_edges[:-1:n], m, p0 = [m.max(),m_bin_edges[:-1:n][m.argmax()]])
#     ax.plot(x,gaussian(x,popt[0],popt[1]), color = 'r')
#     ax[1].hist(t_bin_edges[:-1], t_bin_edges, weights = dts_binned[i], alpha = 1, label = t_dict[i], color = cmap(i/20.0));
#     ax[1].set(xlim = (t_bin_chunk[i],t_bin_chunk[i+1]))

# +
cmap = plt.cm.cool
skip = 1
if skip > 1:
    fig, axes = plt.subplots((19//skip + 1),1,figsize = (15,3*(19//skip + 1)))
else:
    fig, axes = plt.subplots(19,1,figsize = (15,3*19))

stds = np.zeros(19)
for i, ax in enumerate(axes):
#     if skip > 1:
#         i = range(19)[::skip][i]
#     if i < 10:
#         n=1
#     elif i < 15:
#         n=2
#     else:
#         n=3
    n=1
    m,_,_=ax.hist(m_bin_edges[:-1], m_bin_edges[::n], weights = dms_binned[i], alpha = 1, density=True, label = t_dict[i], color = cmap(i/20.0));
    ax.legend()
    ax.axvline(x=0, lw = 1, color = 'k', ls = '-')
    ax.set(xlim = [-2,2], xlabel = 'mag')
    x = np.linspace(-2,2,1000)
#     ax.plot(x,gaussian(x,m.max(),m_bin_edges[:-1:n][m.argmax()]))
    #Also make sure that bins returned from .hist match m_bin_edges : it is
    try:
        popt, _ = curve_fit(gaussian, m_bin_edges[:-1:n], m, p0 = [m.max(),m_bin_edges[:-1:n][m.argmax()]])
        ax.plot(x,gaussian(x,popt[0],popt[1]), color = 'r')
        stds[i] = (2*np.pi)**-0.5*1/popt[0]
    except:
        pass
#     ax[1].hist(t_bin_edges[:-1], t_bin_edges, weights = dts_binned[i], alpha = 1, label = t_dict[i], color = cmap(i/20.0));
#     ax[1].set(xlim = (t_bin_chunk[i],t_bin_chunk[i+1]), xlabel = 'mjd')
plt.subplots_adjust(hspace=0.3)
plt.savefig('/disk1/hrb/python/analysis/qsos/plots/dtdm/dtdm_stacked_{}.pdf'.format(key),bbox_inches='tight')

# + active=""
# cmap = plt.cm.cool
# skip = 1
# if skip > 1:
#     fig, axes = plt.subplots((19//skip + 1),2,figsize = (15,3*(19//skip + 1)))
# else:
#     fig, axes = plt.subplots(19,2,figsize = (15,3*19))
#
# stds = np.zeros(19)
# for i, ax in enumerate(axes):
# #     if skip > 1:
# #         i = range(19)[::skip][i]
# #     if i < 10:
# #         n=1
# #     elif i < 15:
# #         n=2
# #     else:
# #         n=3
#     n=1
#     m,_,_=ax.hist(m_bin_edges[:-1], m_bin_edges[::n], weights = dms_binned[i], alpha = 1, density=True, label = t_dict[i], color = cmap(i/20.0));
#     ax.legend()
#     ax.axvline(x=0, lw = 1, color = 'k', ls = '-')
#     ax.set(xlim = [-2,2], xlabel = 'mag')
#     x = np.linspace(-2,2,1000)
# #     ax.plot(x,gaussian(x,m.max(),m_bin_edges[:-1:n][m.argmax()]))
#     #Also make sure that bins returned from .hist match m_bin_edges : it is
#     try:
#         popt, _ = curve_fit(gaussian, m_bin_edges[:-1:n], m, p0 = [m.max(),m_bin_edges[:-1:n][m.argmax()]])
#         ax.plot(x,gaussian(x,popt[0],popt[1]), color = 'r')
#         stds[i] = (2*np.pi)**-0.5*1/popt[0]
#     except:
#         pass
#     ax[1].hist(t_bin_edges[:-1], t_bin_edges, weights = dts_binned[i], alpha = 1, label = t_dict[i], color = cmap(i/20.0));
# #     ax[1].set(xlim = (t_bin_chunk[i],t_bin_chunk[i+1]), xlabel = 'mjd')
# plt.subplots_adjust(hspace=0.3)
# # plt.savefig('/disk1/hrb/python/analysis/qsos/plots/dtdm/dtdm_stacked.pdf',bbox_inches='tight')
# -

fig, ax = plt.subplots(1,1,figsize=(14,10))
for i in range(19)[::3]:
    ax.hist(m_bin_edges[:-1], m_bin_edges, weights = dms_binned[i], alpha = 0.3, density=True, label = t_dict[i], color = cmap(i/20.0));
    ax.legend()
    ax.axvline(x=0, lw = 0.1, color = 'k', ls = '-')
#     ax.set(yscale='log')


