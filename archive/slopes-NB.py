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

# # Notebook for calculating the average slope for a lightcurve, and plotting its distribution

def slope(group):
    """
    finds slope m in line y=mx+c
    """

    group        = group.dropna()
    mag          = group['mag']
    magerr_invsq = group['magerr']**-2
    t            = group['mjd']

    t_optimal = (t*magerr_invsq).sum()/magerr_invsq.sum()
    m_optimal = (mag*(t-t_optimal)*magerr_invsq).sum()/((t-t_optimal)**2*magerr_invsq).sum()

    m_regular = ((t-t.mean())*(mag-mag.mean())).sum()/((t-t.mean())**2).sum()

    return {'m_optimal':m_optimal,'m_regular':m_regular}


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
# from profilestats import profile
from scipy.stats import binned_statistic
from module.analysis import *
from os import listdir
import os
data_path = '/disk1/hrb/python/'
import time


def reader(n_subarray):
    return pd.read_csv(wdir+'data/merged/qsos/{}_band/lc_{}.csv'.format(band,n_subarray), nrows=None, index_col = 'uid', dtype = {'catalogue': np.uint8, 'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, 'uid': np.uint32})


band = 'r'
ID = 'uid'
dr = analysis(ID)
dr.read_in(reader)

# +
# boolean = (dr.df['catalogue']==1)
# boolean = (df['cat']==6) | (df['cat']==10)
# boolean = (df['cat']==9) | (df['cat']==11)
# -

# result_rf = dr.df[boolean].groupby('uid').apply(slope).apply(pd.Series)
result_obs = dr.df.groupby('uid').apply(slope).apply(pd.Series)

result_obs.to_csv(wdir + 'data/computed/qsos/slopes/slopes_obsframe.csv')

result_obs = pd.read_csv('../data/computed/qsos/slopes/slopes_obsframe.csv', index_col='uid')
result_obs.columns

# +
fig, ax = plt.subplots(1,1,figsize=(20,10))
result_obs['m_optimal'].hist(bins=100, range=(-0.01,0.01), ax=ax)

ax.set(yscale='log')

# -

result_obs = pd.read_csv('../data/computed/qsos/slopes/slopes_obsframe_all.csv', index_col='uid')
result_obs = result_obs[abs(result_obs['m_optimal'])<0.5]
print('All surveys (observer frame)\n')
print('optimal slope mean:   {:.9f} mag/decade\nregular slope mean:   {:.9f} mag/decade'.format(*result_obs.mean()*3650))
print('optimal slope median: {:.9f} mag/decade\nregular slope median: {:.9f} mag/decade'.format(*result_obs.median()*3650))


result_obs_ztf = pd.read_csv('../data/computed/qsos/slopes/slopes_obsframe_all.csv', index_col='uid')
result_obs_ztf = result_obs_ztf[abs(result_obs_ztf['m_optimal'])<0.5]
print('ZTF (observer frame)\n')
print('optimal slope mean:   {:.9f} mag/decade\nregular slope mean:   {:.9f} mag/decade'.format(*result_obs_ztf.mean()*3650))
print('optimal slope median: {:.9f} mag/decade\nregular slope median: {:.9f} mag/decade'.format(*result_obs_ztf.median()*3650))


result_rf_ztf = pd.read_csv('../data/computed/qsos/slopes/slopes_restframe_ztf.csv', index_col='uid')
result_rf_ztf = result_rf_ztf[abs(result_rf_ztf['m_optimal'])<0.5]
print('ZTF (rest frame)\n')
print('optimal slope mean:   {:.9f} mag/decade\nregular slope mean:   {:.9f} mag/decade'.format(*result_rf_ztf.mean()*3650))
print('optimal slope median: {:.9f} mag/decade\nregular slope median: {:.9f} mag/decade'.format(*result_rf_ztf.median()*3650))


result_rf = pd.read_csv('../data/computed/qsos/slopes/slopes_restframe_all.csv', index_col='uid')
result_rf = result_rf[abs(result_rf['m_optimal'])<0.5]
print('All surveys (rest frame)\n')
print('optimal slope mean:   {:.9f} mag/decade\nregular slope mean:   {:.9f} mag/decade'.format(*result_rf.mean()*3650))
print('optimal slope median: {:.9f} mag/decade\nregular slope median: {:.9f} mag/decade'.format(*result_rf.median()*3650))


result_rf_ztf.sort_values('m_optimal').head(20).index



fig, ax = plt.subplots(1,2,figsize=(20,10))
range=(-1,1)
bins=201
# (result_obs    ['m_optimal']*3650).hist(range=range, ax=ax[0], bins=bins, alpha=0.5, label='observers frame')
# (result_obs_ztf['m_optimal']*3650).hist(range=range, ax=ax[0], bins=bins, alpha=0.5, label='observers frame (ztf)')
(result_rf     ['m_optimal']*3650).hist(range=range, ax=ax[1], bins=bins, alpha=0.5, label='rest frame')
# (result_rf_ztf ['m_optimal']*3650).hist(range=range, ax=ax[1], bins=bins, alpha=0.5, label='rest frame (ztf)')
for axis in ax:
    axis.set(xlabel='mag/decade')
    axis.legend()
    axis.axvline(x=0, color='k', ls='--', lw=0.5)
plt.suptitle('distribution of slopes');

fig, ax = plt.subplots(1,1,figsize=(10,8))
abs(result['m_optimal']).hist(bins=200, ax=ax, range=(0,2))
ax.set(yscale='log',ylim=[0,1000000], xlim=[0,2])


