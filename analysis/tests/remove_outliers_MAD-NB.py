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
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from funcs.analysis.analysis import *
# %matplotlib inline

obj = 'qsos'
ID  = 'uid'
band = 'r'
dr = analysis(ID, obj, band)
dr.read_in(redshift=False, cleaned=False, nrows=10000)
dr.group(read_in=True, survey='all', redshift=False, colors=False, restrict=True)


# # Clean lightcurves using MAD. Need to do this per filter

# ### Using rolling window to find outliers

# + active=""
# def check_outliers_(df, uids):
#     outliers = []
#     # for single band
#     MAD_max = pd.DataFrame(columns=['uid','MAD_max', 'MAD_argmax','mag','mjd'])
#     # make sure we have multiindexing
#     for uid in uids:
#         group = df.loc[uid].sort_values('mjd')
#         MAD_maxes = {'uid':uid}
#         
#         mjd_diff = np.diff(group['mjd'])
#         min_dt = np.array([mjd_diff[0]] + [min(mjd_diff[i],mjd_diff[i+1]) for i in range(len(mjd_diff)-1)] + [mjd_diff[-1]])
#         
#         MADS = abs(group['mag'] - group['mag'].rolling(6, center=True).median().fillna(method='bfill').fillna(method='ffill'))*np.exp(-min_dt)
#
#         for MAD in MADS[MADS>0.5]:
#             MAD_maxes['MAD_max'] = MAD
#             MAD_maxes['MAD_argmax'] = MAD.argmax()
#             MAD_maxes['mjd'] = group['mjd'].values[MAD.argmax()]
#             MAD_maxes['mag'] = group['mag'].values[MAD.argmax()]
#             MAD_max = MAD_max.append(MAD_maxes, ignore_index=True)
#
#     return MAD_max.astype({'uid':np.uint32}).set_index('uid')
# -

def calculate_MAD(group):    
    mjd_diff = np.diff(group['mjd'])
    min_dt = np.array([mjd_diff[0]] + [min(mjd_diff[i],mjd_diff[i+1]) for i in range(len(mjd_diff)-1)] + [mjd_diff[-1]])
    MAD = abs(group['mag'] - group['mag'].rolling(6, center=True).median().fillna(method='bfill').fillna(method='ffill'))*np.exp(-min_dt*3)
    
    return MAD


dr.df['MAD']=dr.df.groupby(by='uid').apply(calculate_MAD).reset_index(level=0, drop=True)

dr.df

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax.hist(dr.df['MAD'], bins=100)
ax.set(ylim=[0,1000])

dr.df.loc[342201]

# +
# uids = dr.df.sample(30, weights='MAD', random_state=42).index
# -

ax = dr.plot_series([68], filtercodes='r', show_outliers=True, xlim=[58215,58230])

dr.coords.loc[68].values

dr.df.loc[68]

ax = dr.plot_series(uids, filtercodes='r', show_outliers=True)#, xlim=[58215,58230])

# + active=""
# ax = dr.plot_series(uids, filtercodes='r', show_outliers=True)#, xlim=[58215,58230])

# + active=""
# 1000  : 9min 20s
# 5000  : 8.37s
# 10000 : 
# -

dr.summary()

# # %%timeit -n 1 -r 1
MAD_max = check_outliers(dr.df, uids[:50])

test = check_outliers(dr.df, dr.idx_uid[:100])['MAD_max'].sort_values(ascending=False)

test

dr.plot_series(test.index[:10], filtercodes='r')

fig, ax = plt.subplots(1,1, figsize = (20,10))
check_outliers(dr.df, uids[:100])['MAD_max'].hist(bins=20, ax=ax, alpha=0.5)
check_outliers(dr.df, dr.idx_uid[:100])['MAD_max'].hist(bins=20, ax=ax, alpha=0.5)

MAD_max = check_outliers(df_ztf, uids)


# +
# uids2 = MAD_max.sort_values(['g','r'], ascending=False).head(20)['uid'].values
# -

def check_outliers_plot(df, uids):
    MAD_max = pd.DataFrame(columns=['uid','g','r','i','z','y'])
    # make sure we have multiindexing
    colors = {'u':'m', 'g':'g', 'r':'r','i':'k','z':'b'}
    fig, axes = plt.subplots(20,1, figsize = (30,70))
    for uid, ax in zip(uids, axes.ravel()):
        group_all = df.loc[uid].sort_values('mjd')
        MAD_maxes = {'uid':uid}
        append = False
        for band in group_all.index.unique():
            group = group_all.loc[band]
            ax.scatter(group['mjd'], group['mag'], label=band, s=0.6, c=colors[band])
            ax.errorbar(group['mjd'], group['mag'], yerr=group['magerr'], lw=0.5, color=colors[band])
            try:
                n = len(group)
                MAD = abs(group['mag'] - group['mag'].rolling(n//2, center=True).median().fillna(method='bfill').fillna(method='ffill'))
                if MAD.max() > 0:
                    MAD_maxes[band] = MAD.max()
                    append = True
                outlier = group[MAD>1]
                ax.scatter(outlier['mjd'], outlier['mag'], marker='*', s=100, c='r')
            except:
                print('pass')
                pass
        
        if append:
            MAD_max = MAD_max.append(MAD_maxes, ignore_index=True)

        ax.legend()
    return MAD_max.astype({'uid':np.uint32})


MAD_max = check_outliers_plot(df, uids)

dr.plot_series([346031,346031])
