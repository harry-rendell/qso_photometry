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

import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
from module.analysis.analysis import *
# %matplotlib inline

# +
obj = 'qsos'
ID  = 'uid'
band = 'r'
redshift_bool = True

# obj = 'calibStars'
# ID  = 'uid_s'
# band = 'r'
# redshift_bool = False

# Here we load the analysis class. This has various attibutes and methods outlined in /module/analysis.py
# Examples:
# Photometry is in dr.df

# Grouped statistics is in dr.grouped
# DR12 VAC properties are in dr.properties
dr = analysis(ID, obj, band)
# -

dr.read_in(redshift=redshift_bool)
dr.group(keys = ['uid'],read_in=True, redshift=redshift_bool, survey = 'all')

dr.group(keys = ['uid'],read_in=True, redshift=redshift_bool, survey='SSS')

dr.df['redshift'].hist(bins=100);
plt.figure() 
plt.hist(dr.df_grouped['mag_mean'], bins=200);

dr.df_grouped['mag_mean'].max()

fig, ax = dr.plot_series([2], filtercodes='r')
ax[0].set(xlim=[58300, 58400])
plt.xlabel('mjd',fontsize=20)
plt.ylabel('mag',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks([22,21.5,21,20.5],fontsize=18)



# +
# # Save a subset of object above a threshold magnitude limit
# threshold_mag = 19
# (dr.df['mag']<threshold_mag).sum()
# dr.df['mag'].max()
# np.savetxt('../data/computed/{}/binned/bright_{}/mag_lt_{}_uid.txt'.format(obj, threshold_mag, threshold_mag), dr.df_grouped[dr.df_grouped['mag_mean']<threshold_mag].index, fmt='%i')
# -

subdf = dr.df[dr.df['catalogue']==5]
subdf['mjd'].describe()

# +
# Seeing how our magnitude error distribution changes for different time range bins

# for mjd1,mjd2 in [(58100,58200),(58200,58300),(58300,58400),(58400,58500),(58500,58600),(58600,58700)]:
fig, ax = plt.subplots(1,1,figsize=(10,5))
# for mjd1,mjd2 in [(51000,51500),(51500,52000),(52000,52500),(52500,53000),(53000,53500),(53500,54000),(54000,54500),(54500,55000)]:
for mjd1,mjd2 in [(51000,52000),(52000,53000),(53000,54000),(54000,55000)]:
    subdf[((mjd1<subdf['mjd']) & (subdf['mjd']<mjd2))].hist('magerr',bins=100, 
                                                            label='{} < ∆t < {}'.format(mjd1,mjd2),
                                                            ax=ax,
                                                            alpha=0.5,
                                                            range=(0,0.25),
                                                            density=True)
ax.legend()

