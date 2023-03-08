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

# + jupyter={"outputs_hidden": true}
color_transf = pd.read_csv('color_transf_coef_to_ps.txt',index_col=0)
color_transf

# +
dtype1 = {'uid': np.uint32, 'objID': np.uint64, 'get_nearby_distance': np.float64, 'mjd_r': np.float64}
dtype2 = {band + 'psf'   : np.float32 for band in 'ugriz'}
dtype3 = {band + 'psferr': np.float32 for band in 'ugriz'}
cols = ['uid','objID', 'get_nearby_distance', 'mjd_r'] + [b + 'psf' for b in 'ugriz'] + [b + 'psferr' for b in 'ugriz']
df_sdss = pd.read_csv('/disk1/hrb/python/data/surveys/sdss/raw_sdss_secondary.csv', usecols = cols, index_col = 0, dtype = {**dtype1, **dtype2, **dtype3})

# Rename columns
df_sdss = df_sdss.rename(columns = {'mjd_r': 'mjd'})
# -

x = df_sdss['gpsf'] - df_sdss['ipsf']
for band in 'griz':
    a0, a1, a2, a3 = color_transf.loc[band].values
    # Convert to SDSS AB mags
    df_sdss[band+'psf_ps'] = df_sdss[band+'psf'] - a0 + a1*x + a2*(x**2) + a3*(x**3) #remove _ps to overwrite

df_sdss.head()

fig, axes = plt.subplots(2,2,figsize = (15,10));
for ax, band in zip(axes.ravel(),'griz'):
    df_sdss[band+'psf'].hist(range = (15,25),bins = 250, alpha = 0.5, ax = ax, label = 'sdss')
    df_sdss[band+'psf_ps'].hist(range = (15,25),bins = 250, alpha = 0.5, ax = ax, label = 'ps')
    ax.legend()
    ax.set(xlabel = band)
