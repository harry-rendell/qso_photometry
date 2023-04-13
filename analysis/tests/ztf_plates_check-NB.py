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

# # Notebook for looking at the longest ∆t observations (between SSA and ZTF)

import pandas as pd 
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'size':16})
import matplotlib.cm as cmap
# from profilestats import profile
from os import listdir, path
from time import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
from funcs.preprocessing.dtdm import dtdm_raw_analysis

dtdm_star = dtdm_raw_analysis('calibStars','uid_s','r')

df = pd.DataFrame(columns=['uid_s','dt','dm','de','dm2_de2','cat']).astype({'cat':'uint8'}).set_index('uid_s')

# +
# # writing
# for i in range(25):
#     dtdm_star.read(i)
#     subdf = dtdm_star.df[dtdm_star.df['cat']==3]
#     df = df.append(subdf)
# df.to_csv('../data/computed/calibStars/dtdm/dtdm_ztf_ps.csv')
# -

# reading
df = pd.read_csv('../data/computed/calibStars/dtdm/dtdm_ztf_ps.csv', index_col=0)

df['dt'].hist(bins=150)

fig, ax = plt.subplots(1,1, figsize=(20,10))
df['dm'].hist(bins=150,ax=ax)

df2 = df[df['dt']>9537]

fig, ax = plt.subplots(1,1, figsize=(20,10))
df1['dm'].hist(bins=150,ax=ax,range=(-1,1))

df1['dm'].mean()

fig, ax = plt.subplots(1,1, figsize=(20,10))
df2['dm'].hist(bins=150,ax=ax,range=(-1,1))

df2['dm'].mean()

fig, ax = plt.subplots(1,1, figsize=(20,10))
df['dm2_de2'].hist(bins=150,ax=ax)

fig, ax = plt.subplots(1,1, figsize=(20,10))
df['dm2_de2'].hist(bins=150,ax=ax)
ax.set(yscale='log')

len(df[df['dm2_de2']>5])/len(df)*100

df['dm'].mean()
