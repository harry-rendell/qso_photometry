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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.preprocessing import parse, data_io, binning
from module.classes.dtdm import dtdm_raw_analysis
from module.classes.analysis import analysis
from module.plotting.common import savefigs

i=50

# +
# STARS
dsr = dtdm_raw_analysis('calibStars', 'r', 'calibStars')
dsr.read_pooled_stats('log', key='all')
dsr.read(i=i)

fig, ax = plt.subplots(1,1,figsize=(10,6))
dsr.df.hist('de', ax=ax, bins=300, range=(-5,5))
ax.set(yscale='log')

# +
# STARS
dsg = dtdm_raw_analysis('calibStars', 'g', 'calibStars')
dsg.read_pooled_stats('log', key='all')
dsg.read(i=3)

fig, ax = plt.subplots(1,1,figsize=(10,6))
dsg.df.hist('dm', ax=ax, bins=200, range=(-5,5), alpha=0.5)
ax.set(yscale='log')

# QSOS
dqg = dtdm_raw_analysis('qsos', 'g', 'qsos')
dqg.read_pooled_stats('log', key='all')
dqg.read(i=3)

dqg.df.hist('dm', ax=ax, bins=200, range=(-5,5), alpha=0.5)
ax.set(yscale='log')
# -

# QSOS
dqr = dtdm_raw_analysis('qsos', 'r', 'qsos')
dqr.read_pooled_stats('log', key='all')
dqr.read(i=3)
# fig, ax = plt.subplots(1,1,figsize=(10,6))
# dqr.df.hist('dm', ax=ax, bins=200, range=(-5,5), alpha=0.5)
# ax.set(yscale='log')

np.histogram(dqr.df['dt'].values,bins=200)

dqg.df


