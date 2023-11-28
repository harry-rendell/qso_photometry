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
from module.preprocessing import data_io, parse, binning
from module.preprocessing.binning import construct_T_edges
from module.assets import load_grouped, load_grouped_tot, load_vac
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

obj = 'qsos'
ID  = 'uid' if obj == 'qsos' else 'uid_s'
band = 'r'

surveys = load_grouped(obj, 'gri')
tot = load_grouped_tot(obj, 'gri')
vac = load_vac(obj)

df = tot['r'][['mag_std']].join(vac['MBH'])

df1 = surveys['ztf']['r'][['mag_std']].join(vac['Lbol'])
df2 = surveys['ps']['r'][['mag_std']]


fig, ax = plt.subplots(1,1, figsize=(15,10))
sns.histplot(df1, x='mag_std', y='Lbol', log_scale=(True, False), ax=ax, binrange=((-2, 0), (42, 48)), color='orange')
sns.histplot(df2, x='mag_std', y='Lbol', log_scale=(True, False), ax=ax, binrange=((-2, 0), (42, 48)))

df1 = df1.rename(columns={'mag_std': 'mag_std_ztf'})
df2 = df2.rename(columns={'mag_std': 'mag_std_ps'})
df3 = df1.join(df2)

fig, ax = plt.subplots(1,1, figsize=(15,10))
sns.histplot(df3, x='mag_std_ztf', y='mag_std_ps', log_scale=(True, True), ax=ax, binrange=((-2, 0.5), (-2.5, 0.5)), color='red')
