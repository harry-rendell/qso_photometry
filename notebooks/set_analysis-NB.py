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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
from module.config import cfg
from module.preprocessing import color_transform, parse, data_io, lightcurve_statistics, binning

obj = 'qsos'
band = 'r'
ID = 'uid' if obj == 'qsos' else 'uid_s'
sets = pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/sets/clean_{band}.csv', comment='#', index_col=ID)

sets

sets['vac'].sum()


