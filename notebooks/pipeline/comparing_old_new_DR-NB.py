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
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool
import sys
sys.path.append('../')
dpath = cfg.D_DIR + ''

# # SDSS
# ---

sdss_old = pd.read_csv(dpath + '/surveys/sdss/qsos/sdss_secondary_old.csv')
sdss_old

sdss_new = pd.read_csv(dpath + '/surveys/sdss/qsos/sdss_secondary.csv')
sdss_new.sort_values('uid', ascending=True)

sdss_melt = pd.read_csv(dpath + '/surveys/sdss/qsos/qsosSecondary.csv')

sdss_melt['get_nearby_distance'].describe()

# # PS
# ---



# # ZTF
# ---


