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
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
# from module.preprocessing import parse, data_io, binning
# from module.classes.dtdm import dtdm_raw_analysis
# from module.classes.analysis import analysis
# from module.plotting.common import savefigs
from module.classes.dtdm_binned import dtdm_binned_class

qr = dtdm_binned_class('qsos', 'r', 'qsos', 'label', verbose=True)

sr = dtdm_binned_class('calibStars', 'r', 'qsos', 'label', verbose=True)

# +
plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')

fig, ax, _ = qr.hist_dm(1.5, alpha = 0.5, cmap=plt.cm.cool)

# +
plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')

fig, ax, _ = qr.hist_dm(1, alpha = 0.5, cmap=plt.cm.cool)
sr.hist_dm(1, figax=(fig, ax), alpha = 0.5, cmap=plt.cm.autumn)

# -


