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

# + language="bash"
# jupytext --to py dtdm_distribution-NB.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb dtdm_distribution-NB.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

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

qr = dtdm_binned_class('qsos', 'i', 'Quasars', 'label', subset='all', verbose=True)

sr = dtdm_binned_class('calibStars', 'i', 'Stars', 'label', subset='all', verbose=True)

qr_inner = dtdm_binned_class('qsos', 'i', 'Quasars (inner)', 'label', subset='inner', verbose=True)

sr_inner = dtdm_binned_class('calibStars', 'r', 'Stars (inner)', 'label', subset='inner', verbose=True)

a = {f'{b[0]}-{b[1]}':a[0]*a[1] for a,b in zip(combinations_with_replacement([3,5,7,11],2),combinations_with_replacement(['ssa','sdss','ps','ztf'],2 ))}

for name, index in a.items():
    print(name, sr_inner.dcs_binned[0][index])

fig, ax, _ = sr_inner.hist_dm(1, alpha = 0.5, cmap=plt.cm.cool, overlay_exponential=True, overlay_lorentzian=True, overlay_gaussian=False)

# +
plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')

fig, ax, _ = qr.hist_dm(1.5, alpha = 0.5, cmap=plt.cm.cool)

# +
plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')

fig, ax, _ = qr.hist_dm(1, alpha = 0.5, cmap=plt.cm.cool)
# sr.hist_dm(1, figax=(fig, ax), alpha = 0.5, cmap=plt.cm.autumn)

# -

fig, ax, _ = qr_inner.hist_dm(1, alpha = 0.5, cmap=plt.cm.cool)
