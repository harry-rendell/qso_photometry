# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: astro
#     language: python
#     name: python3
# ---

# %%
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

# %%
band = 'i'
OBJ = 'qsos'
survey = 'ztf'
ID = 'uid' if OBJ == 'qsos' else 'uid_s'
nrows = None 
kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
            'nrows': nrows,
            'ID':ID,
            'basepath': cfg.D_DIR + f'surveys/{survey}/{OBJ}/clean/{band}_band/',
            'usecols':[ID,'mjd','mag','magerr']}

df = data_io.dispatch_reader(kwargs, multiproc=True)

# %%
a = df.sort_values('magerr')
