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
from module.preprocessing import parse, data_io, binning
from module.classes.dtdm import dtdm_raw_analysis
from module.classes.analysis import analysis
from module.plotting.common import savefigs

config = {'obj':'qsos','ID':'uid'  ,'t_max':6751,'n_bins_t':200,'n_bins_m':200, 'n_bins_m2':248, 'n_t_chunk':19, 'width':2, 'steepness':0.005, 'leftmost_bin':-0.244}


OBJ    = 'qsos'
ID     = 'uid' if OBJ == 'qsos' else 'uid_s'
BAND   = 'r'
q = dtdm_raw_analysis(OBJ, ID, BAND, 'qsos')
# dtdm_qsos_lbol.calculate_stats_looped_key(26, 'log', 'Lbol', save=True)

q.calculate_stats_looped(n_chunks=4, log_or_lin='log', max_t=cfg.PREPROC.MAX_DT_REST_FRAME[OBJ][BAND])

fig, ax = plt.subplots(2,2)
q.plot_stats('',(fig,ax))


