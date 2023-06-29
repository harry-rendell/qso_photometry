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

dtdm_qsos_lbol = dtdm_raw_analysis('qsos', 'uid', 'r', 'qsos')
dtdm_qsos_lbol.read_pooled_stats('log', key='Lbol')

dtdm_qsos_mbh = dtdm_raw_analysis('qsos', 'uid', 'r', 'qsos')
dtdm_qsos_mbh.read_pooled_stats('log', key='MBH')

dtdm_qsos_nedd = dtdm_raw_analysis('qsos', 'uids', 'r', 'qsos')
dtdm_qsos_nedd.read_pooled_stats('log', key='nEdd')

dtdm_qsos_lbol.pooled_stats.keys()

fig, ax = dtdm_qsos_lbol.plot_stats_property(['SF cwf a'], figax=None, xscale='log', yscale='linear', ylim=(0, 1), ylabel='Structure Function$^2$')

fig, ax = dtdm_qsos_mbh.plot_stats_property(['SF cwf a'], figax=None, xscale='log', yscale='linear', ylim=(0, 1), ylabel='Structure Function$^2$')

fig, ax = dtdm_qsos_nedd.plot_stats_property(['SF cwf a'], figax=None, xscale='log', yscale='linear', ylim=(0, 1), ylabel='Structure Function$^2$')
