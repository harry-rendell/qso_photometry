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

dtdm_qsos= dtdm_raw_analysis('qsos', 'uid', 'r', 'qsos')
dtdm_qsos.read_pooled_stats('log', key=None)

dtdm_qsos.pooled_stats.keys()

fig, ax = dtdm_qsos.plot_stats(['SF cwf p','SF cwf n'], figax=None, xscale='log', yscale='linear', ylim=(-0.3 , 1), ylabel='Structure Function$^2$')


