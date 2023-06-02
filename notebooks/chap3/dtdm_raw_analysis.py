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
from module.preprocessing import parse, data_io
from module.classes.dtdm import dtdm_raw_analysis

OBJ    = 'qsos'
ID     = 'uid' if OBJ == 'qsos' else 'uid_s'
BAND   = 'r'

q = dtdm_raw_analysis(OBJ, ID, BAND, 'qsos')
# dtdm_qsos_lbol.calculate_stats_looped_key(26, 'log', 'Lbol', save=True)

plt.hist(a['dm'].head(1000000), bins=1000);

a


