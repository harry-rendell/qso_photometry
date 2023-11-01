# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
# ---

# + language="bash"
# jupytext --to py dtdm_raw_analysis_basic_stats-NB.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb dtdm_raw_analysis_basic_stats-NB.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.classes.dtdm import dtdm_raw_analysis
from module.plotting.common import savefigs
