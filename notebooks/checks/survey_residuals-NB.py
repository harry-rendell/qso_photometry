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

# + language="bash"
# jupytext --to py survey_residuals-NB.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb survey_residuals-NB.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.classes.grouped_analysis import SurveyComparison
from module.plotting.common import savefigs
from module.modelling.models import bkn_pow, bkn_pow_smooth
plt.style.use(cfg.FIG.STYLE_DIR + 'style.mplstyle')

SAVE_FIGS = True

# +
obj = 'qsos'
ID  = 'uid' if obj == 'qsos' else 'uid_s'
band = 'r'

sc_qr = SurveyComparison(obj, band)
sc_qr.read()
sc_qr.calculate_resdiuals_between_surveys()
sc_qr.residuals.describe()

# +
obj = 'calibStars'
ID  = 'uid' if obj == 'qsos' else 'uid_s'
band = 'r'

sc_sr = SurveyComparison(obj, band)
# -

sc_sr.plot_2d_residuals('sdss')


def plot_all_residuals_hist_all_bands(objects, range=(-0.5,0.5), figax=None, **kwargs):
    fig, axes = plt.subplots(3,3 , figsize=(10,11))
    for i, object in enumerate(objects):
        object.plot_all_residuals_hist(range=range, figax=(fig,axes[:,i]), **kwargs)
    plt.subplots_adjust(wspace=0.15)
    plt.subplots_adjust(hspace=0.35)
    
    handles = [b for a in axes[0,:] for b in a.get_legend_handles_labels()[0]]
    labels =  [b for a in axes[0,:] for b in a.get_legend_handles_labels()[1]]
    plt.figlegend(handles=handles, labels=labels, ncols=3, loc='upper center', bbox_to_anchor=(0.52, 0.94), columnspacing=9, frameon=False)
    return fig, axes


# star_objs = [SurveyComparison('calibStars', b) for b in 'gri']
fig, ax = plot_all_residuals_hist_all_bands(star_objs, range=(-1,1))
# if SAVE_FIGS:
    # savefigs(fig, 'colour_transf/RESIDUALS-calibStars', 'chap2')


star_objs = [SurveyComparison('calibStars', b) for b in 'gri']
fig, ax = plot_all_residuals_hist_all_bands(star_objs, range=(-1,1))
# if SAVE_FIGS:
    # savefigs(fig, 'colour_transf/RESIDUALS-calibStars', 'chap2')


qsos_objs = [SurveyComparison('qsos', b) for b in 'gri']
fig, ax = plot_all_residuals_hist_all_bands(qsos_objs, range=(-1,1))
if SAVE_FIGS:
    savefigs(fig, 'colour_transf/RESIDUALS-qsos', 'chap2')


