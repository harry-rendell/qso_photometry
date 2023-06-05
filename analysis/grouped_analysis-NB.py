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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
from module.config import cfg
from module.preprocessing import colour_transform, parse, data_io, lightcurve_statistics

from matplotlib_venn import venn2, venn3, venn3_unweighted, venn3_circles

obj = 'qsos'
band = 'i'
sdss = pd.read_csv(cfg.D_DIR + 'surveys/sdss/{}/clean/{}_band/grouped.csv'.format(obj,band), index_col=0)
ps   = pd.read_csv(cfg.D_DIR + 'surveys/ps/{}/clean/{}_band/grouped.csv'.format(obj,band), index_col=0)
ztf  = pd.read_csv(cfg.D_DIR + 'surveys/ztf/{}/clean/{}_band/grouped.csv'.format(obj,band), index_col=0)
ssa  = pd.read_csv(cfg.D_DIR + 'surveys/supercosmos/{}/clean/{}_band/grouped.csv'.format(obj,band), index_col=0)
tot  = pd.read_csv(cfg.D_DIR + 'merged/{}/clean/grouped_{}.csv'.format(obj,band), index_col=0)

tot

# +
from module.plotting.plotting_common import savefigs
# ["#074f57", "#077187", "#74a57f", "#9ece9a", "#e4c5af"]
fig, ax = plt.subplots(1,1, figsize=(10,8))
total = len(sdss.index.union(ztf.index.union(ps.index)))

v1 = venn3_unweighted(
    [set(sdss.index), set(ps.index), set(ztf.index)],
    set_labels=['SDSS','PS','ZTF'],
    set_colors=["#74a57f", "#9ece9a", "#e4c5af"],
    subset_label_formatter=lambda x: f"{(x/total):1.0%}",
    ax=ax,
    alpha=1
)

venn3_circles([1]*7, ax=ax, lw=0.5)

savefigs(fig, 'SURVEY-DATA_venn_diagram', 'chap2')


# -

def savefig_paper(fig,imgname,dirname=None,dpi=100,noaxis=False):
    '''
    Save a low-res png and a high-res eps in one line.
    
    str imgname: image name without any extension or directory name
    Lifehack: use 
    savefig = partial(rz.savefig_paper,dirname=<dirname>)
    for easy one-line saving!
    '''
    if dirname is None:
        raise ValueError('Dirname not set up.') 
    kwargs ={'bbox_inches':'tight'}
    if noaxis:
        #https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image - Richard Yu Liu
        fig.subplots_adjust(0,0,1,1,0,0)
        for ax in fig.axes:
            ax.axis('off')
        kwargs['pad_inches'] = 0
    fig.savefig(dirname+imgname+'.png',dpi=100,**kwargs)
    fig.savefig(dirname+imgname+'.pdf',**kwargs)


# +
vac = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr12q/SDSS_DR12Q_BH_matched.csv', index_col='uid')
x = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/qsos_subsample_coords.csv', comment='#', usecols=['uid'],index_col='uid')
for name, survey in zip(['ssa','sdss','ps','ztf', 'vac'], [ssa, sdss, ps, ztf, vac]):
    x[name] = all_uids.index.isin(survey.index)

x.to_csv(cfg.D_DIR + 'catalogues/qsos/sets/clean_{}.csv'.format(band))
    
x[ x['vac'].values & np.any(x[['sdss','ps']].values, axis=1)]
