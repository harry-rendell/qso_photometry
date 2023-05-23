import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from ..config import cfg

def plot_series(df, uids, axes=None, grouped=None, **kwargs):
    """
    Simple plotting function for lightcurves
    """
    if np.issubdtype(type(uids),np.integer): uids = [uids]
    if axes is None:
        fig, axes = plt.subplots(len(uids),1,figsize = (25,3*len(uids)), sharex=True)
    if len(uids)==1:
        axes=[axes]
    for uid, ax in zip(uids,axes):
        x = df.loc[uid]
        ax.errorbar(x=x['mjd'], y=x['mag'], yerr=x['magerr'], lw = 0.5, markersize = 3)
        ax.scatter(x['mjd'], x['mag'], s=10)
        ax.invert_yaxis()
        ax.set(xlabel='MJD', ylabel='mag', **kwargs)
        ax.text(0.02, 0.9, 'uid: {}'.format(uid), transform=ax.transAxes, fontsize=10)

        if grouped is not None:
            y = grouped.loc[uid]
            mask = ( abs(x['mag']-y['mag_med']) > 3*y['mag_std'] ).values
            ax.scatter(x[mask]['mjd'], x[mask]['mag'], color='r', s=30)

    if len(uids)==1:
        axes=axes[0]
        
    plt.subplots_adjust(hspace=0)
    return axes

def savefigs(fig, imgname, dirname, dpi=100, noaxis=False, **kwargs):
    """
    Save a low-res png and high-res pdf for fast compiling of thesis

    Parameters
    ----------
    fig : matplotlib figure handle
    imgname : str
        name for plot without extension
    dirname : str
        Absolute or relative (to cfg.FIG.THESIS_PLOT_DIR) directory for saving.
        Creates directories if output path does not exist.
    dpi : int
    """
    
    # Remove extension if user has accidentally provided one
    imgname = imgname.split('.')[0]

    if not os.path.exists(dirname):
        dirname = os.path.join(cfg.FIG.THESIS_PLOT_DIR, dirname)
    os.makedirs(dirname, exist_ok=True)

    if noaxis:
        #https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image - Richard Yu Liu
        fig.subplots_adjust(0,0,1,1,0,0)
        for ax in fig.axes:
            ax.axis('off')
        kwargs['pad_inches'] = 0

    kwargs['bbox_inches'] = 'tight'
    fig.savefig(os.path.join(dirname,imgname)+'.png',dpi=100, **kwargs)
    fig.savefig(os.path.join(dirname,imgname)+'.pdf', **kwargs)

# from  matplotlib import rcParams
# rcParams['pdf.fonttype'] = 42
# rcParams['font.family'] = 'serif'
# rcParams['figure.facecolor'] = 'w'
# defcolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# #Is this for a presentation?
# isPrezi=False
# SMALL_SIZE = 16
# MEDIUM_SIZE = 22
# BIGGER_SIZE = 28
# VERY_SMALL=12
# if isPrezi:
#     plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
#     plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# else:
#     plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
#     plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes

# plt.rc('legend', fontsize=VERY_SMALL)    # legend fontsize