import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from ..config import cfg

def plot_series(df, uids, sid=None, bands='gri', grouped=None, **kwargs):
    """
    Simple plotting function for lightcurves
    """

    plt_color = {'u':'m', 'g':'g', 'r':'r', 'i':'k', 'z':'b'}
    marker_dict = {3:'v', 5:'D', 7:'s', 11:'o'}
    survey_dict = {3: 'SSS', 5:'SDSS', 7:'PS1', 11:'ZTF'}

    if np.issubdtype(type(uids),np.integer): uids = [uids]
    fig, axes = plt.subplots(len(uids),1,figsize = (20,3*len(uids)), sharex=True)
    
    if len(uids)==1:
        axes=[axes]
    for uid, ax in zip(uids,axes):
        single_obj = df.loc[uid].sort_values('mjd')
        for band in bands:
            single_band = single_obj[single_obj['band']==band]
            if sid is not None:
                # Restrict data to a single survey
                single_band = single_band[single_band['sid']==sid]
            for sid_ in single_band['sid'].unique():
                x = single_band[single_band['sid']==sid_]
                ax.errorbar(x['mjd'], x['mag'], yerr = x['magerr'], lw = 0.5, markersize = 3, marker = marker_dict[sid_], label = survey_dict[sid_]+' '+band, color = plt_color[band])

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
    return fig, axes

def savefigs(fig, imgname, dirname, dpi=100, noaxis=False, **kwargs):
    """
    Save a low-res png and high-res pdf for fast compiling of thesis

    Example: savefigs(fig, 'survey/SURVEY-DATA-venn_diagram', 'chap2')
    or
    Example: savefigs(fig, 'SURVEY-DATA-venn_diagram', 'chap2')

    Parameters
    ----------
    fig : matplotlib figure handle
    imgname : str
        name for plot without extension. Prefix with a folder to create a subdirectory.
    dirname : str
        Absolute or relative (to cfg.FIG.THESIS_PLOT_DIR) directory for saving.
        Creates directories if output path does not exist.
    dpi : int
    """
    
    # Remove extension if user has accidentally provided one
    imgname = imgname.split('.')[0]

    if not os.path.exists(dirname):
        # If we provide a relative path, assume that we're plotting straight to our overleaf project
        dirname = os.path.join(cfg.THESIS_DIR, dirname)
        assert os.path.exists(dirname), f"Relative path {dirname} does not exist"
        pdf_path = os.path.join(dirname, 'graphics')
        png_path = os.path.join(dirname, 'draft_graphics')
    else:
        # If we provide an absolute path that exists, save both png and pdf to the same provided directory
        pdf_path = dirname
        png_path = dirname

    if noaxis:
        #https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image - Richard Yu Liu
        fig.subplots_adjust(0,0,1,1,0,0)
        for ax in fig.axes:
            ax.axis('off')
        kwargs['pad_inches'] = 0

    kwargs['bbox_inches'] = 'tight'
    if "/" in imgname:
        # create subdirectories if they don't exist
        os.makedirs(os.path.join(png_path, os.path.dirname(imgname)), exist_ok=True)
        os.makedirs(os.path.join(pdf_path, os.path.dirname(imgname)), exist_ok=True)

    fig.savefig(os.path.join(png_path,imgname)+'.png',dpi=dpi, **kwargs)
    fig.savefig(os.path.join(pdf_path,imgname)+'.pdf', **kwargs)

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