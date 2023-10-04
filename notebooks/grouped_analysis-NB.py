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
from module.preprocessing import color_transform, parse, data_io, lightcurve_statistics, binning
# from matplotlib_venn import venn2, venn3, venn3_unweighted, venn3_circles

obj = 'qsos'
ID  = 'uid' if obj == 'qsos' else 'uid_s'
band = 'r'

redshift = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr14q/dr14q_redshift.csv').set_index(ID)
sdss = {band:pd.read_csv(cfg.D_DIR + 'surveys/sdss/{}/clean/{}_band/grouped.csv'.format(obj,band)       , index_col=0) for band in 'gri'}
ps   = {band:pd.read_csv(cfg.D_DIR + 'surveys/ps/{}/clean/{}_band/grouped.csv'.format(obj,band)         , index_col=0) for band in 'gri'}
ztf  = {band:pd.read_csv(cfg.D_DIR + 'surveys/ztf/{}/clean/{}_band/grouped.csv'.format(obj,band)        , index_col=0) for band in 'gri'}
ssa  = {band:pd.read_csv(cfg.D_DIR + 'surveys/ssa/{}/clean/{}_band/grouped.csv'.format(obj,band), index_col=0) for band in 'gri'}
tot  = {band:pd.read_csv(cfg.D_DIR + 'merged/{}/clean/grouped_{}.csv'.format(obj,band)                  , index_col=0) for band in 'gri'}
surveys = {'ssa':ssa, 'sdss':sdss, 'ps':ps, 'ztf':ztf, 'tot':tot}


# columns of interest
# coi = ['n_tot','mjd_min','mjd_max','mag_min','mag_max']
coi = ['n_tot']
dtypes = {col:dtype for col, dtype in cfg.PREPROC.stats_dtypes.items() if col in coi}
for survey_name, surv in surveys.items():
    print('_'*40+survey_name+'_'*40)
    for band in 'gri':
        print(' '*20 + '_'*20+band+'_'*20 + ' '*20)
        summary = surv[band][coi].describe()
        for x in summary.values.flatten():
            print('{:.2f}'.format(x))
        # print(summary.to_latex(caption='<caption>',label='<label>', float_format="%.2f"))


CREATE_SETS = False
if CREATE_SETS:
    # Cell for creating sets to track which objects are in which surveys.
    # Could put in a script.
    for obj in ['qsos','calibStars']:
        ID  = 'uid' if obj == 'qsos' else 'uid_s'
        for band in 'gri':
            x = pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/{obj}_subsample_coords.csv', comment='#', usecols=[ID],index_col=ID)
            sdss = pd.read_csv(cfg.D_DIR + f'surveys/sdss/{obj}/clean/{band}_band/grouped.csv', index_col=0)
            ps   = pd.read_csv(cfg.D_DIR + f'surveys/ps/{obj}/clean/{band}_band/grouped.csv', index_col=0)
            ztf  = pd.read_csv(cfg.D_DIR + f'surveys/ztf/{obj}/clean/{band}_band/grouped.csv', index_col=0)
            ssa  = pd.read_csv(cfg.D_DIR + f'surveys/ssa/{obj}/clean/{band}_band/grouped.csv', index_col=0)
            if obj == 'qsos':
                vac = pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/dr12q/SDSS_DR12Q_BH_matched.csv', index_col=ID)        
                for name, survey in zip(['ssa','sdss','ps','ztf', 'vac'], [ssa, sdss, ps, ztf, vac]):
                    x[name] = x.index.isin(survey.index)
            elif obj == 'calibStars':
                for name, survey in zip(['ssa','sdss','ps','ztf'], [ssa, sdss, ps, ztf]):
                    x[name] = x.index.isin(survey.index)

            print(x)

            # x.to_csv(cfg.D_DIR + f'catalogues/{obj}/sets/clean_{band}.csv')
            comment = ( "# This file shows which qsos appear in which surveys.\n"
                        "# Note, it uses data from cleaned/grouped.csv for each survey\n"
                        "# vac (if present) is DR12Q value added catalogue (SDSS_DR12Q_BH_matched.csv)" )
            data_io.to_csv(x, cfg.D_DIR + f'catalogues/{obj}/sets/clean_{band}.csv', comment=comment)
            # x[ x['vac'].values & np.any(x[['sdss','ps']].values, axis=1)]

CREATE_BOUNDS_DICT = False
if CREATE_BOUNDS_DICT:
    # Cell to calculate max dt to put in config.py for extract_features_from_dtdm_group.py.
    # Could put in a script.
    obj = 'qsos'
    ID = 'uid'
    dic1 = {}
    dic2 = {}

    vac = pd.read_csv(os.path.join(cfg.D_DIR,'catalogues/qsos/dr16q/dr16q_vac_shen_matched.csv'), index_col=ID)
    vac = vac.rename(columns={'z':'redshift_vac'});
    vac = parse.filter_data(vac, cfg.PREPROC.VAC_BOUNDS, dropna=False)
    bounds_z = np.array([-3.5,-1.5,-1,-0.5,0,0.5,1,1.5,3.5])
    # Note, in dr16q, bad nEdd entries are set to 0 (exactly) so we can remove those.
    vac['nEdd'] = vac['nEdd'].where((vac['nEdd']!=0).values)

    for property_ in ['Lbol','MBH','nEdd']:
        for band in 'gri':
            tot  = pd.read_csv(cfg.D_DIR + 'merged/{}/clean/grouped_{}.csv'.format(obj,band), index_col=0)
            tot = tot.join(redshift, how='left')
            tot['mjd_ptp_rf'] = tot['mjd_ptp']/(1+tot['z'])
            
            sets = pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/sets/clean_{band}.csv', comment='#', index_col=ID)
            uids = sets[sets['vac'].values & np.any(sets[['sdss','ps']].values, axis=1)].index
            tot = tot[tot.index.isin(uids)]
            vac = vac.dropna(subset=[property_])
            tot = tot.join(vac)

            groups, bounds_values = binning.calculate_groups(vac[property_], bounds = bounds_z)
            max_dt_of_group = []
            for group in groups:
                max_dt_of_group.append(int(np.ceil(tot.loc[tot.index.isin(group),'mjd_ptp_rf'].max())))
            dic1[band] = max_dt_of_group
        dic2[property_] = dic1
            # print('rest frame:')
            # print('max ∆t rest frame:',tot['mjd_ptp_rf'].max())
            # print('obs frame')
            # print('max ∆t obs frame:',tot['mjd_ptp'].max())
            # dic1[band] = int(np.ceil(tot['mjd_ptp_rf'].max()))
            # dic2[band] = int(np.ceil(tot['mjd_ptp'].max()))
    dic2

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

savefigs(fig, 'SURVEY-DATA-venn_diagram', 'chap2')


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
