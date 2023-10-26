from .config import cfg
import pandas as pd
import os
from .preprocessing import parse

def load_grouped(obj, bands=None, return_dict=True, **kwargs):
    """
    Load grouped data for a given object and band.
    If return_dict = True, then return a dictionary of DataFrames.
    Otherwise, return in the order:
        sdss, ps, ztf, ssa, tot
    """
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    if len(bands) == 1:
        sdss = pd.read_csv(cfg.D_DIR + f'surveys/sdss/{obj}/clean/{bands}_band/grouped.csv', index_col=ID, **kwargs)
        ps   = pd.read_csv(cfg.D_DIR + f'surveys/ps/{obj}/clean/{bands}_band/grouped.csv', index_col=ID, **kwargs)
        ztf  = pd.read_csv(cfg.D_DIR + f'surveys/ztf/{obj}/clean/{bands}_band/grouped.csv', index_col=ID, **kwargs)
        ssa  = pd.read_csv(cfg.D_DIR + f'surveys/ssa/{obj}/clean/{bands}_band/grouped.csv', index_col=ID, **kwargs)
    else:
        sdss = {b:pd.read_csv(cfg.D_DIR + f'surveys/sdss/{obj}/clean/{b}_band/grouped.csv'       , index_col=ID, **kwargs) for b in bands}
        ps   = {b:pd.read_csv(cfg.D_DIR + f'surveys/ps/{obj}/clean/{b}_band/grouped.csv'         , index_col=ID, **kwargs) for b in bands}
        ztf  = {b:pd.read_csv(cfg.D_DIR + f'surveys/ztf/{obj}/clean/{b}_band/grouped.csv'        , index_col=ID, **kwargs) for b in bands}
        ssa  = {b:pd.read_csv(cfg.D_DIR + f'surveys/ssa/{obj}/clean/{b}_band/grouped.csv', index_col=ID, **kwargs) for b in bands}
    
    if return_dict:
        return {'sdss':sdss, 'ps':ps, 'ztf':ztf, 'ssa':ssa}
    else:
        return sdss, ps, ztf, ssa
    
def load_grouped_tot(obj, bands=None, **kwargs):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    if len(bands) == 1:
        tot  = pd.read_csv(cfg.D_DIR + f'merged/{obj}/clean/grouped_{bands}.csv', index_col=ID, **kwargs)
    else:
        tot = {b:pd.read_csv(cfg.D_DIR + f'merged/{obj}/clean/grouped_{b}.csv', index_col=ID, **kwargs) for b in bands}

    return tot

def load_sets(obj, band, **kwargs):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    return pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/sets/clean_{band}.csv', index_col=ID, comment='#', **kwargs)

def load_coords(obj, **kwargs):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    return pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/{obj}_subsample_coords.csv', index_col=ID, comment='#', **kwargs)

def load_redshifts(**kwargs):
    if 'usecols' in kwargs: kwargs['usecols'] += ['uid']
    return pd.read_csv(cfg.D_DIR + f'catalogues/qsos/dr14q/dr14q_redshift.csv', index_col='uid', **kwargs).squeeze()

def load_n_tot(obj, **kwargs):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    return pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/n_tot.csv', index_col=ID, **kwargs)


def load_vac(obj, catalogue_name='dr16q_vac', **kwargs):
    """
    Load value-added catalogues for our quasar sample.
    Options of:
        dr12_vac
            Kozlowski 2016
            https://arxiv.org/abs/1609.09489
        dr14_vac
            Rakshit 2020
            https://arxiv.org/abs/1910.10395
        dr16q_vac : Preferred
            Shen 2022
            https://arxiv.org/abs/2209.03987
    """
    ID = 'uid'
    if 'usecols' in kwargs: kwargs['usecols'] += [ID]
    if obj == 'calibStars':
        raise Exception('Stars have no value-added catalogues')
    
    if catalogue_name == 'dr12_vac':
        # cols = z, Mi, L5100, L5100_err, L3000, L3000_err, L1350, L1350_err, MBH_MgII, MBH_CIV, Lbol, Lbol_err, nEdd, sdss_name, ra, dec, uid
        fpath = 'catalogues/qsos/dr12q/SDSS_DR12Q_BH_matched.csv'
        prop_range_all = {'Mi':(-30,-20),
                          'mag_mean':(15,23.5),
                          'mag_std':(0,1),
                          'redshift':(0,5),
                          'Lbol':(44,48),
                          'nEdd':(-3,0.5)}

    elif catalogue_name == 'dr14_vac':
        # cols = ra, dec, uid, sdssID, plate, mjd, fiberID, z, pl_slope, pl_slope_err, EW_MgII_NA, EW_MgII_NA_ERR, FWHM_MgII_NA, FWHM_MgII_NA_ERR, FWHM_MgII_BR, FWHM_MgII_BR_ERR, EW_MgII_BR, EW_MgII_BR_ERR, MBH_CIV, MBH_CIV_ERR, MBH, MBH_ERR, Lbol
        fpath = 'catalogues/qsos/dr14q/dr14q_spec_prop_matched.csv'
        prop_range_all = {'mag_mean':(15,23.5),
                          'mag_std':(0,1),
                          'redshift':(0,5),
                          'Lbol':(44,48)}

    elif catalogue_name == 'dr16q_vac':
        # cols = ra, dec, redshift_vac, Lbol, Lbol_err, MBH_HB, MBH_HB_err, MBH_MgII, MBH_MgII_err, MBH_CIV, MBH_CIV_err, MBH, MBH_err, nEdd, nEdd_err
        fpath = 'catalogues/qsos/dr16q/dr16q_vac_shen_matched.csv'
        prop_range_all = cfg.PREPROC.VAC_BOUNDS
    else:
        raise Exception('Unrecognised value-added catalogue')

    vac = pd.read_csv(os.path.join(cfg.D_DIR,fpath), index_col=ID, **kwargs)
    # vac = vac.rename(columns={'z':'redshift_vac'});
    if (catalogue_name == 'dr16q_vac') & ('nEdd' in vac.columns):
        # Note, in dr16q, bad nEdd entries are set to 0 (exactly) so we can remove those.
        vac['nEdd'] = vac['nEdd'].where((vac['nEdd']!=0).values)
    
    vac = parse.filter_data(vac, prop_range_all, dropna=False)

    return vac