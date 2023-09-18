from .config import cfg
import pandas as pd

def load_grouped(obj, band=None, return_dict=True):
    """
    Load grouped data for a given object and band.
    If return_dict = True, then return a dictionary of DataFrames.
    Otherwise, return in the order:
        sdss, ps, ztf, ssa, tot
    """
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if band is not None:
        sdss = pd.read_csv(cfg.D_DIR + f'surveys/sdss/{obj}/clean/{band}_band/grouped.csv', index_col=ID)
        ps   = pd.read_csv(cfg.D_DIR + f'surveys/ps/{obj}/clean/{band}_band/grouped.csv', index_col=ID)
        ztf  = pd.read_csv(cfg.D_DIR + f'surveys/ztf/{obj}/clean/{band}_band/grouped.csv', index_col=ID)
        ssa  = pd.read_csv(cfg.D_DIR + f'surveys/supercosmos/{obj}/clean/{band}_band/grouped.csv', index_col=ID)
    else:
        sdss = {band:pd.read_csv(cfg.D_DIR + f'surveys/sdss/{obj}/clean/{b}_band/grouped.csv'       , index_col=ID) for b in 'gri'}
        ps   = {band:pd.read_csv(cfg.D_DIR + f'surveys/ps/{obj}/clean/{b}_band/grouped.csv'         , index_col=ID) for b in 'gri'}
        ztf  = {band:pd.read_csv(cfg.D_DIR + f'surveys/ztf/{obj}/clean/{b}_band/grouped.csv'        , index_col=ID) for b in 'gri'}
        ssa  = {band:pd.read_csv(cfg.D_DIR + f'surveys/supercosmos/{obj}/clean/{b}_band/grouped.csv', index_col=ID) for b in 'gri'}
    
    if return_dict:
        return {'sdss':sdss, 'ps':ps, 'ztf':ztf, 'ssa':ssa}
    else:
        return sdss, ps, ztf, ssa
    
def load_grouped_tot(obj, band=None):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    if band is not None:
        tot  = pd.read_csv(cfg.D_DIR + f'merged/{obj}/clean/grouped_{band}.csv', index_col=ID)
    else:
        tot = {band:pd.read_csv(cfg.D_DIR + f'merged/{obj}/clean/grouped_{b}.csv', index_col=ID) for b in 'gri'}

    return tot

def load_sets(obj, band):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    return pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/sets/clean_{band}.csv', index_col=ID, comment='#')

def load_coords(obj):
    ID = 'uid' if obj == 'qsos' else 'uid_s'
    return pd.read_csv(cfg.D_DIR + f'catalogues/{obj}/{obj}_subsample_coords.csv', index_col=ID, comment='#')

def load_redshifts():
    return pd.read_csv(cfg.D_DIR + f'catalogues/qsos/dr14q/dr14q_redshift.csv', index_col='uid').squeeze()