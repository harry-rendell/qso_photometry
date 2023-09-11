import pandas as pd
import numpy as np
from ..preprocessing.data_io import groupby_apply_dispatcher
from eztao.ts import drw_fit

def apply_drw_fit(group):
    if (len(group) < 3) or (np.ptp(group['mjd_rf']) < 10):
        return {'sig':np.nan, 'tau':np.nan, 'sig_ztf':np.nan, 'tau_ztf':np.nan}

    t, y, yerr = group[['mjd_rf', 'mag', 'magerr']].values.T
    try:
        sig, tau = drw_fit(t, y, yerr)
    except:
        print('error with uid:', group.index[0])
        return {'sig':np.nan, 'tau':np.nan, 'sig_ztf':np.nan, 'tau_ztf':np.nan}

    mask = (group['sid']==11).values
    try:
        sig_ztf, tau_ztf = drw_fit(t[mask], y[mask], yerr[mask])
    except:
        print('error with uid:', group.index[0])
        sig_ztf, tau_ztf = np.nan, np.nan

    return {'sig':sig, 'tau':tau, 'sig_ztf':sig_ztf, 'tau_ztf':tau_ztf}

def groupby_apply_drw_fit(df, kwargs):
    return groupby_apply_dispatcher(apply_drw_fit, df, kwargs)