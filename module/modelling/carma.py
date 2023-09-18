import os
import pandas as pd
import numpy as np
from ..preprocessing.data_io import groupby_apply_dispatcher
from ..config import cfg
from eztao.ts import drw_fit
from eztao.carma import DRW_term
from eztao.ts import gpSimRand

def apply_drw_fit(group):
    """
    Find best DRW parameters for a single lightcurve using whole lightcurve and just ZTF
    """
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
    """
    This may be superseeded by data_io.dispatch_groupby
    """
    return groupby_apply_dispatcher(apply_drw_fit, df, kwargs)

def generate_mask(t, survey_features):
    indices = np.array([], dtype='int')
    sid = np.array([], dtype='int')
    for s in survey_features.keys():
        idx_lower = np.argmin(abs(survey_features[s]['mjd_min'][0] - t))
        idx_upper = np.argmin(abs(survey_features[s]['mjd_max'][0] - t))
        n_tot = int(max(1,3*np.random.normal(survey_features[s]['n_tot'][0], survey_features[s]['n_tot'][1])))
        indices = np.append(indices, np.unique(np.random.randint(idx_lower, idx_upper, size=n_tot)))
        sid = np.append(sid, np.full(n_tot, cfg.PREPROC.SURVEY_IDS[s]))
        # idxs = np.random.randint(0, len(t), size=np.random.randint(0,len(t)), replace=False)
        idxs = np.argsort(indices)

    return indices[idxs], sid[idxs]

def generate_simulated_lc(survey_features):
    amp = -1
    while amp < 0:
        amp = np.random.normal(loc=0.6, scale=0.14)
    tau = -1
    while tau < 0:
        tau = np.random.normal(loc=600, scale=1000)
    DRW_kernel = DRW_term(np.log(amp), np.log(tau))
    t,y,yerr = gpSimRand(DRW_kernel, SNR=10, duration=365*70, N=int(365*70*0.1), nLC=1, log_flux=True, season=False, full_N=10000)
    t += 33238 # shift lcs to first observation of SSS
    y += 20
    idxs, sid = np.sort(generate_mask(t, survey_features))
    
    return t[idxs], y[idxs], yerr[idxs], sid

def generate_mock_dataset(uids_and_suffix, kwargs):
    # Use os.urandom to generate a random seed, required if we are running this function in parallel
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    uids, suffix = uids_and_suffix
    survey_features = kwargs['survey_features']
    b = []
    for uid in uids:
        mjd, mag, magerr, sid = generate_simulated_lc(survey_features)
        c = pd.DataFrame({'mjd':mjd, 'mag':mag, 'magerr':magerr, 'sid':sid})
        c['uid'] = uid
        c['mjd_rf'] = c['mjd']/(1+abs(np.random.uniform(1.83, 0.78)))
        b.append(c)

    b = pd.concat(b)
    b = b.astype({col:dtype for col, dtype in cfg.PREPROC.lc_dtypes.items() if col in b.columns})
    b['band'] = kwargs['band']
    b.to_csv(cfg.D_DIR + f'merged/sim/clean/lc_{suffix}.csv', index=False)
