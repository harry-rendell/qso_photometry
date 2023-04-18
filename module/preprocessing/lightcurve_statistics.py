import numpy as np
import pandas as pd
from ..config import cfg

# def groupby_apply_stats(args):
#     df, kwargs = args
#     return df.groupby(df.index.names).apply(stats).apply(pd.Series).astype(cfg.PREPROC.stats_dtypes)

def groupby_apply_average(df):
    s = df.groupby([df.index.name, 'mjd_floor']).apply(average_nightly_obs)
    return pd.DataFrame(s.values.tolist(), index=s.index, dtype='float32').reset_index('mjd_floor', drop=True)

def groupby_apply_stats(df):
    s = df.groupby(df.index.name).apply(stats)
    return pd.DataFrame(s.values.tolist(), index=s.index, dtype='float32')

def average_nightly_obs(group):
    n = len(group)
    mjd, mag, magerr, mag_median, mag_std = group[['mjd','mag','magerr','mag_med','mag_std']].values.T
    
    if np.ptp(mag) > 0.5:
        mask = abs(mag-mag_median) < 5*mag_std
        if mask.sum()==0:
            mag_mean = np.nan
            uid = group.index[0]
            err_msg = 'Error with uid: '+str(uid)+'. Could not average mags at mjd: '+str(int(mjd[0]))+'\n'
            with open(cfg.USER.W_DIR + 'python/scripts/logs/average_nightly_obs_log.txt', 'a') as myfile:
                myfile.write(err_msg)
                
        else:
            mag = mag[mask] # remove points that are 1mag away from the median of the group
            magerr = magerr[mask]
            mjd = mjd[mask]
            
    mjd_mean  = np.mean(mjd)
    magerr_mean = ((magerr ** 2).sum()/n) ** 0.5 # sum errors in quadrature. Do not use 'error on optimal average' since it makes the errors unphysically small.
    mag_mean  = -2.5*np.log10(np.average(10**(-(mag-8.9)/2.5), weights = magerr**-2)) + 8.9
        
    return {'mjd':mjd_mean, 'mag':mag_mean, 'magerr':magerr_mean}

def stats(group):
    # assign pandas columns to numpy arrays
    mjds       = group['mjd'].values
    mag        = group['mag'].values
    magerr     = group['magerr'].values

    # number of observations
    n_tot   = len(group)

    # time
    mjd_min =  mjds.min()
    mjd_max =  mjds.max()
    mjd_ptp =  np.ptp(group['mjd'])

    # magnitudes, using PS system
    mag_min = min(mag)
    mag_max = max(mag)
    mag_med = -2.5*np.log10(np.median(10**(-(mag-8.9)/2.5))) + 8.9
    mag_mean= -2.5*np.log10(np.mean  (10**(-(mag-8.9)/2.5))) + 8.9
    mag_std = np.std(mag)
    
    # native (untransformed) magnitudes
    if 'mag_orig' in group:
        mag_native = group['mag_orig'].values
        mag_med_native  = -2.5*np.log10(np.median(10**(-(mag_native-8.9)/2.5))) + 8.9
        mag_mean_native = -2.5*np.log10(np.mean  (10**(-(mag_native-8.9)/2.5))) + 8.9
        mag_stats_native = {'mag_mean_native':mag_mean_native,
                            'mag_med_native':mag_med_native}
    else:
        mag_stats_native = {}
    
    # magnitude errors
    magerr_max = magerr.max()
    magerr_med = np.median(magerr)
    magerr_mean= np.mean(magerr)
    
    # using flux flux
    flux = 10**(-(mag-8.9)/2.5)
    fluxerr = 0.921*flux*magerr # ln(10)/2.5 ~ 0.921
    fluxerr_mean_opt = ( flux*(fluxerr**-2) ).sum()/(fluxerr**-2).sum()
    # calculate the optimal flux average then convert back to mags
    mag_opt_mean_flux = -2.5*np.log10(fluxerr_mean_opt) + 8.9
    # magerr_opt_std_flux = not clear how to calculate this. magerr_opt_std should suffice.

    # optimal (inverse-variance weighted) averages (see aaa04)
    mag_opt_mean   = ( mag*(magerr**-2) ).sum()/(magerr**-2).sum()
    magerr_opt_std = (magerr**-2).sum()**-0.5

    return {**{'n_tot':n_tot,
            'mjd_min':mjd_min,
            'mjd_max':mjd_max,
            'mjd_ptp':mjd_ptp,
            'mag_min':mag_min,
            'mag_max':mag_max,
            'mag_mean':mag_mean,
            'mag_med':mag_med,
            'mag_opt_mean':mag_opt_mean,
            'mag_opt_mean_flux':mag_opt_mean_flux,
            'mag_std':mag_std,
            'magerr_max':magerr_max,
            'magerr_mean':magerr_mean,
            'magerr_med':magerr_med,
            'magerr_opt_std':magerr_opt_std}, **mag_stats_native}
