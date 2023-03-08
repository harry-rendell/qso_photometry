import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funcs.analysis import *

band = 'r'

def reader(n_subarray):
    return pd.read_csv('../data/merged/{}/{}_band/lc_{}.csv'.format(obj, band, n_subarray), nrows=None, index_col = ID, dtype = {'catalogue': np.uint8, 'mag_ps': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})

def average_nightly_obs(group):
    n = len(group)
    cat, mjd, mag, magerr = group[['catalogue','mjd','mag','magerr']].values.T
#     assert len(np.unique(cat))==1, group.index.unique()
    cat = cat[0]
    
    if (n==2) & (np.ptp(mag)>0.4):
        uid = group.index[0]
        median = grouped.loc[uid,['mag_med']].values
        idx = np.argmin(abs(mag-median))
        mag_mean = mag[idx] # throw away bad point (one that is furthest from median of lc)
        mjd_mean = mjd[idx]
        magerr_mean = magerr[idx]
        
    else:
        if n>2:
            uid = group.index[0]
            median_lc = grouped.loc[uid,['mag_med']].values
            mask = (abs(mag-(np.median(mag)+median_lc)/2) < 1)
            if mask.sum()==0:
                mag_mean = np.nan
                print('error with uid:',uid,'. Couldnt not average mags at mjd(s):',mjd)
#         mag_mean  = -2.5*np.log10(np.mean(10**(-(mag-8.9)/2.5))) + 8.9
            else:
                mag = mag[mask] # remove points that are 1mag away from the median of the group
                magerr = magerr[mask]
                mjd = mjd[mask]
            
        mjd_mean  = np.mean(mjd)
        magerr_mean = (magerr ** -2).sum() ** -0.5 # sum errors in quadrature
        mag_mean  = -2.5*np.log10(np.average(10**(-(mag-8.9)/2.5), weights = magerr**-2)) + 8.9
        
    return {'catalogue':cat,'mjd':mjd_mean, 'mag':mag_mean, 'magerr':magerr_mean}

obj = 'qsos'
ID  = 'uid'
dr = analysis(ID)
dr.read_in(reader, redshift=False)

dr.df['mjd_floor'] = np.floor(dr.df['mjd']).astype('int')

grouped = pd.read_csv('../data/merged/qsos/r_band/grouped_stats_{}.csv'.format(band), index_col='uid')

mask = dr.df.reset_index()[['uid','mjd_floor']].duplicated(keep=False).values
single_obs = dr.df[~mask].drop(columns='mjd_floor')  # observations that do not share the same night with any other observations
multi_obs = dr.df[mask] # observations that have at least one other observation that night. We need to groupby these then add them back onto df above. then sort_values(['uid','mjd'])

from time import time
start = time()
avgd  = multi_obs.groupby(['uid','mjd_floor']).apply(average_nightly_obs).apply(pd.Series).reset_index('mjd_floor', drop=True).astype({'catalogue':'int'})
end   = time()
print('elapsed: {:.2f}'.format(end-start))