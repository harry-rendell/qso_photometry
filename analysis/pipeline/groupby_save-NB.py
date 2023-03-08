# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# +
# nrows=None
# def reader(n_subarray):
#     return pd.read_csv('../data/merged/{}/{}_band/unclean/lc_{}.csv'.format(obj,band,n_subarray), nrows=nrows, index_col = ID, dtype = {'catalogue': np.uint8, 'mag_ps': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})

# +
nrows=None
def reader(n_subarray):
#     return pd.read_csv('../../data/merged/{}/{}_band/unclean/lc_{}.csv'.format(obj,band,n_subarray), nrows=nrows, comment='#', index_col = ID, dtype = {'catalogue': np.uint8, 'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})
    return pd.read_csv('../../data/merged/{}/{}_band/with_ssa/lc_{}.csv'.format(obj,band,n_subarray), nrows=nrows, comment='#', index_col = ID, dtype = {'catalogue': np.uint8, 'mag': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})




# -

class save_group():
    def __init__(self, obj='qsos', ID='uid', catalogue=0):
        self.obj = obj
        self.ID = ID
        self.catalogue = catalogue

    def read(self):
        if __name__ == '__main__':
            pool = Pool(4)
            df_list = pool.map(reader, range(4))
            self.df = pd.concat(df_list)#.rename(columns={'mag_ps':'mag'})
            if catalogue != 0:
                self.df = self.df[self.df['catalogue']==self.catalogue]
            
    def group_by(self, uid_subset=None):
        if uid_subset is None:
            uid_subset = self.df.index
        print('uids length:',len(uid_subset))
        print('creating uid chunks to assign to each core')
        
        # rather than doing this we could just split up the uids into 4 then use loc
        # this method might be better though as it distributes load evenly across all cores
        # wwait e could check where the splits happen and see if these line up with array_split(self.df.index.unique(),4)
        
        uids  = np.array_split(uid_subset,4)
        uid_0 = uids[0].unique()
        uid_1 = np.setdiff1d(uids[1].unique(),uid_0,assume_unique=True)
        uid_2 = np.setdiff1d(uids[2].unique(),uid_1,assume_unique=True)
        uid_3 = np.setdiff1d(uids[3].unique(),uid_2,assume_unique=True)

        assert (len(np.unique(uid_0)) == len(uid_0)), 'uid_0 has duplicates'
        assert (len(np.unique(uid_1)) == len(uid_1)), 'uid_1 has duplicates'
        assert (len(np.unique(uid_2)) == len(uid_2)), 'uid_2 has duplicates'
        assert (len(np.unique(uid_3)) == len(uid_3)), 'uid_3 has duplicates'

        X = np.concatenate((uid_0,uid_1,uid_2,uid_3))
        assert (len(np.unique(X)) == len(X)), 'There is overlap between chunks'

        print('assigning chunk to each core')
        if True:#__name__ == 'funcs.'+self.__class__.__name__:
            pool = Pool(4)
            df_list = pool.map(groupby_apply_single_survey, [self.df.loc[uid_0],self.df.loc[uid_1],self.df.loc[uid_2],self.df.loc[uid_3]])
            grouped = pd.concat(df_list)
        print('done')
        return grouped


# +
# These nested functions aren't very pretty, but without it multiprocessing.pool isn't happy.
def groupby_apply_single_survey(df):
    # first define the function to apply to the groups
    def stats(group):

        # assign pandas columns to numpy arrays
        mjds       = group['mjd'].values
        mag        = group['mag_ps'].values
        mag_native = group['mag'].values
        magerr     = group['magerr'].values

        # number of observations
        n_tot   = len(group)

        # time
        mjd_min =  min(mjds)
        mjd_max =  max(mjds)
        mjd_ptp =  np.ptp(group['mjd'])

        # magnitudes, using PS system
        mag_min = min(mag)
        mag_max = max(mag)
        mag_med = -2.5*np.log10(np.median(10**(-(mag-8.9)/2.5))) + 8.9
        mag_mean= -2.5*np.log10(np.mean  (10**(-(mag-8.9)/2.5))) + 8.9
        mag_std = np.std(mag)
        
        # native (untransformed) magnitudes
        mag_med_native  = -2.5*np.log10(np.median(10**(-(mag_native-8.9)/2.5))) + 8.9
        mag_mean_native = -2.5*np.log10(np.mean  (10**(-(mag_native-8.9)/2.5))) + 8.9
        
        # magnitude errors
        magerr_max = max(magerr)
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

        return {'n_tot':n_tot, 'mjd_min':mjd_min, 'mjd_max':mjd_max, 'mjd_ptp':mjd_ptp,
                'mag_min':mag_min, 'mag_max':mag_max, 'mag_mean':mag_mean, 'mag_med':mag_med, 'mag_mean_native':mag_mean_native, 'mag_med_native':mag_med_native, 'mag_opt_mean':mag_opt_mean, 'mag_opt_mean_flux':mag_opt_mean_flux, 'mag_std':mag_std,
                'magerr_max':magerr_max, 'magerr_mean':magerr_mean, 'magerr_med':magerr_med, 'magerr_opt_std':magerr_opt_std}
    
    return df.groupby(df.index.names).apply(stats).apply(pd.Series).astype({'n_tot':'uint16'})



# + active=""
# # These nested functions aren't very pretty, but without it multiprocessing.pool isn't happy.
# def groupby_apply(df):
#     # first define the function to apply to the groups
#     def stats(group):
#
#         # assign pandas columns to numpy arrays
#         mjds    = group['mjd'].values
#         mag     = group['mag'].values
#         magerr  = group['magerr'].values
#
#         # number of observations
#         n_sss_r1= (group['catalogue']==1).sum()
#         n_sss_r2= (group['catalogue']==1).sum()
#         n_sdss  = (group['catalogue']==5).sum()
#         n_ps    = (group['catalogue']==7).sum()
#         n_ztf   = (group['catalogue']==11).sum()
#         n_tot   = len(group)
#
#         # time
#         mjd_min =  min(mjds)
#         mjd_max =  max(mjds)
#         mjd_ptp =  np.ptp(group['mjd'])
#
#         # magnitudes
#         mag_min = min(mag)
#         mag_max = max(mag)
#         mag_med = -2.5*np.log10(np.median(10**(-(mag-8.9)/2.5))) + 8.9
#         mag_mean= -2.5*np.log10(np.mean  (10**(-(mag-8.9)/2.5))) + 8.9
#         mag_std = np.std(mag)
#
#         # magnitude errors
#         magerr_max = max(magerr)
#         magerr_med = np.median(magerr)
#         magerr_mean= np.mean(magerr)
#         
#         # flux
#         flux = 10**(-(mag-8.9)/2.5)
#         fluxerr = 0.921*flux*magerr # ln(10)/2.5 ~ 0.921
#         fluxerr_mean_opt = ( flux*(fluxerr**-2) ).sum()/(fluxerr**-2).sum()
#         # calculate the optimal flux average then convert back to mags
#         mag_opt_mean_flux = -2.5*np.log10(fluxerr_mean_opt) + 8.9
#         # magerr_opt_std_flux = not clear how to calculate this. magerr_opt_std should suffice.
#
#         # optimal (inverse-variance weighted) averages (see aaa04)
#         mag_opt_mean   = ( mag*(magerr**-2) ).sum()/(magerr**-2).sum()
#         magerr_opt_std = (magerr**-2).sum()**-0.5
#
#         return {'n_tot':n_tot, 'n_sss_r1':n_sss_r1, 'n_sss_r2':n_sss_r2, 'n_sdss':n_sdss, 'n_ps':n_ps, 'n_ztf':n_ztf,
#                 'mjd_min':mjd_min, 'mjd_max':mjd_max, 'mjd_ptp':mjd_ptp,
#                 'mag_min':mag_min, 'mag_max':mag_max, 'mag_mean':mag_mean, 'mag_med':mag_med, 'mag_opt_mean':mag_opt_mean, 'mag_opt_mean_flux':mag_opt_mean_flux, 'mag_std':mag_std,
#                 'magerr_max':magerr_max, 'magerr_mean':magerr_mean, 'magerr_med':magerr_med, 'magerr_opt_std':magerr_opt_std}
#     
#     return df.groupby(df.index.names).apply(stats).apply(pd.Series).astype({col:'uint16' for col in ['n_tot','n_sss_r1','n_sss_r2','n_sdss','n_ps','n_ztf']})

# +
# obj = 'calibStars'
# ID  = 'uid_s'
# band = 'g'
obj = 'qsos'
ID  = 'uid'
band = 'r'
survey = 'sss_r1'

catalogue_dict = {'all':0, 'sss_r1':1, 'sss_r2':3, 'sdss':5, 'ps':7, 'ztf':11}
catalogue = catalogue_dict[survey] # Only use data from named survey
# -



# +
from time import time
# nrowses = np.array([10e4, 20e4, 30e4, 40e4, 60e4, 100e4], dtype='int')
# times = []

# for nrows in nrowses:
for band in 'r':
    dr = save_group(obj, ID, catalogue=catalogue)
    dr.read()
    start = time()
    grouped = dr.group_by() 
    end = time()
    print('elapsed: {:.3f}s'.format(end-start))
    grouped.to_csv('../../data/merged/{}/{}_band/grouped_stats_{}_{}.csv'.format(obj,band,band,survey))
# -

grouped


