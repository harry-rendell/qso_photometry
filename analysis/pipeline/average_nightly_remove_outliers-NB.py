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

import sys
sys.path.append('../')
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from funcs.analysis.analysis import *
from funcs.config import cfg
import time
# %matplotlib inline

wdir = cfg.USER.W_DIR
# obj = 'calibStars'
# ID  = 'uid_s'
obj = 'qsos'
ID = 'uid'
band = 'r'
dr = analysis(ID, obj, band)
dr.read_in(redshift=False, cleaned=False, nrows=100)
dr.df['mjd_floor'] = np.floor(dr.df['mjd']).astype('int')
dr.group(keys=[ID], read_in=True, redshift=False, colors=False, survey='all', restrict=True) # previously we read in grouped explicitly and used it. Now changed to dr.df_grouped

# ### How many repeat observations per night

# +
dr.summary()
sdss = dr.df[dr.df['catalogue']==5].reset_index()
frac = sdss.duplicated(subset=['uid','mjd_floor'], keep='first').sum()/len(sdss)
print('Number of SDSS observations within the same night of the same object: {:.2f}%'.format(frac*100))
print('-'*20)

ps = dr.df[dr.df['catalogue']==7].reset_index()
frac = ps.duplicated(subset=['uid','mjd_floor'], keep='first').sum()/len(ps)
print('Number of PS   observations within the same night of the same object: {:.2f}%'.format(frac*100))
print('-'*20)

ztf = dr.df[dr.df['catalogue']==11].reset_index()
frac = ztf.duplicated(subset=['uid','mjd_floor'], keep='first').sum()/len(ztf)
print('Number of ZTF  observations within the same night of the same object: {:.2f}%'.format(frac*100))
print('-'*20)

frac = dr.df.reset_index().duplicated(subset=['uid','mjd_floor'], keep='first').sum()/len(dr.df)
print('Total observations within the same night of the same object: {:.2f}%'.format(frac*100))

# + active=""
# test = ztf.groupby(['uid','mjd_floor']).agg({'mag':np.ptp})

# + active=""
# test = ztf.groupby(['uid','mjd_floor']).size()

# + active=""
# test2 = test.reset_index('mjd_floor').loc[:,0].sort_values(ascending=False)

# + active=""
# fig, ax= plt.subplots(1,1, figsize=(16,10))
# ax.hist(test2, bins=100);
# ax.set(yscale='log', ylabel='counts', xlabel='number of observations per night')
# -

# # dr.coords

# Proof that ZTF errors are grossly underestimated
dr.plot_series([471006], xlim=[58298.35,58298.43])

dr.summary() # gives us dr.n_qsos
frac = len(dr.df[dr.df.reset_index().duplicated(subset=['uid','mjd_floor'], keep='first').values].index.unique())/dr.n_qsos
print('Fraction of objects that have another observation within the same night: {:.2f}%'.format(frac*100))

# ___
# # Removing bad objects
# two methods:
# 1. Remove objects which are fainter than the limiting magnitude of the survey (need to find limits for each band)
# 2. Remove objects which have a magerr > 0.198

# +
## catalogue = dr.df['catalogue']
# mag_med = dr.df_grouped['mag_med']
# mag_med = mag_med[mag_med.index.isin(catalogue.index)]
# mask = ((catalogue == 5) & (mag_med < 22.7)) | ((catalogue == 7) & (mag_med < 23.2)) | ((catalogue == 11) & (mag_med < 20.6))

magerr_med = dr.df_grouped['magerr_med']
mask = (magerr_med > cfg.PREPROC.MAG_ERR_THRESHOLD)
print('Number of datapoints removed: {:,}/{:,} \t {:.2f}%'.format(mask.sum(), len(mask), mask.sum()/len(mask)))


# -

dr.df[mask]['magerr'].hist(bins=100)
dr.df[mask]['magerr'].describe()

dr.df[~mask]['magerr'].hist(bins=100)
dr.df[~mask]['magerr'].describe()

# +
## This needs to be in its own notebook or script
# # WRITE
# grouped_q = pd.read_csv('../data/merged/{}/{}_band/grouped_stats_{}.csv'.format('qsos',band,band), index_col='uid')
# bins=np.linspace(16,21,51)
# counts, _ = np.histogram(grouped_q['mag_mean'], bins=bins)
# uid_s_list = []
# for i in range(len(counts)):
#     uid_s_list.append(np.random.choice(dr.df_grouped[(bins[i] < dr.df_grouped['mag_mean']) & (dr.df_grouped['mag_mean'] < bins[i+1])].index, size=counts[i], replace=False))
# # uid_s_list = np.concatenate(uid_s_list)
# np.savetxt('/disk1/hrb/python/data/merged/calibStars/r_band/uid_s_matched.csv', uid_s_list, fmt='%i')

# # READ
# if obj == 'calibStars':
#     uid_s_matched_sub = np.loadtxt('/disk1/hrb/python/data/merged/calibStars/r_band/uid_s_matched.csv', dtype='int')
#     uid_s_matched = np.append(dr.df_grouped[dr.df_grouped['mag_mean']>21].index, uid_s_matched_sub)
#     dr.df_grouped = dr.df_grouped.loc[uid_s_matched]
#     dr.df = dr.df[dr.df.index.isin(uid_s_matched)]
#     del uid_s_matched, uid_s_matched_sub
    
# # Plot match
# fig, ax = plt.subplots(1,1, figsize=(10,8))
# grouped_q['mag_mean'].hist(bins=51, ax=ax, label='qsos', alpha=0.5, range=(16,23))
# # grouped2['mag_mean'].hist(bins=51, ax=ax, label='stars', alpha=0.5, range=(16,23))
# dr.df_grouped['mag_mean'].hist(bins=51, ax=ax, label='stars', alpha=0.5, range=(16,23))
# ax.legend()

# fig, ax = plt.subplots(1,1, figsize=(10,8))
# grouped_q['mag_std'].hist(bins=51, ax=ax, label='qsos', alpha=0.5, range=(0,0.6))
# # grouped2['mag_std'].hist(bins=51, ax=ax, label='stars', alpha=0.5, range=(0,0.6))
# dr.df_grouped['mag_std'].hist(bins=51, ax=ax, label='stars', alpha=0.5, range=(0,0.6))
# ax.legend()
# -

# ### Averaging nightly observations

# * We want to combine multiple observations on the same night into a single datapoint, while throwing out bad points.
# * histograms below show the distribution of mjd's mod 1, ie what UTC time during the night the observations are taken.
# * it is safe to round mjd down to the nearest integer and then bin them (since the distribution doesnt carry over from 1 back to 0, which would split the same night observations over two mjd's)

# This plot shows 
fig, axes = plt.subplots(3,1, figsize=(15,5))
for cat, ax in zip([5,7,11],axes.ravel()):
    (dr.df[dr.df['catalogue']==cat]['mjd'] % 1).hist(bins=100, range=(0,1), ax=ax, alpha=1, density=True, label=dr.survey_dict[cat])
    ax.legend()
    ax.set(xlabel='mjd mod 1', yscale='log')

CHECK_OBS_DIFFERENT_SURVEY_ON_SAME_DAY = False
if CHECK_OBS_DIFFERENT_SURVEY_ON_SAME_DAY:
    # Check that we do not have multiple observations from different surveys that lie on the same day.
    # If we do, we need to treat them differently.
    mask1 = dr.df.reset_index().duplicated([ID,'mjd_floor'], keep=False)
    mask2 = dr.df.reset_index().duplicated([ID,'mjd_floor','catalogue'], keep=False)
    if (mask1 ^ mask2).sum()>0:
        print(dr.df[(mask1 ^ mask2).values])
        raise Warning('The observations listed above have data from multiple surveys on the same day')
    
    # With our current data, we have observations from PS and ZTF on MJD=55102.
    # For now we will just average this data as normal


# +
### Check this against scripts_test/remove_outliers_MAD as it will be a better method

def average_nightly_obs(group):
    # Each group is all the data collected on a given night
    n = len(group) # number of observations on that night
    cat, mjd, mag, magerr = group[['catalogue','mjd','mag','magerr']].values.T
    # TODO: The statement below should be asserted prior to calling this function, since it will slow it down.
    # assert len(np.unique(cat))==1, group.index.unique() 
    cat = cat[0] # Take the first entry since they should all be the same

    if (n==2) & (np.ptp(mag)>1): # why 0.4? maybe better to use np.sum(error)
        # If we have two observations on the same night and the difference is greater than 0.4 mag
        # then keep the datapoint which is closest to the median of the total light curve (from grouped)
        uid = group.index[0]
        median = dr.df_grouped.loc[uid,['mag_med']].values
        idx = np.argmin(abs(mag-median))
        mag_mean = mag[idx]
        mjd_mean = mjd[idx]
        magerr_mean = magerr[idx]

    else:
        if n>2:
            # Remove points that are 1mag away from the median of the group
            uid = group.index[0]
            median_lc = dr.df_grouped.loc[uid,['mag_med']].values
            mask = (abs(mag-(np.median(mag)+median_lc)/2) < 1)
            if mask.sum()==0:
                # If there are no data points within 1mag of the median of the light curve, then set mag_mean=np.nan
                mag_mean = np.nan
                err_msg = 'error with uid: '+str(uid)+'. Couldnt not average mags at mjd(s): '+str(mjd)
                print(err_msg)
                with open('average_nightly_obs_log.txt','a+') as file:
                    file.write(err_msg)
            else:
                mag = mag[mask]
                magerr = magerr[mask]
                mjd = mjd[mask]

        mjd_mean  = np.mean(mjd)
        magerr_mean = (magerr ** -2).sum() ** -0.5 # sum errors in quadrature
        mag_mean  = -2.5*np.log10(np.average(10**(-(mag-8.9)/2.5), weights = magerr**-2)) + 8.9

    return {'catalogue':cat,'mjd':mjd_mean, 'mag':mag_mean, 'magerr':magerr_mean}


# +
mask = dr.df.reset_index()[[ID,'mjd_floor']].duplicated(keep=False).values
single_obs = dr.df[~mask].drop(columns='mjd_floor')  # observations that do not share the same night with any other observations
multi_obs = dr.df[mask] # observations that have at least one other observation that night. We need to groupby these then add them back onto df above. then sort_values([ID,'mjd'])

# del dr.df # We don't need this DataFrame anymore, delete it for memory management
# -

multi_obs_ps = multi_obs[multi_obs['catalogue']==7]
multi_obs_ps_ptp = multi_obs_ps.groupby(['uid','mjd_floor']).agg({'mag':np.ptp})

multi_obs_ztf = multi_obs[multi_obs['catalogue']==11]
multi_obs_ztf_ptp = multi_obs_ztf.groupby(['uid','mjd_floor']).agg({'mag':np.ptp})

fig, ax = plt.subplots(1,1, figsize=(20,5))
ax.hist(multi_obs_ztf_ptp, bins=100)
ax.set(yscale='log', ylabel='counts',xlabel='∆m for observations within a night for ZTF (mag)')

fig, ax = plt.subplots(1,1, figsize=(20,5))
ax.hist(multi_obs_ps_ptp, bins=100)
ax.set(yscale='log', ylabel='counts',xlabel='∆m for observations within a night for PS (mag)')

uids = multi_obs_ztf_ptp[multi_obs_ztf_ptp['mag'] > 2].head(10).index.get_level_values(0)
dr.plot_series(uids, survey=11, filtercodes='r')

uids = multi_obs_ps_ptp[multi_obs_ps_ptp['mag'] > 2].head(10).index.get_level_values(0)
dr.plot_series(uids, survey=7, filtercodes='r')

# ---

test_objects = analysis(ID, obj, band)
test_objects.df = dr.df.loc[np.random.choice(multi_obs.index.unique(),size=10)]

# +
start = time.time()
if cfg.USER.USE_MULTIPROCESSING:
    chunks = split_into_non_overlapping_chunks(multi_obs, 4)
    # del multi_obs # We don't need this DataFrame anymore, delete it for memory management

    # For multiprocessing
    def multiobs_groupby(chunk):
        return chunk.groupby([ID,'mjd_floor']).apply(average_nightly_obs).apply(pd.Series).reset_index('mjd_floor', drop=True).astype({'catalogue':'int'})

    if __name__ == '__main__':
        pool = Pool(cfg.USER.N_CORES)
        averaged = pd.concat(pool.map(multiobs_groupby, chunks))

else:
    averaged  = multi_obs.groupby([ID,'mjd_floor']).apply(average_nightly_obs).apply(pd.Series).reset_index('mjd_floor', drop=True).astype({'catalogue':'int'})

end = time.time()
print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(end-start)))

SAVE_INTERMEDIATE = True
if SAVE_INTERMEDIATE:
    averaged.to_csv('averaged_qsos.csv')
# -

READ_INTERMEDIATE = False
if READ_INTERMEDIATE:
    averaged = pd.read_csv('averaged_qsos.csv',index_col=ID)

# Add the data from nights with multiple observations back on
dr.df = single_obs.append(averaged).sort_values([ID,'mjd'])
dr.df.index = dr.df.index.astype('int')

# ### remove outliers

# test our data cleaning is working
survey = 11
uids = test_objects.df.index.unique()
ax = dr.plot_series(uids, survey=survey, filtercodes='r')
# test_objects.plot_series(uids, survey=survey, filtercodes='r', axes=ax)
for uid, axis in zip(uids, ax):
    mjd, mag, magerr = test_objects.df.loc[uid,['mjd','mag','magerr']].values.T
    axis.errorbar(mjd, mag, yerr = magerr, lw = 0.5, markersize = 10, color='b')
    axis.set(xlim=[58200,58500])
# fig.savefig('averaging_nightly_observations_ztf.pdf', bbox_inches='tight')

# + active=""
# def reader(n_subarray):
#     return pd.read_csv('../data/merged/{}/{}_band/lc_{}.csv'.format(obj, band, n_subarray), nrows=None, index_col = ID, dtype = {'catalogue': np.uint8, 'mag_ps': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})

# + active=""
# band='i'

# + active=""
# obj = 'qsos'
# ID  = 'uid'
# dr = analysis(ID)
# dr.read_in(reader, redshift=False)
# # dr.df['mjd_floor'] = np.floor(dr.df['mjd']).astype('int')
# -

dr.df = dr.df.join(dr.df_grouped[['mag_med','mag_std']], on=ID)
dr.df['Z'] = (dr.df['mag']-dr.df['mag_med'])/dr.df['mag_std']


def remove_outliers(group):
    dmjd1, dmag1 = group[['mjd','mag']].diff(-1).fillna(0).values.T
    dmjd2, dmag2 = group[['mjd','mag']].diff(+1).fillna(0).values.T
    clean = group[~(((abs(dmag1)>0.6) & (abs(dmag2)>0.6)) & ((abs(dmjd1)<500) | (abs(dmjd2)<500)))]
    return clean


start = time.time()
clean_phot = dr.df[['catalogue','mjd','mag','magerr']].groupby(ID).apply(remove_outliers).reset_index(level=1, drop=True)
end   = time.time()
print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(end-start)))


# +
# # testing outlier detection
# fig, ax = dr.plot_series(uids, survey=3, filtercodes='r')
# for i, axis in enumerate(ax):
#     group = dr.df.loc[uids[i]]
#     dmjd1, dmag1 = group[['mjd','mag']].diff(-1).fillna(0).values.T
#     dmjd2, dmag2 = group[['mjd','mag']].diff(+1).fillna(0).values.T
#     n_remove = (((abs(dmag1)>0.6) & (abs(dmag2)>0.6)) & ((abs(dmjd1)<500) | (abs(dmjd2)<500))).sum()
#     axis.text(0.02, 0.72, 'n_remove: {:d}'.format(n_remove), transform=axis.transAxes, fontsize=10)
#     axis.text(0.02, 0.80, 'Z: {:.2f}'.format(sample['Z'].values[i]), transform=axis.transAxes, fontsize=10)
# -

# ### save processed data

def save(args):
    i, chunk = args
    f = open('/disk1/hrb/python/data/merged/{}/{}_band/lc_{}.csv'.format(obj,band,i), 'w')
    comment = '# CSV of cleaned photometry.\n# Data has been averaged nightly and outliers have been removed\n'
    f.write(comment)
    chunk.to_csv(f)


# multiprocesssor
chunks = np.array_split(clean_phot,4)
if __name__ == '__main__':
    pool = Pool(4)
    pool.map(save, enumerate(chunks))

# +
# single processor
# for i, chunk in enumerate(np.array_split(clean_phot,4)):
#     f = open('/disk1/hrb/python/data/merged/qsos/{}_band/lc_{}.csv'.format(band,i), 'a')
#     comment = '# CSV of cleaned photometry.\n# Data has been averaged nightly and outliers have been removed'
#     f.write(comment)
#     chunk.to_csv(f)
# -


