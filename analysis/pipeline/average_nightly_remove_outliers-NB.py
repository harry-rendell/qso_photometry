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
# %matplotlib inline

band = 'r'
wdir = '/disk1/hrb/python/'


def reader(n_subarray):
    return pd.read_csv(wdir+'data/merged/{}/{}_band/unclean/lc_{}.csv'.format(obj, band, n_subarray), nrows=None, index_col = ID, dtype = {'catalogue': np.uint8, 'mag_ps': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})



# obj = 'calibStars'
# ID  = 'uid_s'
obj = 'qsos'
ID = 'uid'
dr = analysis(ID, obj)
dr.read_in(reader, redshift=False)
dr.df['mjd_floor'] = np.floor(dr.df['mjd']).astype('int')

dr.df

# ### How many repeat observations per night

# +
dr.summary()
sdss = dr.df[dr.df['catalogue']==5].reset_index()
frac = sdss.duplicated(subset=['uid','mjd_floor'], keep='first').sum()/len(sdss)
print('Number of SDSS observations within the same night of the same object: {:.2f}%'.format(frac*100))
ps = dr.df[dr.df['catalogue']==7].reset_index()
frac = ps.duplicated(subset=['uid','mjd_floor'], keep='first').sum()/len(ps)
print('Number of PS   observations within the same night of the same object: {:.2f}%'.format(frac*100))
ztf = dr.df[dr.df['catalogue']==11].reset_index()
frac = ztf.duplicated(subset=['uid','mjd_floor'], keep='first').sum()/len(ztf)
print('Number of ZTF  observations within the same night of the same object: {:.2f}%'.format(frac*100))

frac = dr.df.reset_index().duplicated(subset=['uid','mjd_floor'], keep='first').sum()/len(dr.df)
print('Total observations within the same night of the same object: {:.2f}%'.format(frac*100))
# -

dr.summary() # gives us dr.n_qsos
frac = len(dr.df[dr.df.reset_index().duplicated(subset=['uid','mjd_floor'], keep='first').values].index.unique())/dr.n_qsos
print('Fraction of objects that have another observation within the same night: {:.2f}%'.format(frac*100))

grouped = pd.read_csv(wdir+'data/merged/{}/{}_band/grouped_stats_{}_all.csv'.format(obj,band,band), index_col=ID)

grouped

dr.df

# +
# Remove objects which are fainter than the limiting magnitude of the survey (or we dould do magerr=0.198)
catalogue = dr.df['catalogue']
# mag_med = grouped['mag_med']
# mag_med = mag_med[mag_med.index.isin(catalogue.index)]
# mask = ((catalogue == 5) & (mag_med < 22.7)) | ((catalogue == 7) & (mag_med < 23.2)) | ((catalogue == 11) & (mag_med < 20.6))

magerr_med = grouped['magerr_med']
magerr_med = magerr_med[magerr_med.index.isin(catalogue.index)]
mask = (catalogue > 1) & (magerr_med > 0.198)


# -

mask

dr.df

dr.df[mask]['magerr'].hist(bins=100)
dr.df[mask]['magerr'].describe()

dr.df[~mask]['magerr'].hist(bins=100)
dr.df[~mask]['magerr'].describe()

dr.df[mask]['magerr'].hist(bins=100)
dr.df[mask]['magerr'].describe()

dr.df[~mask]['magerr'].hist(bins=100)
dr.df[~mask]['magerr'].describe()



(grouped['mag_med']<22.7) &  (dr.df['catalogue']==5)

# +
# # WRITE
# grouped_q = pd.read_csv('../data/merged/{}/{}_band/grouped_stats_{}.csv'.format('qsos',band,band), index_col='uid')
# bins=np.linspace(16,21,51)
# counts, _ = np.histogram(grouped_q['mag_mean'], bins=bins)
# uid_s_list = []
# for i in range(len(counts)):
#     uid_s_list.append(np.random.choice(grouped[(bins[i] < grouped['mag_mean']) & (grouped['mag_mean'] < bins[i+1])].index, size=counts[i], replace=False))
# # uid_s_list = np.concatenate(uid_s_list)
# np.savetxt('/disk1/hrb/python/data/merged/calibStars/r_band/uid_s_matched.csv', uid_s_list, fmt='%i')
# -

# READ
if obj == 'calibStars':
    uid_s_matched_sub = np.loadtxt('/disk1/hrb/python/data/merged/calibStars/r_band/uid_s_matched.csv', dtype='int')
    uid_s_matched = np.append(grouped[grouped['mag_mean']>21].index, uid_s_matched_sub)
    grouped = grouped.loc[uid_s_matched]
    dr.df = dr.df[dr.df.index.isin(uid_s_matched)]

dr.df

del uid_s_matched, uid_s_matched_sub

# +
fig, ax = plt.subplots(1,1, figsize=(10,8))
grouped_q['mag_mean'].hist(bins=51, ax=ax, label='qsos', alpha=0.5, range=(16,23))
# grouped2['mag_mean'].hist(bins=51, ax=ax, label='stars', alpha=0.5, range=(16,23))
grouped['mag_mean'].hist(bins=51, ax=ax, label='stars', alpha=0.5, range=(16,23))
ax.legend()

fig, ax = plt.subplots(1,1, figsize=(10,8))
grouped_q['mag_std'].hist(bins=51, ax=ax, label='qsos', alpha=0.5, range=(0,0.6))
# grouped2['mag_std'].hist(bins=51, ax=ax, label='stars', alpha=0.5, range=(0,0.6))
grouped['mag_std'].hist(bins=51, ax=ax, label='stars', alpha=0.5, range=(0,0.6))
ax.legend()
# -

# ### Averaging nightly observations

# > We want to combine multiple observations on the same night into a single datapoint, while throwing out bad points
# > histograms below show that it is safe to round mjd down to the nearest integer and then bin them (since the distribution doesnt carry over from 1 back to 0

# + active=""
# from astropy.time import Time
# t = Time('2021-04-13 15:09:00')
# t.mjd
# -

fig, axes = plt.subplots(3,1, figsize=(15,5))
for cat, ax in zip([1,2,3],axes.ravel()):
    (dr.df[dr.df['catalogue']==cat]['mjd'] % 1).hist(bins=100, range=(0,1), ax=ax, alpha=1, density=True, label=dr.survey_dict[cat])
    ax.legend()
    ax.set(xlabel='mjd mod 1', yscale='log')


# +
def average_nightly_obs(group):
    n = len(group)
    cat, mjd, mag, magerr = group[['catalogue','mjd','mag','magerr']].values.T
#     assert len(np.unique(cat))==1, group.index.unique()
    cat = cat[0]
    
    if (n==2) & (np.ptp(mag)>0.4): # why 0.4? maybe better to use np.sum(error)
        # If we have two observations on the same night and the difference is greater than 0.4 mag
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
                err_msg = 'error with uid: '+str(uid)+'. Couldnt not average mags at mjd(s): '+str(mjd)
                print(err_msg)
                log.append(err_msg)
                
#         mag_mean  = -2.5*np.log10(np.mean(10**(-(mag-8.9)/2.5))) + 8.9
            else:
                mag = mag[mask] # remove points that are 1mag away from the median of the group
                magerr = magerr[mask]
                mjd = mjd[mask]
            
        mjd_mean  = np.mean(mjd)
        magerr_mean = (magerr ** -2).sum() ** -0.5 # sum errors in quadrature
        mag_mean  = -2.5*np.log10(np.average(10**(-(mag-8.9)/2.5), weights = magerr**-2)) + 8.9
        
    return {'catalogue':cat,'mjd':mjd_mean, 'mag':mag_mean, 'magerr':magerr_mean}


# -

mask = dr.df.reset_index()[[ID,'mjd_floor']].duplicated(keep=False).values
single_obs = dr.df[~mask].drop(columns='mjd_floor')  # observations that do not share the same night with any other observations
# multi_obs = dr.df[mask] # observations that have at least one other observation that night. We need to groupby these then add them back onto df above. then sort_values([ID,'mjd'])
# log = []
# del dr.df

# multiprocesssor
chunks = np.array_split(multi_obs.index, 4)
uids = [(chunk[0],int(chunk[-1]-1)) for chunk in chunks] # need to check we arent making gaps with the -1
chunks = [multi_obs.loc[uids[i][0]:uids[i][1]] for i in range(4)]
del dr.df
del multi_obs


# +
# For multiprocessing
def multiobs_groupby(chunk):
    
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
                    err_msg = 'error with uid: '+str(uid)+'. Couldnt not average mags at mjd(s): '+str(mjd)
                    print(err_msg)
                    log.append(err_msg)

    #         mag_mean  = -2.5*np.log10(np.mean(10**(-(mag-8.9)/2.5))) + 8.9
                else:
                    mag = mag[mask] # remove points that are 1mag away from the median of the group
                    magerr = magerr[mask]
                    mjd = mjd[mask]

            mjd_mean  = np.mean(mjd)
            magerr_mean = (magerr ** -2).sum() ** -0.5 # sum errors in quadrature
            mag_mean  = -2.5*np.log10(np.average(10**(-(mag-8.9)/2.5), weights = magerr**-2)) + 8.9

        return {'catalogue':cat,'mjd':mjd_mean, 'mag':mag_mean, 'magerr':magerr_mean}
    
    return chunk.groupby([ID,'mjd_floor']).apply(average_nightly_obs).apply(pd.Series).reset_index('mjd_floor', drop=True).astype({'catalogue':'int'})

if __name__ == '__main__':
    pool = Pool(4)
    result = pd.concat(pool.map(multiobs_groupby, chunks))
    
result.to_csv('averaged_qsos')

# +
#For single core
from time import time
start = time()
avgd  = multi_obs.groupby([ID,'mjd_floor']).apply(average_nightly_obs).apply(pd.Series).reset_index('mjd_floor', drop=True).astype({'catalogue':'int'})
end   = time()
print('elapsed: {:.2f}'.format(end-start))

np.savetxt('/disk1/hrb/python/data/merged/{}/{}_band/log.txt'.format(obj,band),np.array(log), fmt='%s')

avgd.to_csv('averaged_qsos.csv') # save intermediate step if having memory issues
# -

avgd = pd.read_csv('averaged_qsos.csv',index_col=ID)

avgd

dr.df = single_obs.append(avgd).sort_values([ID,'mjd'])
# x = x[x['catalogue']==2]

dr.df.index = dr.df.index.astype('int')

dr.df

# ### remove outliers

fig, ax = dr.plot_series(uids, survey=2, filtercodes='r', markersize=0.5)
x2 = x[x['catalogue']==2]
# x2 = x
for uid, axis in zip(uids, ax):
    mjd, mag, magerr = x2.loc[uid,['mjd','mag','magerr']].values.T
    axis.errorbar(mjd, mag, yerr = magerr, lw = 0.5, markersize = 10)
#     axis.set(xlim=[58200,58300])
fig.savefig('averaging_nightly_observations_ztf.pdf', bbox_inches='tight') # test our data cleaning is working

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

dr.df = dr.df.join(grouped[['mag_med','mag_std']], on=ID)
dr.df['Z'] = (dr.df['mag']-dr.df['mag_med'])/dr.df['mag_std']


def remove_outliers(group):
    dmjd1, dmag1 = group[['mjd','mag']].diff(-1).fillna(0).values.T
    dmjd2, dmag2 = group[['mjd','mag']].diff(+1).fillna(0).values.T
    clean = group[~(((abs(dmag1)>0.6) & (abs(dmag2)>0.6)) & ((abs(dmjd1)<500) | (abs(dmjd2)<500)))]
    return clean


from time import time
start = time()
clean_phot = dr.df[['catalogue','mjd','mag','magerr']].groupby(ID).apply(remove_outliers).reset_index(level=1, drop=True)
end   = time()
print('elapsed: {:.2f}s'.format(end-start))

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

# #### def save(args):
#     i, chunk = args
#     f = open('/disk1/hrb/python/data/merged/{}/{}_band/lc_{}.csv'.format(obj,band,i), 'w')
#     comment = '# CSV of cleaned photometry.\n# Data has been averaged nightly and outliers have been removed\n'
#     f.write(comment)
#     chunk.to_csv(f)

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


