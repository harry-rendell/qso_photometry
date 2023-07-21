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
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

norm = rnd.normal()
poiss = rnd.poisson(lam=5, size=50000)/2e4*18**2

plt.hist(poiss, bins=21, range=(0,.3))


def single_core(sub_df):
#             if uids == None:
#                 sub_df = self.df[['mjd', 'mag', 'magerr', 'cat']]
#             else:
#                 sub_df = self.df[['mjd', 'mag', 'magerr', 'cat']].loc[uids]

    df = pd.DataFrame(columns=['uid', 'dt', 'dm', 'de', 'cat'])
    for uid, group in sub_df.groupby('uid'):
        #maybe groupby then iterrows? faster?
        mjd_mag = group[['mjd','mag']].values
        magerr = group['magerr'].values
        cat	 = group['cat'].values
        n = len(mjd_mag)
        # dtdm defined as: ∆m = (m2 - m1), ∆t = (t2 - t1) where (t1, m1) is the first obs and (t2, m2) is the second obs.
        # Thus a negative ∆m corresponds to a brightening of the object
        unique_pair_indicies = np.triu_indices(n,1)

        dcat = 3*cat + cat[:,np.newaxis]
        dcat = dcat[unique_pair_indicies]

        dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
        dtdm = dtdm[unique_pair_indicies]
        dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

        dmagerr = ( magerr**2 + magerr[:,np.newaxis]**2 )**0.5
        dmagerr = dmagerr[unique_pair_indicies]

        duid = np.full(int(n*(n-1)/2),uid,dtype='uint32')
        # collate data to DataFrame and append
        df = df.append(pd.DataFrame(data={'uid':duid,'dt':dtdm[:,0],'dm':dtdm[:,1], 'de':dmagerr,'cat':dcat}))

        if (uid % 500 == 0):
            print(uid)
    return df


def mock_lc_flux(n_obj, flux_mean=1e-4, flux_std=6e-5):
    fluxmeans = np.random.normal(flux_mean, flux_std, size=n_obj)
    N   = np.random.randint(100,200, size=n_obj)
    mag_fn = lambda flux: -2.5*np.log10(flux)+8.9
    
    def fn(n, fluxmean):
        mjd = np.sort(np.random.uniform(5000,6000, size=n))
        mag = mag_fn(np.random.normal(fluxmean, 0.00001, size=n))
        cat = np.sort(np.random.randint(1,4, size=n))
        return mjd, mag, cat
    
    df = pd.DataFrame(columns=['uid','mjd','mag','cat'])
    for i in range(n_obj):
        mjd, mag, cat = fn(N[i], fluxmeans[i])
        uid = np.full(N[i], i, dtype='uint32')
        df = df.append(pd.DataFrame({'uid':uid,'mjd':mjd, 'mag':mag, 'cat':cat}))
    
    df = df.set_index('uid')
    return df


class simulation():
    
    def __init__(self, n_obj, mag_mean=18.9, mag_std=2):
        self.n_obj = n_obj
        self.mag_mean = mag_mean
        self.mag_std = mag_std
        
    
    def mock_lc(self):
        
        magmeans = np.random.normal(self.mag_mean, self.mag_std, size=self.n_obj)
        N   = np.random.randint(100,200, size=self.n_obj)
        mag_fn = lambda flux: flux

        def fn(n, magmean):
            mjd = np.sort(np.random.uniform(5000,6000, size=n))
            magerr = rnd.poisson(lam=5, size=n)/2e4 * (magmean ** 2)
            mag = np.random.normal(magmean, magerr, size=n)
            flux= mag_fn(mag)
            cat = np.random.randint(1,4, size=n)
            return mjd, mag, magerr, cat

        df = pd.DataFrame(columns=['uid','mjd','mag','cat'])
        for i in range(self.n_obj):
            mjd, mag, magerr, cat = fn(N[i], magmeans[i])
            uid = np.full(N[i], i, dtype='uint32')
            df = df.append(pd.DataFrame({'uid':uid,'mjd':mjd, 'mag':mag, 'magerr':magerr, 'cat':cat}))

        self.df = df.set_index('uid')
    
    def calculate_dtdm(self, uids=None):
        if __name__ == '__main__':
            p = Pool(4)
            df_list = p.map(single_core, np.array_split(self.df[['mjd', 'mag', 'magerr', 'cat']], 4))

        self.dtdm = pd.concat(df_list)
        
    def histograms(self):
        fig, ax = plt.subplots(1,2, figsize=(18,6))
        
        self.dtdm.hist('dm', ax=ax[0], bins=200, range=(-1,1))
        self.dtdm.hist('de', ax=ax[1], bins=100, range=(0,0.4))

sim = simulation(500)

sim.mock_lc()

sim.calculate_dtdm()

sim.dtdm

sim.histograms()


def plot(x,y,err):
    fig, ax = plt.subplots(1,1, figsize=(18,5))
    ax.errorbar(x, y, err, lw = 0.2, markersize = 3, marker='o')
    ax.invert_yaxis()
    ax.set(xlabel='mjd', ylabel='mag')


plot(mjd,mag,magerr)

# # SF Errors

from scipy.stats import chi2

dm = rnd.normal(0,10,size=100000)
dm2 = dm**2

fig, ax = plt.subplots(1,1, figsize=(18,9))
ax.hist(dm, bins=200);

fig, ax = plt.subplots(1,1, figsize=(18,9))
ax.hist(dm2, bins=100, range=(0,5));

dm.var()

dm2.var()

2*dm.var()**2

# note the last two outputs are very similar, showing that var(∆m^2) = 2var(∆m)^2
