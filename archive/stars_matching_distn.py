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
import matplotlib
font = {'size'   : 14}
matplotlib.rc('font', **font)

from multiprocessing import Pool
def reader(n_subarray):
    return pd.read_csv('/disk1/hrb/python/data/surveys/ztf/calibStars/lc_{}.csv'.format(n_subarray), usecols=[0,1,2,3,4,5,6,7,8],index_col = 'uid_s')#, index_col=['uid_s','filtercode']);


# + active=""
# scaling factor:
# g = 23000
# r = 35500
# i = can't get a proper scaling to match distributions. Too dissimilar.
# -

sdss = pd.read_csv('/disk1/hrb/python/data/surveys/sdss/calibStars/SDSS_S82_calibStars_lcs.dat', sep='\s+', usecols = np.array([0,2,3,9,10,12,15,16,18,21,22,24,27,28,30,33,34,36]), index_col='uid_s')
sdss.index.rename('uid_s')
scaling_dict = {'g':24000,'r':35500,'i':35500}

# +
# band = 'r'
idxs_list = [[],[],[]]
fig, ax = plt.subplots(1,3, figsize = (20,10))
for j,band in enumerate('gri'):
    
    #read in and plot qso disbribution for given band
    quantity = 'mu'
    scaling = scaling_dict[band]
    n, bins = np.histogram(sdss[band + '_m'+quantity], bins=200,range=(15,24))
    n_qso, bins_qso = pd.read_csv('computed/qso_mag_dist_nbins_{}.csv'.format(band)).values.T
    n_qso = n_qso[:-1]
    
    #match star distribution to qso distribution
    sample_size = np.round(n_qso*scaling).astype('int')
    mag = sdss[band + '_m'+quantity].values
    masks = np.array([(bins_qso[i] < mag) & (mag < bins_qso[i+1]) for i in range(len(bins_qso)-1)])
    idxs = np.empty(0, dtype='int')
    for i in range(200):   
        idxs = np.append(idxs, np.random.choice(sdss[masks[i]].index, size=np.minimum(len(sdss[masks[i]].index),sample_size[i]))) #take a subset of indices to match qso dist. Return min number_of_stars_in_bin, qso_dsnt
#     n_matched, _ = np.histogram(sdss.loc[idxs, band + '_m'+quantity], bins=200,range=(15,24))
    
    
    
    
    
    ax[j].plot(0.5*(bins_qso[:-1]+bins_qso[1:]),n_qso*scaling, color = 'b', label = 'qsos')
    ax[j].plot(0.5*(bins[1:]+bins[:-1]), n, label = 'stars_full')
#     ax[j].plot(0.5*(bins[1:]+bins[:-1]), n_matched, color = 'r', label='stars_matched')
    
    
#     print(f'{band}: Sample size intial: \t{n.sum():,}\n{band}: Sample size final: \t{n_matched.sum():,}\n{band}: Number removed: \t{n.sum()-n_matched.sum():,}\n')
    
    idxs_list[j] = idxs
    
# idxs = np.loadtxt('computed/calibStars_uids_distn_matched_ztf.csv').astype('uint')
total_idxs = np.union1d(idxs_list[0], idxs_list[1])
# total_idxs = idxs_list[1]

for j,band in enumerate('gri'):
    
    n_matched, _ = np.histogram(sdss.loc[total_idxs, band + '_m'+quantity], bins=200,range=(15,24))
    ax[j].plot(0.5*(bins[1:]+bins[:-1]), n_matched, color = 'r', label='stars_matched')
    ax[j].legend()
    ax[j].set(ylim=[0,17500])
    
# total_idxs = np.random.choice(total_idxs, size = 530000)
# for j,band in enumerate('gri'):
    
#     n_matched, _ = np.histogram(sdss.loc[total_idxs, band + '_m'+quantity], bins=200,range=(15,24))
#     ax[j].plot(0.5*(bins[1:]+bins[:-1]), n_matched, color = 'k', label='stars_matched')
#     ax[j].legend()
#     ax[j].set(ylim=[0,17500])
fig.savefig('matching_star_distn.pdf',bbox_inches='tight')
# -

np.savetxt('computed/calibStars_uids_distn_matched.csv',total_idxs, fmt='%i')
