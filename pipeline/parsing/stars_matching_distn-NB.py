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
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.preprocessing import colour_transform, parse, data_io, lightcurve_statistics

rdir = cfg.USER.R_DIR
ddir = cfg.USER.D_DIR
wdir = cfg.USER.W_DIR
OBJ = 'calibStars'
ID = 'uid_s'

# +
### **** IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT **** 
###  1) To select sources with reliable photometry in the u and z bands
###     don't forget to require Nobs >= 4
###  2) to avoid a slight bias (~0.02 mag) at the faint end in the gri  
###     bands, require msig*sqrt(Nobs) < 0.03 
### **** IMPORTANT IMPORTANT IMPORTANT IMPORTANT IMPORTANT **** 

#------------------------------------------------------------------------------
# Read in data
#------------------------------------------------------------------------------

# From http://faculty.washington.edu/ivezic/sdss/catalogs/stripe82.html
# cols [1,2,5,9,10,15,16,21,22,27,28,33,34]
# are ra, dec, n_epochs, mag_mean_{ugriz}, mag_mean_err_{ugriz}
# below, we only select {gri}
raw_star_data = pd.read_csv(cfg.USER.D_DIR + 'catalogues/calibStars/stripe82calibStars_v4.2.dat', 
                       sep='\s+',
                       comment='#',
                       usecols=[0,1,2,5,13,15,16,19,21,22,25,27,28,31,33,34],
                       names=['uid_s','ra','dec','n_epochs']+[a+b for b in 'griz' for a in ['n_obs_','mag_mean_','mag_mean_err_']],
                       dtype=cfg.COLLECTION.CALIBSTAR_dtypes)

raw_star_data['uid_s'] = raw_star_data['uid_s'].str[-7:].astype('uint32')
raw_star_data = raw_star_data.set_index('uid_s')

#------------------------------------------------------------------------------
# Save result immediately so we can crossmatch itself in TOPCAT
#    to find pairs of objects that are within 2" of eachother.
#------------------------------------------------------------------------------
# raw_star_data.to_csv(cfg.USER.D_DIR + 'catalogues/calibStars/stripe82calibStars_v4.2.csv')
for b in 'gri':
    raw_star_data['phot_err_'+b] = raw_star_data['mag_mean_err_'+b] * np.sqrt(raw_star_data['n_obs_'+b])
    
#------------------------------------------------------------------------------
# Filter out bad data using bounds from config.py
#------------------------------------------------------------------------------
bounds = {
        **{'mag_mean_'+b:(cfg.PREPROC.FILTER_BOUNDS['mag']) for b in 'gri'},
        **{'mag_mean_err_'+b:(cfg.PREPROC.FILTER_BOUNDS['magerr']) for b in 'gri'},
        **{'n_obs_'+b: (4,np.inf) for b in 'gri'}, # CONDITION (1)
        **{'phot_err_'+b: (0,0.5) for b in 'gri'} # CONDITION (2)
         }

star_data_filtered = parse.filter_data(raw_star_data, bounds, dropna=True,)
print('Star data:')
display(star_data_filtered)

#------------------------------------------------------------------------------
# Read in grouped qsos data
#------------------------------------------------------------------------------
grouped_g = pd.read_csv(ddir + 'merged/qsos/clean/grouped_g.csv', index_col='uid', usecols=['uid','mag_mean']).rename({'mag_mean':'mag_mean_g'}, axis=1)
grouped_r = pd.read_csv(ddir + 'merged/qsos/clean/grouped_r.csv', index_col='uid', usecols=['uid','mag_mean']).rename({'mag_mean':'mag_mean_r'}, axis=1)
grouped_i = pd.read_csv(ddir + 'merged/qsos/clean/grouped_i.csv', index_col='uid', usecols=['uid','mag_mean']).rename({'mag_mean':'mag_mean_i'}, axis=1)

# Note, an outer join will cause grouped_qsos to have some NaN entries
grouped_qsos = grouped_g.join([grouped_r,grouped_i], how='outer')
del grouped_g, grouped_r, grouped_i
print('No. qsos in outer join:', len(grouped_qsos))

#------------------------------------------------------------------------------
# Bin magnitudes of qso and star photometry and count how qsos are in each bin.
#------------------------------------------------------------------------------
counts = []
n=101
for band in 'gri':
    a = pd.DataFrame(pd.cut(raw_star_data['mag_mean_'+band], np.linspace(15,23,n))).reset_index().set_index('mag_mean_'+band)
    b = pd.DataFrame(pd.cut(grouped_qsos['mag_mean_'+band], np.linspace(15,23,n))).value_counts().rename('qso_counts_'+band).reset_index(level=0).set_index('mag_mean_'+band)
    counts.append(a.join(b, on='mag_mean_'+band).reset_index().set_index('uid_s').rename({'mag_mean_'+band:'mag_mean_binned_'+band},axis=1))
star_data_filtered = star_data_filtered.join(counts)
# -

grouped_qsos

grouped_sdss





# + active=""
# plt.style.available

# +
SAVE_FIG = False
SAVE_SUBSAMPLE = False
SAVE_COLORS = False

#------------------------------------------------------------------------------
# Take a subsample of the star data, weighting by the number of qsos in each bin
#------------------------------------------------------------------------------
subsample = star_data_filtered.sample(400000, random_state=42, weights=(sum([star_data_filtered['qso_counts_'+b] for b in 'gri']))).sort_index()

#------------------------------------------------------------------------------
# Remove objects within 2" of eachother (this could be done at the start,
#   but will affect the sampling above and will mean we have to refetch data.
# If we really can be bothered, we can rerun the data collection with the sampling
#   above put at the start of the notebook.  
#------------------------------------------------------------------------------
valid_uids = pd.read_csv(ddir+'catalogues/{}/valid_uids_superset.csv'.format(OBJ), usecols=[ID], index_col=ID, comment='#')
subsample = parse.filter_data(subsample, valid_uids=valid_uids)

#------------------------------------------------------------------------------
# Plot mag distributions
#------------------------------------------------------------------------------
plt.style.use('default')
plt.style.use(cfg.FIG.STYLE_DIR + 'style1.mplstyle')
plt.style.use('seaborn-colorblind')
fig, axes = plt.subplots(3,1, figsize=(18,18))

kwargs = {'bins': n+100, 'edgecolor':'k', 'lw':0.4, 'range':(15,23.5), 'density':False}
for ax, band in zip(axes, 'gri'):
    key = 'mag_mean_'+band
    ax.hist(star_data_filtered[key], alpha=0.3, label='Star full-sample', **kwargs)
    ax.hist(grouped_qsos[key], alpha=0.5, label='Quasar sample', **kwargs)
    ax.hist(subsample[key], alpha=0.5, label='Star sub-sample', **kwargs)
    # ax.hist(raw_star_data[key], alpha=0.4, label='star full sample unfiltered', **kwargs) # Uncomment this to see the effect of parse.filter_data on our data.
    
    ax.set(xlim=[16,23.5], xlabel=r'${}$-band (mag)'.format(band), ylabel='Number of objects')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in [0,2,1]], [labels[i] for i in [0,2,1]], loc='upper left') # [0,2,1] to change the order of the labels in the legend

if SAVE_FIG:
    fig.savefig(rdir + 'plots/calibStars/matching_mag_distributions/seaborn-colorblind.pdf', bbox_inches='tight')
      
if SAVE_SUBSAMPLE:
    savecols = ['ra','dec','n_epochs'] + [a+b for b in 'gri' for a in ['mag_mean_','mag_mean_err_']]
    subsample.to_csv(ddir+'catalogues/calibStars/calibStars_subsample_coords.csv', columns=savecols)
    for i, chunk in enumerate(np.array_split(subsample,2)):
        chunk.to_csv(ddir+'catalogues/calibStars/calibStars_subsample_coords_{}.csv'.format(i), columns=savecols)
    data_io.to_ipac(subsample, ddir+'catalogues/calibStars/calibStars_subsample_coords_ipac.txt', columns=['ra','dec'])
    
#------------------------------------------------------------------------------
# Create columns for colours and save them separately
#------------------------------------------------------------------------------
for b1, b2 in zip('gri','riz'):
    subsample[b1+'-'+b2] = subsample['mag_mean_'+b1] - subsample['mag_mean_'+b2]

if SAVE_COLORS:
    subsample.to_csv(ddir+'computed/calibStars/colors_sdss.csv', columns=['g-r','r-i','i-z'])
