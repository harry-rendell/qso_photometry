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

# + language="bash"
# jupytext --to py parsing_PS-NB.ipynb # Only run this if the notebook is more up-to-date than -NB.py
# # jupytext --to --update ipynb parsing_PS-NB.ipynb # Run this to update the notebook if changes have been made to -NB.py
# -

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", ".."))
from module.config import cfg
from module.preprocessing import parse, data_io

OBJ    = 'qsos'
ID     = 'uid' if OBJ == 'qsos' else 'uid_s'
BAND   = 'r'
wdir = cfg.W_DIR
ddir = cfg.D_DIR

cols = [ID, 'objID_ps', 'filter', 'obsTime', 'psfFlux', 'psfFluxErr']
# ps_secondary = pd.read_csv(cfg.D_DIR + 'surveys/ps/{}/ps_secondary.csv'.format(OBJ), dtype=cfg.COLLECTION.PS.dtypes, nrows=None, usecols=cols).set_index(ID).rename({'filter':'filtercode'})
ps_neighbours = pd.read_csv(cfg.D_DIR + 'surveys/ps/{}/ps_neighbours_2arcsec.csv'.format(OBJ), dtype=cfg.COLLECTION.PS.dtypes).set_index(ID)
ps_neighbours['sep'] *= 60

ps_neighbours['sep'].max()

# Save distance column
ps_neighbours['sep'].to_csv(os.path.join(cfg.RES_DIR, 'plot_data', f'distances_ps_{OBJ}.csv'))

valid_uids = pd.read_csv(ddir+'catalogues/{}/{}_subsample_coords.csv'.format(OBJ,OBJ), usecols=[ID], index_col=ID, comment='#')
ps_secondary = parse.filter_data(ps_secondary, valid_uids=valid_uids)
ps_neighbours = parse.filter_data(ps_neighbours, valid_uids=valid_uids)

# We are querying StackObjectThin thus we expect a one to one match, however, sometimes additional IDs are returned. We filter these out. There are about 421 of these duplicates.

CHECK_MAX_SEP = True
if CHECK_MAX_SEP:
    # Note, we queried PanSTARRS for objects within 1". To check this, we can join the coord query table from mastweb and sort by separation to show sep<1"
    ps_secondary_merged = ps_secondary.join(ps_neighbours, on=ID, lsuffix='_ps')
    display(ps_secondary_merged.sort_values('sep', ascending=False))

CHECK_FOR_NON_UNIQUE_MATCHES = True
if CHECK_FOR_NON_UNIQUE_MATCHES:
    # Display cases where both ID and objID_ps are duplicated
    print('-'*50)
    print('Duplicate rows:')
    mask = ps_neighbours.reset_index().duplicated([ID,'objID_ps'], keep=False).values
    display(ps_neighbours[mask])
    ps_neighbours_no_duplicates = ps_neighbours[~mask]
    
    # Display cases where one uid matches to multiple PS objIDs, or vice versa
    print('-'*50)
    print('Non 1-to-1 matches:')
    mask1 = ps_neighbours_no_duplicates.index.duplicated(keep=False)
    mask2 = ps_neighbours_no_duplicates.duplicated('objID_ps', keep=False)
    display(ps_neighbours_no_duplicates[(mask1 ^ mask2)])

# Given we have duplicates/non unique matches above, remove them here by selecting the closest objID for each uid.
ps_neighbours_sorted = ps_neighbours.sort_values('sep', ascending=True)
mask = ps_neighbours_sorted.index.duplicated(keep='first')
print('Number of duplicates and non-unique matches removed: {:,}'.format(mask.sum()))
print('Number of unique matches: {:,}'.format((~mask).sum()))
ps_primary_matches = ps_neighbours_sorted[~mask].sort_values(ID)
assert ps_primary_matches.index.is_unique
assert ps_primary_matches['objID_ps'].is_unique

objIDs_ps_to_keep = ps_primary_matches['objID_ps'].values

mask = ps_secondary['objID_ps'].isin(objIDs_ps_to_keep)
print('Number of non-primary objID observations: {:,}'.format((~mask).sum()))
df_ps = ps_secondary[mask]

print('Number of unique matches for which we have photometry: {:,}'.format(len(df_ps['objID_ps'].unique())))

# ### Convert fluxes to mags

df_ps = df_ps[df_ps['psfFlux']!=0]
df_ps = df_ps.rename(columns = {'obsTime': 'mjd', 'filter': 'filtercode'})
df_ps['mag'] = -2.5*np.log10(df_ps['psfFlux']) + 8.90
df_ps['magerr'] = 1.086*df_ps['psfFluxErr']/df_ps['psfFlux']
df_ps = df_ps.drop(['psfFlux','psfFluxErr','objID_ps'], axis = 1)
df_ps = df_ps.set_index('filtercode', append=True)#.astype(np.float32)

# # Basic stats
# ---



print(df_ps.head(20).to_latex(caption='<caption>',label='<label>', float_format='%.3f'))

# # Save data
# ---

# +
# Add comment to start of csv file
comment =  ("# CSV of photometry with no other preprocessing or cleaning.\n"
            "# mag : photometry in native PanSTARRS photometric system")

for band in 'griz':
    chunks = parse.split_into_non_overlapping_chunks(df_ps.loc[pd.IndexSlice[:, band],:].droplevel('filtercode'), 4)
    # keyword arguments to pass to our writing function
    kwargs = {'comment':comment,
              'basepath':cfg.D_DIR + 'surveys/ps/{}/unclean/{}_band/'.format(OBJ, band)}

    data_io.dispatch_writer(chunks, kwargs)

# +
# Finding mismatched objects
# prim_sec = pd.merge(  primary[['uid','ra_ps','dec_ps','objID_ps','sep','sep_upper']],
#                     secondary[['uid','ra_ps','dec_ps','objID_ps','sep','sep_upper']], on='uid', suffixes = ('_p','_s'))
# prim_sec['sep_diff_arcsec'] = (prim_sec.sep_s - prim_sec.sep_p)*60.0
# plt.figure(figsize=(15, 4))
# plt.xticks(np.linspace(0,2.5,26))
# plt.hist(prim_sec['sep_diff'], bins = 70);
# -

# There are about 30 objects (ones with sep_diff < ~0.3 arcsec) which did not stack properly - these have duplicate object IDs, however it is much smaller than our entire sample so we will just use a single objID, whichever has a closer ra,dec to the reference coords. Run the code above to see this.
