# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.preprocessing import colour_transform, parse, data_io

OBJ    = 'calibStars'
ID     = 'uid_s'
wdir = cfg.USER.W_DIR

# Query used to obtain data (context = DR16):

# # The code below tries to find the closest nearby object to the dr14q coords.
# It produces `sdss_secondary_search_coords.csv` which is uploaded to casjobs and used for querying. The real parsing of secondary observations happens later in the notebook.

# + active=""
# select
#
# q.uid as uid, p.objID, 
# p.ra as ra, p.dec as dec,
# q.ra as ra_ref, q.dec as dec_ref,
# s.class as class,
# nb.distance as get_nearby_distance
#
# into mydb.dr14q_nearby_w_class
# from mydb.dr14q_uid_coords q
# cross apply dbo.fGetNearbyObjEq(q.ra, q.dec, 1.0/60.0) as nb
#
# join PhotoPrimary p on p.objid=nb.objid
# join specobj as s on s.bestobjid=p.objid

# + active=""
# SAVE = False
#
# datatype = {'uid': np.uint32,   'objID'  : np.uint64, 
#             'ra' : np.float128, 'ra_ref' : np.float128,
#             'dec': np.float128, 'dec_ref': np.float128,
#             'get_nearby_distance': np.float64}
#
# sdss_data = pd.read_csv(cfg.USER.D_DIR + 'retrieved_data/dr14q_nearby_3arcsec_class.csv', dtype = datatype, float_precision = 'round trip')
#
# print('shape:',sdss_data.shape)
# display(sdss_data.head())
#
# #For some reason, duplicate rows are returned. Drop these.
# sdss_data.drop_duplicates(subset = ['ra','dec'], keep = 'first', inplace = True)
# sdss_data.sort_values('get_nearby_distance',inplace = True)
# assert sdss_data['objID'].is_unique, 'objID is not unique'
#
# mask = sdss_data.duplicated(subset = 'uid',keep = 'first')
# sdss_qso = sdss_data[~mask]
# sdss_nb  = sdss_data[mask][['uid','objID','ra','dec','class','get_nearby_distance']]
# #removes other, further, neighbours
# sdss_nb.drop_duplicates(subset = 'uid',inplace = True)
# merged = pd.merge(sdss_qso,sdss_nb, on = 'uid', how = 'left', suffixes = ('','_nb'))
# #Keep objects that have class == QSO or have <0.3" sep to reference coords (0.3" and 1" give the same thing - sanity check)
# merged = merged[(merged['class'] == 'QSO') | (merged['get_nearby_distance'] < 0.005)]
# merged.head()
# #Merged is a table containing the matched objects and their closest neighbour (if there is one within 3", o/w NaN)
#
# #WARNING
# # Class  | number | % of total
# # ----------------------------
# # QSO    | 524111 | 99.73093573
# # Galaxy | 1173   | 0.22320537
# # Star   | 241    | 0.0458589
# # Total  | 525525 | 1
#
# #Merge sdss qsos with our original dr14q list
# dr14q_coords = pd.read_csv(cfg.USER.D_DIR + 'catalogues/qsos/dr14q/dr14q_uid_coords.csv', dtype = {'ra': np.float32, 'dec': np.float32, 'uid': np.uint64})
# merged.sort_values('uid',inplace = True)
#
# new_search = merged[['uid','class','get_nearby_distance_nb']]
# new_search_dr14q = pd.merge(dr14q_coords, new_search, how = 'left', on = 'uid')
# new_search_dr14q = new_search_dr14q.fillna(0.05)
# new_search_dr14q['get_nearby_distance_nb'] += -0.01
# new_search_dr14q = new_search_dr14q.rename(columns={'get_nearby_distance_nb': 'sep'})
#
# #We join the upper search limit onto our original dr14q_coords list to prevent getting neighbours.
# #Save this as dr14q_coords_unique.csv
# drqsdss = pd.merge(dr14q_coords, merged[['uid','objID','class']], how = 'left', on = 'uid')
# drqsdss.fillna(0.0, inplace = True)
# drqsdss = drqsdss.astype({'objID': 'int64'})
# if SAVE:
#     drqsdss.to_csv(wdir + 'coords/dr14q_coords_unique.csv', index = False)
#     
#     
# sdss_dr14q_filtered = merged[['uid','objID','ra_ref','dec_ref','get_nearby_distance_nb']]
# sdss_dr14q_filtered = sdss_dr14q_filtered.fillna(0.05)
# sdss_dr14q_filtered['get_nearby_distance_nb'] += -0.01 # subtract a small amount from the upper search radius to prevent picking neighbours.
# #remove 4 qsos because they don't match up
#
# if SAVE:
#     sdss_dr14q_filtered.to_csv(cfg.USER.D_DIR + 'catalogues/qsos/dr14q/sdss_secondary_search_coords.csv', index = False)
# sdss_dr14q_filtered
# -

# # Parsing secondary observations

# +
cols = [ID, 'objID'] + [x for y in zip(['mag_'+b for b in 'griz'], ['magerr_'+b for b in 'griz']) for x in y] + ['mjd','get_nearby_distance']
sdss_unmelted = pd.read_csv(cfg.USER.D_DIR + 'surveys/sdss/{}/sdss_secondary.csv'.format(OBJ), usecols=cols, dtype = cfg.COLLECTION.SDSS.dtypes)

sdss_unmelted = sdss_unmelted.drop_duplicates(subset=[ID,'objID','mjd']).set_index(ID)
sdss_unmelted['get_nearby_distance'] *= 60

valid_uids = pd.read_csv(cfg.USER.D_DIR + 'catalogues/{}/{}_subsample_coords.csv'.format(OBJ,OBJ), usecols=[ID], index_col=ID, comment='#')
sdss_unmelted = parse.filter_data(sdss_unmelted, valid_uids=valid_uids)

# Add columns for colors
for b1, b2 in zip('gri','riz'):
    sdss_unmelted[b1+'-'+b2] = sdss_unmelted['mag_'+b1] - sdss_unmelted['mag_'+b2]

SAVE_COLORS = True
if SAVE_COLORS:
    colors = sdss_unmelted[['g-r','r-i','i-z']].groupby(ID).agg('mean')
    colors.to_csv(cfg.USER.D_DIR + 'computed/{}/colors_sdss.csv'.format(OBJ))

df_sdss_unpivot1 = pd.melt(sdss_unmelted, id_vars = 'objID', value_vars = ['mag_'   +b for b in 'griz'], var_name = 'filtercode', value_name = 'mag')
df_sdss_unpivot2 = pd.melt(sdss_unmelted, id_vars = 'objID', value_vars = ['magerr_'+b for b in 'griz'], var_name = 'filtercode', value_name = 'magerr')

df_sdss_unpivot1['filtercode'] = df_sdss_unpivot1['filtercode'].str[-1]
df_sdss_unpivot2['filtercode'] = df_sdss_unpivot2['filtercode'].str[-1]

sdss_melted = pd.merge(sdss_unmelted.reset_index()[[ID,'objID','mjd']], pd.merge(df_sdss_unpivot1, pd.merge(df_sdss_unpivot2, sdss_unmelted[['objID','g-r','r-i','i-z']], on='objID'), on = ['objID','filtercode']), on = 'objID').set_index([ID,'filtercode']).drop('objID',axis=1)
# -

sdss_melted

# # Transform to PanSTARRS
# ---

sdss_transformed = colour_transform.transform_sdss_to_ps(sdss_melted, color='g-r', system='tonry').sort_values(['uid_s','mjd']).astype(np.float32)

sdss_transformed.isna().sum()

# # Save data
# ---

# +
# Add comment to start of csv file
comment = ( "# CSV of photometry transformed to PS with no other preprocessing or cleaning.\n"
            "# mag      : transformed photometry in PanSTARRS photometric system\n"
            "# mag_orig : original photometry in native SDSS photometric system")

for band in 'griz':
    chunks = parse.split_into_non_overlapping_chunks(sdss_transformed.loc[pd.IndexSlice[:, band],:].droplevel('filtercode'), 4)
    # keyword arguments to pass to our writing function
    kwargs = {'comment':comment,
              'basepath':cfg.USER.D_DIR + 'surveys/sdss/{}/unclean/{}_band/'.format(OBJ, band)}

    data_io.dispatch_writer(chunks, kwargs)
# -

# ___
# ### Extra checks: Making sure mjd_ugriz does not vary by more than 1 day
# ___

# The following shoes that observations in different bands are taken within 0.001311 days (2 mins) of eachother. Thus we can take a single band timestamp.

CHECK_TIME_DIFF = False
if CHECK_TIME_DIFF:
    df_ugriz = pd.read_csv(path + 'retrieved_data/dr14q_secondary_mjdfloat_test.csv')
    std = df_ugriz[['mjd_' + band for band in 'ugriz']].std(axis=1)
    std.nlargest(5)
