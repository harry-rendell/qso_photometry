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
path = '/disk1/hrb/python/'
obj = 'qsos'
ID = 'uid'

# Query used to obtain data (context = DR16):

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
# cross apply dbo.fGetNearbyObjEq(q.ra, q.dec, 0.05) as nb
#
# join PhotoPrimary p on p.objid=nb.objid
# join specobj as s on s.bestobjid=p.objid

# +
datatype = {'uid': np.uint32,   'objID'  : np.uint64, 
            'ra' : np.float128, 'ra_ref' : np.float128,
            'dec': np.float128, 'dec_ref': np.float128,
            'get_nearby_distance': np.float64}

sdss_data = pd.read_csv(path + 'data/retrieved_data/dr14q_nearby_3arcsec_class.csv', dtype = datatype, float_precision = 'round trip')
# -

print('shape:',sdss_data.shape)
sdss_data.head()

sdss_data['dec'].hist(bins=100)

#For some reason, duplicate rows are returned. Drop these.
sdss_data.drop_duplicates(subset = ['ra','dec'], keep = 'first', inplace = True)
sdss_data.sort_values('get_nearby_distance',inplace = True)
assert sdss_data['objID'].is_unique, 'objID is not unique'

# +
mask = sdss_data.duplicated(subset = 'uid',keep = 'first')
sdss_qso = sdss_data[~mask]
sdss_nb  = sdss_data[mask][['uid','objID','ra','dec','class','get_nearby_distance']]
#removes other, further, neighbours
sdss_nb.drop_duplicates(subset = 'uid',inplace = True)
merged = pd.merge(sdss_qso,sdss_nb, on = 'uid', how = 'left', suffixes = ('','_nb'))
#Keep objects that have class == QSO or have <0.3" sep to reference coords (0.3" and 1" give the same thing - sanity check)
merged = merged[(merged['class'] == 'QSO') | (merged['get_nearby_distance'] < 0.005)]
merged.head()
#Merged is a table containing the matched objects and their closest neighbour (if there is one within 3", o/w NaN)

#WARNING
# Class  | number | % of total
# ----------------------------
# QSO    | 524111 | 99.73093573
# Galaxy | 1173   | 0.22320537
# Star   | 241    | 0.0458589
# Total  | 525525 | 1

# +
#Merge sdss qsos with our original dr14q list
dr14q_coords = pd.read_csv(path + 'data/catalogues/qsos/dr14q/dr14q_uid_coords.csv', dtype = {'ra': np.float128, 'dec': np.float128, 'uid': np.uint64})
merged.sort_values('uid',inplace = True)

new_search = merged[['uid','class','get_nearby_distance_nb']]
new_search_dr14q = pd.merge(dr14q_coords, new_search, how = 'left', on = 'uid')
new_search_dr14q = new_search_dr14q.fillna(0.05)
new_search_dr14q['get_nearby_distance_nb'] += -0.01
new_search_dr14q = new_search_dr14q.rename(columns={'get_nearby_distance_nb': 'sep'})

#We join the upper search limit onto our original dr14q_coords list to prevent getting neighbours.
#Save this as dr14q_coords_unique.csv
drqsdss = pd.merge(dr14q_coords, merged[['uid','objID','class']], how = 'left', on = 'uid')
drqsdss.fillna(0.0, inplace = True)
drqsdss = drqsdss.astype({'objID': 'int64'})
# drqsdss.to_csv(path + 'coords/dr14q_coords_unique.csv', index = False)
# -

sdss_dr14q_filtered = merged[['uid','objID','ra_ref','dec_ref','get_nearby_distance_nb']]
sdss_dr14q_filtered = sdss_dr14q_filtered.fillna(0.05)
sdss_dr14q_filtered['get_nearby_distance_nb'] += -0.01 # subtract a small amount from the upper search radius to prevent picking neighbours.
#remove 4 qsos because they don't match up

sdss_dr14q_filtered.to_csv(path + 'data/catalogues/qsos/dr14q/sdss_secondary_search_coords.csv', index = False)
sdss_dr14q_filtered

sdss_dr14q_filtered['get_nearby_distance_nb'].sort_values()

# Distribution of search radius
fig, ax = plt.subplots()
sdss_dr14q_filtered['get_nearby_distance_nb'].hist(bins=100, ax=ax)
ax.set(yscale='log')

# # Parsing secondary observations

# Query used to find secondary observations:
#

# + active=""
# select
#
# q.uid as uid, p.objID,
# q.ra, q.dec,
# nb.distance as get_nearby_distance,
#
# p.psfMag_u as upsf, p.psfMagErr_u as upsferr,
# p.psfMag_g as gpsf, p.psfMagErr_g as gpsferr,
# p.psfMag_r as rpsf, p.psfMagErr_r as rpsferr,
# p.psfMag_i as ipsf, p.psfMagErr_i as ipsferr,
# p.psfMag_z as zpsf, p.psfMagErr_z as zpsferr,
# f.mjd_r
#
# into mydb.dr14q_secondary
# from mydb.dr14q_uid_coords q
# cross apply dbo.fGetNearbyObjAllEq(q.ra, q.dec, 1.0/60) as nb
#
# join photoobj p on p.objid=nb.objid
# join field f on f.fieldid=p.fieldid
#   
# ORDER BY uid ASC, mjd_r ASC
# -

dtype1 =   {'uid': np.uint32, 'objID'  : np.uint64, 
            'ra' : np.float64, 'ra_ref' : np.float64,
            'dec': np.float64, 'dec_ref': np.float64,
            'get_nearby_distance': np.float64}
dtype2 = {band + 'psf'   : np.float64 for band in 'ugriz'}
dtype3 = {band + 'psferr': np.float64 for band in 'ugriz'}
cols = [ID] + [x for y in zip(['mag_'+b for b in 'griz'], ['magerr_'+b for b in 'griz']) for x in y] + ['mjd_r','get_nearby_distance']


# +
sdss_unmelted = pd.read_csv(path+'data/surveys/sdss/{}/sdss_secondary_unmelted.csv'.format(obj), index_col=ID, usecols=cols, dtype = {**dtype1, **dtype2, **dtype3})

# make observation ID
sdss_unmelted['obsid'] = range(1,len(sdss_unmelted)+1)

# Rename mjd_r to mjd
sdss_unmelted = sdss_unmelted.rename(columns={'mjd_r':'mjd'})

# Add columns for colors
for b1, b2 in zip('gri','riz'):
    sdss_unmelted[b1+'-'+b2] = sdss_unmelted['mag_'+b1] - sdss_unmelted['mag_'+b2]

df_sdss_unpivot1 = pd.melt(sdss_unmelted, id_vars = 'obsid', value_vars = ['mag_'   +b for b in 'griz'], var_name = 'filtercode', value_name = 'mag')
df_sdss_unpivot2 = pd.melt(sdss_unmelted, id_vars = 'obsid', value_vars = ['magerr_'+b for b in 'griz'], var_name = 'filtercode', value_name = 'magerr')

df_sdss_unpivot1['filtercode'] = df_sdss_unpivot1['filtercode'].str[-1]
df_sdss_unpivot2['filtercode'] = df_sdss_unpivot2['filtercode'].str[-1]

sdss_melted = pd.merge(sdss_unmelted.reset_index()[[ID,'obsid','mjd']], pd.merge(df_sdss_unpivot1, pd.merge(df_sdss_unpivot2, sdss_unmelted[['obsid','g-r','r-i','i-z']], on='obsid'), on = ['obsid','filtercode']), on = 'obsid').set_index([ID,'filtercode']).drop('obsid',axis=1)
# -

# Save output from above
sdss_melted.to_csv(path+'data/surveys/sdss/{}/sdss_secondary.csv'.format(obj))

# For some reason, our upper search limit is not working when we have nearby objects. Why? Perhaps due to class? SDSS why u no find nearby obj. If it just doesnt work then we will just cut objects with displacement >1"

# # Making sure mjd_ugriz does not vary by more than 1 day

df_ugriz = pd.read_csv(path + 'retrieved_data/dr14q_secondary_mjdfloat_test.csv')
std = df_ugriz[['mjd_' + band for band in 'ugriz']].std(axis=1)
std.nlargest(5)

# Observations in different bands are taken within 0.001311 days (2 mins) of eachother. Thus we can take a single band timestamp.

'{:0.20f}'.format((np.array([54741.371761], dtype = np.float64))[0])
