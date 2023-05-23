import numpy as np
import pandas as pd
import os
from ..config import cfg
from .parse import split_into_non_overlapping_chunks

def groupby_dtdm_between(df, args):
    for index, group in df.groupby(df.index.name):
        return calculate_dtdm_between_surveys(group, *args)
    
def groupby_dtdm_within(df):
    df_list = []
    for index, group in df.groupby(df.index.name):
        df_list.append(calculate_dtdm_within_surveys(group))
    return pd.concat(df_list, ignore_index=True)

def calculate_dtdm_between_surveys(group, sid_1, sid_2):
    mask = (group['sid']==sid_1).values
    n = mask.sum()*(~mask).sum()

    mjd_mag = group[['mjd','mag']].values
    magerr  = group['magerr'].values
    
    dtdm = mjd_mag[mask, np.newaxis] - mjd_mag[~mask]
    dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

    dmagerr = ( magerr[mask, np.newaxis]**2 + magerr[~mask]**2 )**0.5
    uid = np.full(n,group.index[0],dtype='uint32')

    return pd.DataFrame({'uid':uid, 'dt':dtdm[:,:,0].ravel(), 'dm':dtdm[:,:,1].ravel(), 'de':dmagerr.ravel(), 'dm2_de2': (dtdm[:,:,1]**2 - dmagerr**2).ravel()})

def calculate_dtdm_within_surveys(group):
    mjd_mag = group[['mjd','mag']].values
    magerr = group['magerr'].values
    n = len(mjd_mag)

    unique_pair_indicies = np.triu_indices(n,1)

    dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
    dtdm = dtdm[unique_pair_indicies]
    dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

    dmagerr = ( magerr**2 + magerr[:,np.newaxis]**2 )**0.5
    dmagerr = dmagerr[unique_pair_indicies]
    
    uid = np.full(n*(n-1)//2,group.index[0],dtype='uint32')
    
    return pd.DataFrame(data={'uid':uid,'dt':dtdm[:,0],'dm':dtdm[:,1], 'de':dmagerr})

def groupby_save_pairwise(df, kwargs):
    if not (('basepath' in kwargs) and ('fname' in kwargs)):
        raise Exception('Both basepath and fname must be provided')
    
    # Use appropriate time (rest frame or observer frame)
    mjd_key = kwargs['mjd_key'] if ('mjd_key' in kwargs) else 'mjd'

    # If a subset of objects has been provided, restrict our DataFrame to that subset.
    if 'subset' in kwargs:
        df = df[df.index.isin(kwargs['subset'])]

    # Remove observations that are not in the specified band
    df = df[df['band']==kwargs['band']]

    output_dir = os.path.join(kwargs['basepath'], 'dtdm_'+kwargs['band'])
    os.makedirs(output_dir, exist_ok=True)
    output_fpath = os.path.join(output_dir, kwargs['fname'].replace('lc','dtdm'))
    
    if not os.path.exists(output_fpath):
        with open(output_fpath, 'w') as file:
            file.write(','.join([kwargs['ID'],'dt','dm','de','dsid']) + '\n')
    
    n_chunks = len(df)//40000 + 1 # May need to reduce this down to, say, 30,000 if the memory usage is too large.
    for i, chunk in enumerate(split_into_non_overlapping_chunks(df, n_chunks)):
        # If multiple bands are provided, iterate through them.
        s = chunk.groupby(kwargs['ID']).apply(calculate_dtdm, mjd_key)
        s = pd.DataFrame(s.values.tolist(), columns=s.columns, dtype='float32').astype({k:v for k,v in kwargs['dtypes'].items() if k in s.columns})
        s.to_csv(output_fpath, index=False, header=False, mode='a')
        del s
    print('finished processing file:',kwargs['fname'], flush=True)

def calculate_dtdm(group, mjd_key):
    mjd_mag = group[[mjd_key,'mag']].values
    magerr = group['magerr'].values
    sid = group['sid'].values
    n = len(mjd_mag)

    unique_pair_indicies = np.triu_indices(n,1)
    
    dsid = sid*sid[:,np.newaxis]
    dsid = dsid[unique_pair_indicies]

    dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
    dtdm = dtdm[unique_pair_indicies]
    dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

    dmagerr = ( magerr**2 + magerr[:,np.newaxis]**2 )**0.5
    dmagerr = dmagerr[unique_pair_indicies]
    
    uid = np.full(n*(n-1)//2,group.index[0],dtype='uint32')
    
    return pd.DataFrame(data={group.index.name:uid, 'dt':dtdm[:,0], 'dm':dtdm[:,1], 'de':dmagerr, 'dsid':dsid})
