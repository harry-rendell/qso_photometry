import numpy as np
import pandas as pd
import os
from ..config import cfg

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
    
    dm2_de2 = dtdm[:,1]**2 - dmagerr**2

    uid = np.full(n*(n-1)//2,group.index[0],dtype='uint32')
    
    return pd.DataFrame(data={'uid':uid,'dt':dtdm[:,0],'dm':dtdm[:,1], 'de':dmagerr, 'dm2_de2':dm2_de2})

def groupby_save_pairwise(df, kwargs):
    if not (('basepath' in kwargs) and ('fname' in kwargs)):
        raise Exception('Both basepath and fname must be provided')
    if ('band' in kwargs) & ('band' in df.columns):
        for b in kwargs['band']:
            
            output_dir = os.path.join(kwargs['basepath'], 'dtdm_'+b)
            os.makedirs(output_dir, exist_ok=True)
            
            # If multiple bands are provided, iterate through them.
            print(b,'band - processing:',kwargs['fname'])
            s = df[df['band'] == b].groupby(df.index.name).apply(calculate_dtdm)
            s = pd.DataFrame(s.values.tolist(), columns=s.columns, dtype='float32').astype({k:v for k,v in kwargs['dtypes'].items() if k in s.columns})
            output_fname = kwargs['fname'].replace('lc','dtdm')
            print(b,'band - saving:    ',output_fname)
            s.to_csv(os.path.join(kwargs['basepath'], 'dtdm_'+b, output_fname), index=False)

    else:
        output_dir = os.path.join(kwargs['basepath'], 'dtdm_')
        os.makedirs(output_dir, exist_ok=True)
        s = df[mask].groupby(df.index.name).apply(calculate_dtdm)
        s = pd.DataFrame(s.values.tolist(), index=s.index, dtype='float32')
        s.to_csv(os.path.join(kwargs['basepath'], 'dtdm_'+b, kwargs['fname'].replace('lc','dtdm')))

def calculate_dtdm(group):
    mjd_mag = group[['mjd','mag']].values
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
    
    dm2_de2 = dtdm[:,1]**2 - dmagerr**2

    uid = np.full(n*(n-1)//2,group.index[0],dtype='uint32')
    
    return pd.DataFrame(data={group.index.name:uid,'dt':dtdm[:,0],'dm':dtdm[:,1], 'de':dmagerr, 'dm2_de2':dm2_de2, 'dsid':dsid})


