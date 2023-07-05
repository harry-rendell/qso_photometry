import numpy as np
import pandas as pd
import os
from ..config import cfg
from .parse import split_into_non_overlapping_chunks
from scipy.stats import skew, skewtest, kurtosis, kurtosistest

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
        s.dropna(axis=0, how='any').to_csv(output_fpath, index=False, header=False, mode='a')
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

def calculate_stats_looped(df, kwargs):
    """
    Loop over dtdm files and calculate stats of each file. Append to dictionary.
    Make sure to include name of desired quantites in names.
    
    Parameters
    ----------
    n_chunks : int
        how many files to read in of files to read in.
        maximum value: stars = 200/4 = 50 (though 46 seems to be better, 50 runs of out memory), qsos = 52/4 = 13

    log_or_lin : str

    save : bool

    Returns
    -------
    results : dict of nd_arrays, shape (n_chunk, n_points)
    """
    
    inner = kwargs['inner']
    features = kwargs['features']
    n_points = kwargs['n_points'] # number of points to plot
    mjd_edges = kwargs['mjd_edges']

    # names = ['n','SF 1', 'SF 2', 'SF 3', 'SF 4', 'SF weighted', 'SF corrected', 'SF corrected weighted', 'SF corrected weighted fixed', 'SF corrected weighted fixed 2', 'mean', 'mean weighted']
    results = {feature:np.zeros(shape=(n_points, 2)) for feature in features}
    results['n'] = np.zeros(shape=(n_points), dtype='uint64')

    if inner:
        df = df[np.sqrt(df['dsid'])%1==0]

    df = df.dropna()

    for j, edges in enumerate(zip(mjd_edges[:-1], mjd_edges[1:])):
        mjd_lower, mjd_upper = edges
        boolean = ((mjd_lower < df['dt']) & (df['dt']<mjd_upper)).values# & (df['dm2_de2']>0) # include last condition to remove negative SF values
        # print('number of points in {:.1f} < ∆t < {:.1f}: {}'.format(mjd_lower, mjd_upper, boolean.sum()))
        subset = df[boolean]
        # subset = subset[subset['dm2_de2']<1]
        # subset.loc[(subset['dm2_de2']<0).values,'dm2_de2'] = 0 # Include for setting negative SF values to zero. Need .values for mask to prevent pandas warning
        n = len(subset)
        results['n'][j] = n
        
        if n>0:
            # results['mean'][i,j, (0,1)] = subset['dm'].mean(), subset['dm'].std()
            # results['SF'][i,j,(0,1)] = (subset['dm']**2).mean(), (subset['de']**2).sum()/n
            # results['SF corrected'][i,j,(0,1)] = subset['dm2_de2'].mean(), subset['dm2_de2'].var()
            # results['SF a'][i,j,(0,1)] = (subset['dm']**2).mean(), 1/weights.sum()
            # results['SF b'][i,j,(0,1)] = (subset['dm']**2).mean(), (subset['dm']**2).var()

            weights = subset['de']**-2
            results['mean weighted a'][j,(0,1)] = np.average(subset['dm'], weights = weights), 1/weights.sum()
            results['mean weighted b'][j,(0,1)] = np.average(subset['dm'], weights = weights), subset['dm'].var() 
            if n>8:
                results['skewness'][j,(0,1)] = skew(subset['dm']), skewtest(subset['dm'])[1]
            if n>20:
                results['kurtosis'][j,(0,1)] = kurtosis(subset['dm']), kurtosistest(subset['dm'])[1]
            
            weights = 0.5*subset['de']**-4

            dm2_de2 = subset['dm']**2 - subset['de']**2
            SF = np.average(dm2_de2, weights = weights)

            if SF < 0:
                SF = 0
            results['SF cwf a'][j, 0] = SF
            results['SF cwf a'][j, 1] = 1/weights.sum()

            results['SF cwf b'][j, 0] = SF
            results['SF cwf b'][j, 1] = dm2_de2.var() #we should square root this, but then it's too large
            
            mask_p = (subset['dm']>0).values
            mask_n = (subset['dm']<0).values
            
            try:
                if mask_p.sum()>0:
                    SF_p = np.average(dm2_de2[mask_p], weights = weights[mask_p])
                    if SF_p < 0:
                        SF_p = 0
                    results['SF cwf p'][j,0] = SF_p
                    results['SF cwf p'][j,1] = 1/weights[mask_p].sum()
                
                if mask_n.sum()>0:
                    SF_n = np.average(dm2_de2[mask_n], weights = weights[mask_n])
                    if SF_n < 0:
                        SF_n = 0
                    results['SF cwf n'][j,0] = SF_n
                    results['SF cwf n'][j,1] = 1/weights[mask_n].sum()
            except:
                # pass
                print('weights cannot be normalized')
            
        # else:
            # print('number of points in bin:',n)
    
    return results

def calculate_stats_looped_key(df, kwargs):
    """
    Loop over dtdm files and calculate stats of each file. Append to dictionary.

    Parameters
    ----------
    n_chunks : int
        how many files to read in of files to read in.
        maximum value: stars = 200/4 = 50, qsos = 52/4 = 13

    log_or_lin : str

    save : bool

    Returns
    -------
    results : dict of nd_arrays, shape (n_chunk, n_points)
    """

    features = kwargs['features']
    n_points = kwargs['n_points'] # number of points to plot
    mjd_edges = kwargs['mjd_edges']
    groups = kwargs['groups']
    n_groups = len(groups)

    results = {feature:np.zeros(shape=(n_points, n_groups, 2)) for feature in features}
    results['n'] = np.zeros(shape=(n_points, n_groups), dtype='uint64')

    df = df.dropna()

    for group_idx in range(n_groups):
        subgroup = df[df.index.isin(groups[group_idx])]
        # print('subgroup: {}'.format(group_idx))
        # print('\tmax ∆t: {:.2f}'.format(subgroup['dt'].max()))
        
        for j, edges in enumerate(zip(mjd_edges[group_idx][:-1], mjd_edges[group_idx][1:])):
            mjd_lower, mjd_upper = edges
            boolean = ((mjd_lower < subgroup['dt']) & (subgroup['dt'] < mjd_upper)).values# & (subgroup['dm2_de2']>0) # include last condition to remove negative SF values
            subset = subgroup[boolean]
            # subset.loc[(subset['dm2_de2']<0).values,'dm2_de2'] = 0 # Include for setting negative SF values to zero. Need .values for mask to prevent pandas warning
            n = len(subset)
            results['n'][j, group_idx] = n
            # print('\t\tnumber of points in {:.1f} < ∆t < {:.1f}: {}'.format(mjd_lower, mjd_upper, boolean.sum()))

            if n>0:
                # results['mean'][i,j, (0,1)] = subset['dm'].mean(), subset['dm'].std()
                # results['SF'][i,j,(0,1)] = (subset['dm']**2).mean(), (subset['de']**2).sum()/n
                # results['SF corrected'][i,j,(0,1)] = subset['dm2_de2'].mean(), subset['dm2_de2'].var()
                # results['SF a'][i,j,(0,1)] = (subset['dm']**2).mean(), 1/weights.sum()
                # results['SF b'][i,j,(0,1)] = (subset['dm']**2).mean(), (subset['dm']**2).var()

                weights = subset['de']**-2
                results['mean weighted a'][j,group_idx,(0,1)] = np.average(subset['dm'], weights = weights), 1/weights.sum()
                results['mean weighted b'][j,group_idx,(0,1)] = np.average(subset['dm'], weights = weights), subset['dm'].var() 
                if n>8:
                    results['skewness'][j,group_idx,(0,1)] = skew(subset['dm']), skewtest(subset['dm'])[1]
                if n>20:
                    results['kurtosis'][j,group_idx,(0,1)] = kurtosis(subset['dm']), kurtosistest(subset['dm'])[1]
                
                weights = 0.5*subset['de']**-4

                dm2_de2 = subset['dm']**2 - subset['de']**2
                SF = np.average(dm2_de2, weights = weights)

                if SF < 0:
                    SF = 0
                results['SF cwf a'][j,group_idx,(0,1)] = SF, 1/weights.sum()
                results['SF cwf b'][j,group_idx,(0,1)] = SF, dm2_de2.var() #we should square root this, but then it's too large
                
                mask_p = (subset['dm']>0).values
                mask_n = (subset['dm']<0).values
                
                try:
                    if mask_p.sum()>0:
                        SF_p = np.average(dm2_de2[mask_p], weights = weights[mask_p])
                        if SF_p < 0:
                            SF_p = 0
                        results['SF cwf p'][j,group_idx,(0,1)] = SF_p, 1/weights[mask_p].sum()
                    
                    if mask_n.sum()>0:
                        SF_n = np.average(dm2_de2[mask_n], weights = weights[mask_n])
                        if SF_n < 0:
                            SF_n = 0
                        results['SF cwf n'][j,group_idx,(0,1)] = SF_n, 1/weights[mask_n].sum()
                except:
                    # pass
                    print('weights cannot be normalized')
                
            # else:
                # print('number of points in bin:',n)

    return results