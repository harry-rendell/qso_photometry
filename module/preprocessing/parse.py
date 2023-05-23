import pandas as pd
import numpy as np

def intersection(*args):
    """
    Find intersection of objects for which we have data in surveys provided in args
    TODO:
        We don't need to pass in all data, instead just pass in the analysis class and fetch the unique uids from there.
    """
    surveys = args
    for name, survey in enumerate(surveys):
        print(f'{name} before intersection: {len(survey):,}')

    indicies     = [survey.index for survey in surveys]
    indicies_set = [set(index) for index in indicies]

    # Find the set of uid_s that exist in all three surveys
    intersection = set.intersection(*indicies_set)

    surveys = [survey[index.isin(intersection)] for survey, index in zip(surveys, indicies)]
    print('---------------------------------')
    for name, survey in enumerate(surveys):
        print(f'{name} after  intersection: {len(survey):,}')
    print('---------------------------------')
    return surveys

def filter_data(df, bounds={}, dropna=True, valid_uids=None):
    """
    Remove data that lies outside ranges specified in bounds.
    Note, using inplace=True is approx ~30% more memory efficient and prevents additional dataframes being stored.
    Use inplace=False when testing bounds, but then switch to inplace=True once suitable bounds have been found. 
    Note, the bounds are INCLUSIVE.
    """
    # Restrict our dataframe rows with indices contained in valid_uids, if provided.
    df = df.copy()
    n = len(df.index)
    if valid_uids is not None:
        mask = df.index.isin(valid_uids.index)
        df = df[mask]
        print('No. rows removed that are not in valid_uids: {:,}\n'.format( (~mask).sum() ))
    # Create set of boolean numpy arrays which are true if the key is within the bounds.
    for key, bound in bounds.items():
        boolean = df[key].between(bound[0], bound[1])
        print('Enforcing {:.2f} <= {} <= {:.2f}'.format(bound[0],key,bound[1]).ljust(50,' ') + 'No. points outside bounds: {:,}'.format((~boolean).sum()))
        df[key] = df[key].where(boolean, np.nan, inplace=False)
    
    # Drop rows with no observations in any band
    if dropna:
        df = df.dropna(axis=0, how='any', inplace=False)
        print('-'*85)
        print('No. rows before:  {:,}'.format(n))
        print('No. rows after:   {:,}'.format(len(df.index)))
        print('No. rows removed: {:,} ({:.2f}%)'.format(n - len(df.index), (1-len(df.index)/n)*100))
    
    return df

def compute_colors(survey):
    colors = pd.DataFrame()
    colors['mean_gi'] = survey.df_pivot['mean_g'] - survey.df_pivot['mean_i']
    colors['mean_gr'] = survey.df_pivot['mean_g'] - survey.df_pivot['mean_r']
    colors['mean_ri'] = survey.df_pivot['mean_r'] - survey.df_pivot['mean_i']
    colors['mean_iz'] = survey.df_pivot['mean_i'] - survey.df_pivot['mean_z']
    return colors

def split_into_non_overlapping_chunks(df, n_chunks, bin_size=None, return_bin_edges=False, n_cores=1):
    """
    Split the dataframe into n roughly equally sized chunks in such a way that the index does not 
        overlap between chunks. Returns a list of DataFrames.
    bin_size may be specified if we want chunks of a specific number of uids per chunk (15,000 used for merged data)

    Use cases:
        1. use to split up a dataframe into equal sized chunks. May set return_bin_edges to true to return bin edges
            df=DataFrame
            bin_size=None
            return_bin_edges=True or False
        2. use to generate uid ranges to create files with headers in merge_survey_data.py. Return bin edges only. df=None.
            df=None
            bin_size=list
            return_bin_edges=True
        3. use to generate chunks for saving for merge_survey_data. Return bin_edges and chunks
            df=DataFrame
            bin_size=list
            return_bin_edges=True

    """
    if bin_size is not None:
        idxs = np.arange(0, bin_size*(n_chunks+1), bin_size)
    else:
        idxs = np.percentile(df.index.values, q=np.linspace(0,100,n_chunks+1,dtype='int'), interpolation='nearest')
    
    bin_edges = [(idxs[i],idxs[i+1]) if i == 0 else (int(idxs[i]+1),idxs[i+1]) for i in range(n_chunks)] # Make non-overlapping chunks
    uid_ranges = [f'{lower:06d}_{upper:06d}' for lower, upper in bin_edges]

    # The line below can't really be parallelised as we would have to declare df as a global variable
    #   to prevent it from being copied for every child process (which would exceed memory.)
    if df is not None:
        chunks = [df.loc[bin_edges[i][0]:bin_edges[i][1]] for i in range(n_chunks)]
        if not df.index.is_monotonic_increasing:
            raise Exception('Index must be sorted to split into chunks')
    else:
        chunks = None

    if return_bin_edges:
        return uid_ranges, chunks
    else:
        return chunks
