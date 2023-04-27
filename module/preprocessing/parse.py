import pandas as pd
import numpy as np

def intersection(*args):
	"""
	Find intersection of objects for which we have data in surveys provided in args
	TODO:
		We don't need to pass in all data, instead just pass in the analysis class and fetch the unique uids from there.
	"""
	surveys = args
	for name, survey in zip(['sdss','ps  ','ztf '],surveys):
		print(f'{name} before intersection: {len(survey):,}')

	indicies	 = [survey.index.get_level_values(0) for survey in surveys]
	indicies_set = [set(index) for index in indicies]

	# Find the set of uid_s that exist in all three surveys
	intersection = set.intersection(*indicies_set)

	surveys = [survey[index.isin(intersection)] for survey, index in zip(surveys, indicies)]
	print('---------------------------------')
	for name, survey in zip(['sdss','ps  ','ztf '],surveys):
		print(f'{name} after  intersection: {len(survey):,}')
	print('---------------------------------')
	return surveys

def filter_data(df, bounds, dropna=True, inplace=False):
	"""
	Remove data that lies outside ranges specified in bounds.
	Note, using inplace=True is approx ~30% more memory efficient and prevents additional dataframes being stored.
	Use inplace=False when testing bounds, but then switch to inplace=True once suitable bounds have been found. 
	"""
	print('Filtering with bounds:',bounds)
	if not inplace:
		df = df.copy()
	# Create set of boolean numpy arrays which are true if the key is within the bounds.
	for key, bound in bounds.items():
		boolean = df[key].between(bound[0], bound[1])
		if inplace:
			df[key].where(boolean, np.nan, inplace=True)
		else:
			df[key] = df[key].where(boolean, np.nan, inplace=False)
	
	# Drop rows with no observations in any band
	if dropna:
		n = len(df.index)
		if inplace:
			df.dropna(axis=0, how='any', inplace=True)
		else:
			df = df.dropna(axis=0, how='any', inplace=False)
		print('num obs before:  {:,}'.format(n))
		print('num obs after:   {:,}'.format(len(df.index)))
		print('num obs removed: {:,}'.format(n - len(df.index)))
	
	return df


def compute_colors(survey):
    colors = pd.DataFrame()
    colors['mean_gi'] = survey.df_pivot['mean_g'] - survey.df_pivot['mean_i']
    colors['mean_gr'] = survey.df_pivot['mean_g'] - survey.df_pivot['mean_r']
    colors['mean_ri'] = survey.df_pivot['mean_r'] - survey.df_pivot['mean_i']
    colors['mean_iz'] = survey.df_pivot['mean_i'] - survey.df_pivot['mean_z']
    return colors

def split_into_non_overlapping_chunks(df, n_chunks, bin_size=None, return_bin_edges=False):
    """
    Split the dataframe into n roughly equally sized chunks in such a way that the index does not 
        overlap between chunks. Returns a list of DataFrames.
    bin_size may be specified if we want chunks of a specific number of uids per chunk (15,000 used for merged data)
    """
    if not df.index.is_monotonic:
    	raise Exception('Index must be sorted to split into chunks')
    if bin_size is not None:
    	idxs = np.arange(0, bin_size*(n_chunks+1), bin_size)
    else:
    	idxs = np.percentile(df.index.values, q=np.linspace(0,100,n_chunks+1,dtype='int'), interpolation='nearest')
    bin_edges = [(idxs[i],idxs[i+1]) if i == 0 else (int(idxs[i]+1),idxs[i+1]) for i in range(n_chunks)] # Make non-overlapping chunks
    chunks = [df.loc[bin_edges[i][0]:bin_edges[i][1]] for i in range(n_chunks)]
    
    if return_bin_edges:
    	return bin_edges, chunks
    else:
    	return chunks
