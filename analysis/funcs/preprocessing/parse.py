import pandas as pd
import numpy as np

def intersection(*args):
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

def filter_data(survey_obj, bounds_generic, bounds_specific=None, bands=None, dropna=True, maxmagerr=np.inf):
	key_mag = list(bounds_generic.keys())[0]
	
	if bands==None: #the sdss raw data has a different format to ztf and ps
		print('num obs before: {:,}'.format(survey_obj[key_mag].notna().values.sum()))
		boolean = np.stack([np.array(survey_obj[key].between(bound[0], bound[1])) for key, bound in bounds_generic.items()])
		boolean = boolean.all(axis=0)
		for key in bounds_generic.keys():
			survey_obj.loc[:,key] = survey_obj[key].where(boolean.T)
		
		# Drop rows with no observations in any band
		if dropna:
			survey_obj = survey_obj.dropna(subset=[key_mag])
		print('num obs after:  {:,}'.format(survey_obj[key_mag].notna().values.sum()))

	else:
# 		print('num obj before: {:,}'.format(survey_obj[[key_mag + '_' + b for b in bands]].notna().values.sum()))
		print('num obj before: {:,}'.format(len(survey_obj.index)))

		# Create set of boolean numpy arrays which are true if the key is within the bounds.
# 		boolean_generic  = np.stack([np.array([survey_obj[key + '_' + b].between(bound[0], bound[1])) for key, bound in bounds_generic .items()])
		boolean_generic  = np.array([np.array([survey_obj[key + '_' + b].between(bound[0], bound[1]) for b in bands]) for key, bound in bounds_generic.items()])
	
		if bounds_specific is not None:
			boolean_specific = np.array([np.array(survey_obj[key].between(bound[0], bound[1])) for key, bound in bounds_specific.items()])
			key2 = ['magerr_max_' + band for band in bands]
			# Return .all() so that if either mag, meanerr or magerr_max is outside bounds, both values become NaN
			boolean = np.concatenate([boolean_generic, boolean_specific[np.newaxis,:]]).all(axis=0).T
		else:
			boolean = boolean_generic.all(axis=0).T
			key2 = []
			
		for i, band in enumerate(bands):
			key1 = [key + '_' + band for key in bounds_generic.keys()]
# 			survey_obj.loc[:,key1+key2] = survey_obj.loc[:,key1+key2].where(boolean[:,i])
			# Set the rows which do not satisfy the bands to NaN
			survey_obj.loc[~boolean[:,i], key1+key2] = np.nan
		# Drop rows with no observations in any band
		if dropna:
			survey_obj = survey_obj[survey_obj.loc[:,[key_mag + '_' + b for b in bands]].notna().any(axis=1)]
		print('num obj after:  {:,}'.format(len(survey_obj.index)))
		
	return survey_obj

def compute_colors(survey):
    colors = pd.DataFrame()
    colors['mean_gi'] = survey.df_pivot['mean_g'] - survey.df_pivot['mean_i']
    colors['mean_gr'] = survey.df_pivot['mean_g'] - survey.df_pivot['mean_r']
    colors['mean_ri'] = survey.df_pivot['mean_r'] - survey.df_pivot['mean_i']
    colors['mean_iz'] = survey.df_pivot['mean_i'] - survey.df_pivot['mean_z']
    return colors

def split_into_non_overlapping_chunks(df, n):
    """
    Split the dataframe into n roughly equally sized chunks in such a way that the index does not 
        overlap between chunks. Returns a list of DataFrames.
    """
    idx_quarters = np.percentile(df.index.values, q=np.linspace(0,100,n+1,dtype='int'), interpolation='nearest')
    uids = [(idx_quarters[i],idx_quarters[i+1]) if i == 0 else (int(idx_quarters[i]+1),idx_quarters[i+1]) for i in range(n)] # Make non-overlapping chunks
    chunks = [df.loc[uids[i][0]:uids[i][1]] for i in range(n)]
    return chunks
