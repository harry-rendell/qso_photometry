import pandas as pd
from multiprocessing import Pool
import os
from ..config import cfg

def reader(args):
	"""
	Reading function for multiprocessing
	"""
	i, kwargs = args
	dtypes = kwargs['dtypes'] if 'dtypes' in kwargs else None
	nrows  = kwargs['nrows']  if 'nrows'  in kwargs else None
	usecols = kwargs['usecols'] if 'usecols' in kwargs else None
	skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else None
	basepath = kwargs['basepath']

	# Open the file and skip any comments. Leave the file object pointed to the header.
	# Pass in the header in case we decide to skip rows.
	with open(basepath+'lc_{}.csv'.format(i)) as file:
		ln = 0
		for line in file:
			ln += 1
			if not line.strip().startswith("#"):
				names = line.replace('\n','').split(',')
				break
		return pd.read_csv(file,
						   usecols=usecols,
						   dtype=dtypes,
						   nrows=nrows,
						   names=names,
						   skiprows=skiprows)
# @profile
def dispatch_reader(kwargs, multiproc=True, i=0):
	"""
	Dispatching function for reader
	"""
	if multiproc:
		if __name__ == 'module.preprocessing.data_io':
			pool = Pool(cfg.USER.N_CORES)
			df = pool.map(reader, [(j, kwargs) for j in range(4)]) # This 4 is dictated by how many chunks we have split our data into. Currently 4.
			df = pd.concat(df, ignore_index=True) # overwrite immediately for prevent holding unnecessary dataframes in memory
			if 'ID' in kwargs:
				return df.set_index(kwargs['ID'])
			else:
				return df
	else:
		df = reader((i, kwargs))
		if 'ID' in kwargs:
			return df.set_index(kwargs['ID'])
		else:
			return df

def writer(args):
	"""
	Writing function for multiprocessing
	"""
	i, chunk, kwargs = args
	mode = kwargs['mode'] if 'mode' in kwargs else 'w'
	if 'basepath' in kwargs:
		basepath = kwargs['basepath']
	else:
		raise Exception('user must provide path for saving output')

	# if folder does not exist, create it
	if not os.path.exists(basepath):
		os.makedirs(basepath)

	f = open(basepath+'lc_{}.csv'.format(i), mode)
	if 'comment' in kwargs:
		f.write(kwargs['comment'])
	chunk.to_csv(f)

def dispatch_writer(chunks, kwargs):
	"""
	Dispatching function for writer
	"""
	if __name__ == 'module.preprocessing.data_io':
		pool = Pool(cfg.USER.N_CORES)
		pool.map(writer, [(i, chunk, kwargs) for i, chunk in enumerate(chunks)])

def dispatch_function(function, chunks, kwargs={}):
	"""
	Dispatch generic function on a set of chunks
	"""
	dtypes = kwargs['dtypes'] if 'dtypes' in kwargs else None
	if __name__ == 'module.preprocessing.data_io':
		pool = Pool(cfg.USER.N_CORES)
		df = pool.map(function, [chunk for chunk in chunks]) # This 4 is dictated by how many chunks we have split our data into. Currently 4.
		df = pd.concat(df, ignore_index=False) # overwrite immediately for prevent holding unnecessary dataframes in memory
		dtypes = {k:v for k,v in dtypes.items() if k in df.columns}
		if 'ID' in kwargs:
			return df.set_index(kwargs['ID']).astype(dtypes)
		else:
			return df.astype(dtypes)
