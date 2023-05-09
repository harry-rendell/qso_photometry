import pandas as pd
from multiprocessing import Pool
import os
from .import parse
from ..config import cfg
from astropy.table import Table
from astropy.io import ascii

def to_ipac(df, save_as, columns):
	t = Table.from_pandas(df[columns], index=True)
	ascii.write(t, save_as, format='ipac', overwrite=True)

def reader(fname, kwargs):
	"""
	Reading function for multiprocessing
	"""
	dtypes = kwargs['dtypes'] if 'dtypes' in kwargs else None
	nrows  = kwargs['nrows']  if 'nrows'  in kwargs else None
	usecols = kwargs['usecols'] if 'usecols' in kwargs else None
	skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else None
	delimiter = kwargs['delimiter'] if 'delimiter' in kwargs else ','
	basepath = kwargs['basepath']
	
	if 'ID' in kwargs:
		ID = kwargs['ID']
	else:
		raise Exception('ID must be provided in keyword args')

	# Open the file and skip any comments. Leave the file object pointed to the header.
	# Pass in the header in case we decide to skip rows.
	with open(os.path.join(basepath,fname)) as file:
		ln = 0
		for line in file:
			ln += 1
			if not line.strip().startswith(('#','|','\\')):
				names = line.replace('\n','').split(',')
				break
		return pd.read_csv(file,
						   usecols=usecols,
						   dtype=dtypes,
						   nrows=nrows,
						   names=names,
						   skiprows=skiprows,
						   delimiter=delimiter).set_index(ID)

def dispatch_reader(kwargs, multiproc=True, i=0, max_processes=64):
	"""
	Dispatching function for reader
	"""
	fnames = sorted([f for f in os.listdir(kwargs['basepath']) if (f.startswith('lc_') and f.endswith('.csv'))])
	n_files = len(fnames)
	if multiproc:
		if __name__ == 'module.preprocessing.data_io':
			# Make as many tasks as there are files, unless we have set max_processes
			n_tasks = min(n_files, max_processes)
			with Pool(n_tasks) as pool:
				df = pool.starmap(reader, [(fname, kwargs) for fname in fnames])
			# sorting is required as we cannot guarantee that starmap returns dataframes in the order we expect.
			return pd.concat(df, sort=True)
	else: 
		return reader(fnames[i], kwargs)

def writer(i, chunk, kwargs):
	"""
	Writing function for multiprocessing
	"""
	mode = kwargs['mode'] if 'mode' in kwargs else 'w'
	if 'basepath' in kwargs:
		basepath = kwargs['basepath']
	else:
		raise Exception('user must provide path for saving output')

	# if folder does not exist, create it
	os.makedirs(basepath, exist_ok=True)

	f = open(os.path.join(basepath,'lc_{}.csv'.format(i)), mode)
	if 'comment' in kwargs:
		newline = '' if kwargs['comment'].endswith('\n') else '\n'
		f.write(kwargs['comment']+newline)
	chunk.to_csv(f)

def dispatch_writer(chunks, kwargs, max_processes=64):
	"""
	Dispatching function for writer
	TODO: Is it bad that we sometimes spawn more processes than needed?
	"""
	if __name__ == 'module.preprocessing.data_io':
		n_tasks = min(len(chunks), max_processes)
		with Pool(n_tasks) as pool:
			pool.starmap(writer, [(i, chunk, kwargs) for i, chunk in enumerate(chunks)])

def process_input(function, df_or_fname, kwargs):
	"""
	This function handles inputs
	"""
	if isinstance(df_or_fname, str):
		# In this case, are provided a filename. Read it with reader() then pass to function()
		# Add the fname to the dictionary so function() can access it if necessary.
		kwargs['fname'] = df_or_fname
		return function(reader(df_or_fname, kwargs), kwargs)
	else:
		# In this case, df_or_fname is a DataFrame chunk.
		return function(df_or_fname, kwargs)

def dispatch_function(function, chunks=None, max_processes=64, **kwargs):
	"""
	Parameters
	----------
	function : function for which we use to dispatch on files or DataFrame object via multiprocessing
	chunks : DataFrame, list of DataFrames or None
		if a DataFrame is provided, it will be split automatically into number of max_processes
		if a list of DataFrames are provided, they are unpacked and passed to 'function'
		if left as None, then basepath may be provided in kwargs. files that match basepath/lc_{.*}.csv are read and passed
			to 'function'

	Note, we may provide kwargs as usual or pass a dictionary as **{...}

	Returns
	-------
	returns the output of 'function'
	"""
	if chunks is None:
		if 'basepath' in kwargs:
			chunks = sorted([f for f in os.listdir(kwargs['basepath']) if (f.startswith('lc_') and f.endswith('.csv'))])
		else:
			raise Exception('Either one of chunks or basepath (in kwargs) must be provided')
	elif isinstance(chunks, pd.DataFrame):
		chunks = parse.split_into_non_overlapping_chunks(chunks, max_processes)
	elif not isinstance(chunks, list):
		raise Exception('Invalid input')

	if __name__ == 'module.preprocessing.data_io':
		# Make as many processes as there are files/chunks.
		# There may be more elements in chunks than there are processes,
		#   in this case the tasks will do them in turn.
		n_tasks = min(len(chunks), max_processes)
		with Pool(n_tasks) as pool:
			output = pool.starmap(process_input, [(function, chunk, kwargs) for chunk in chunks])
		
		if not all(o is None for o in output):
			# If it is better to save chunks rather than concatenate result into one DataFrame
			#    (eg in case of calculate dtdm) then only run this block if a result is returned.
			output = pd.concat(output, sort=True) # overwrite immediately for prevent holding unnecessary dataframes in memory

			if 'dtypes' in kwargs:
				dtypes = {k:v for k,v in kwargs['dtypes'].items() if k in output.columns}
			else:
				dtypes={}

			return output.astype(dtypes)
