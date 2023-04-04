
import pandas as pd
from multiprocessing import Pool
from ..config import cfg

def reader(args):
	"""
	Reading function for multiprocessing
	"""
	i, kwargs = args
	dtypes = kwargs['dtypes'] if 'dtypes' in kwargs else None
	nrows  = kwargs['nrows']  if 'nrows'  in kwargs else None
	usecols = kwargs['usecols'] if 'usecols' in kwargs else None
	basepath = kwargs['basepath']
	return pd.read_csv(basepath+'lc_{}.csv'.format(i), 
					   comment='#',
					   dtype=dtypes,
					   nrows=nrows)

def dispatch_reader(kwargs):
	if __name__ == 'funcs.preprocessing.data_io':
		pool = Pool(cfg.USER.N_CORES)
		df_list = pool.map(reader, [(i, kwargs) for i in range(4)]) # This 4 is dictated by how many chunks we have split our data into. Currently 4.
		return pd.concat(df_list, ignore_index=True).set_index('uid')

def writer(args):
	"""
	Writing function for multiprocessing
	"""
	i, chunk, kwargs = args
	mode = kwargs['mode'] if 'mode' in kwargs else 'w'
	f = open(kwargs['basepath']+'lc_{}.csv'.format(i), mode)
	if 'comment' in kwargs:
		f.write(kwargs['comment'])
	chunk.to_csv(f)

def dispatch_writer(chunks, kwargs):
	if __name__ == 'funcs.preprocessing.data_io':
		pool = Pool(cfg.USER.N_CORES)
		pool.map(writer, [(i, chunk, kwargs) for i, chunk in enumerate(chunks)])
