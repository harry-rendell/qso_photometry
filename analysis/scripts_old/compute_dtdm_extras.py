import pandas as pd
import numpy as np
from multiprocessing import Pool
from funcs.analysis.analysis import analysis
from time import time

wdir = '/disk1/hrb/python/'
band = 'r'

obj  = 'qsos'
ID   = 'uid'
time_key = 'mjd_rf'

# obj  = 'calibStars'
# ID   = 'uid_s'
# time_key = 'mjd'

# Use this to read in all data
# Note we cannot have ID as index_col as we get an error
def reader(n_subarray):
	return pd.read_csv(wdir+'data/merged/{}/{}_band/lc_{}.csv'.format(obj, band, n_subarray), comment='#', nrows=None, dtype = {'catalogue': np.uint8, 'mag_ps': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})

def loc_uids(self, lower, upper, width, step, custom=None):
	# width is number of processes per core
	uids = self.df.index.unique()
	if custom == None:
		bounds = calc_bounds(lower, upper, width, step)
	else:
		bounds = custom
	uid_dict = [{'{:06d}_{:06d}'.format(lower,upper):uids[(lower<uids)&(uids<=upper)] for lower, upper in zip(bound,bound[1:])} for bound in bounds]
	return uid_dict

def calc_bounds(lower, upper, width, step):
	return [np.arange(lower*1e4,(upper+1)*1e4,step*1e4, dtype='uint32')[i:(i+width+1)] for i in range(0, int((upper-lower)/step), width)]

def savedtdm(uid_dict):
	for key, uids in uid_dict.items():
		print('computing: {}'.format(key))
		df_batch = dr.save_dtdm_rf_extras(df_ssa, uids, time_key=time_key)
		print('saving:	{}'.format(key))
		df_batch.to_csv(wdir+'data/computed/{}/dtdm/raw/{}/dtdm_raw_{}_{}_ssa.csv'.format(obj,band,band,key),index=False)

lower = 0
upper = 0.0004
width = 1
step  = 0.0001
n_cores = int((upper-lower)/(step*width))
chunks = calc_bounds(lower, upper, width, step)

# Read uids below
# custom = np.loadtxt(wdir+'data/computed/{}/dtdm/raw/{}/missing_uids.csv'.format(obj,band), dtype='int', delimiter=',')
# chunks = custom
# n_cores = len(custom)

print('num cores: ',n_cores)

for i, chunk in enumerate(chunks):
	print('core {}: {}'.format(i+1,chunk))


dr = analysis(ID, obj)
dr.read_in(reader, redshift=True, multi_proc=True)
df_ssa = pd.read_csv('/disk1/hrb/python/data/surveys/supercosmos/ssa_r.csv', index_col='uid')
df_ssa = df_ssa[df_ssa.index.isin(dr.df.index.unique())]
df_ssa = df_ssa.join(dr.redshifts, how='inner')
df_ssa['mjd_rf'] = df_ssa['mjd']/(1+df_ssa['redshift'])
# dr.residual({1: 0.049, 3: -0.0099}) # leaving this out for now as we can compute it later

start = time()

if __name__ == '__main__':
	p = Pool(n_cores)
	p.map(savedtdm, loc_uids(dr, lower, upper, width, step, custom=None))

end = time()

print('time taken: {:.2f} minutes'.format((end-start)/60.0))
exit()
