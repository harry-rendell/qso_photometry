import pandas as pd
import numpy as np
from os import listdir
from multiprocessing import Pool
data_path = '/disk1/hrb/python/analysis/qsos/computed/dtdm/raw/'

#fnames = sorted([a for a in listdir(data_path) if (len(a)==28)])[4:]

fnames = ['dtdm_raw_r_000000_010000.csv',
'dtdm_raw_r_130000_140000.csv',
'dtdm_raw_r_190000_200000.csv',
'dtdm_raw_r_260000_270000.csv',
'dtdm_raw_r_320000_330000.csv',
'dtdm_raw_r_390000_400000.csv']

def fix_uids(fnames):
	for fname in fnames:
		fpath = data_path+fname
		df = pd.read_csv(fpath, usecols = ['uid'] , dtype={'uid':'uint32'}, squeeze=True)
		lower = int(fname[-17:-11])
		upper = int(fname[-10:-4])
		print('{} to {}'.format(lower, upper))
		uids = df.values
		idx = np.argmax(uids==0)
		multiplier = upper // 2**16
		print('multiplier: ', multiplier)
		if idx == 0:
			uids += 2**16*multiplier
		else:
			uids[:idx] += 2**16*(multiplier-1)
			uids[idx:] += 2**16*(multiplier)
		df.to_csv(fpath[:-4]+'_uids.csv', index=False)

n_cores = 4
		
if __name__ == '__main__':
    p = Pool(n_cores)
    p.map(fix_uids, np.array_split(fnames, 4))
