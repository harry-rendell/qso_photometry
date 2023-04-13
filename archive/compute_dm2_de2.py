import pandas as pd
import numpy as np
from multiprocessing import Pool
from os import listdir, path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ID = 'uid_s'
obj = 'calibStars'
band = 'r'
data_path = '/disk1/hrb/python/data/computed/{}/dtdm/raw/r/'.format(obj)

fnames = [a for a in listdir(data_path) if (a.startswith('dtdm_raw_{}_'.format(band)))]
size   = [path.getsize(data_path+file) for file in fnames]
fnames_sorted = [name for i in [0,1,2,3] for sizename, name in sorted(zip(size, fnames))[i::4]]
#fpaths_sorted = [data_path + fname for fname in fnames]

n_cores = 4
print('number of cores:',n_cores)
print('number of files:',len(fnames_sorted))

def subtract_dm(fnames_chunk):
	for fname in fnames_chunk:
		df = pd.read_csv(data_path+fname, index_col = ID, dtype = {ID: np.uint32, 'dt': np.float32, 'dm': np.float32, 'de': np.float32, 'cat': np.uint8})
		df['dm2_de2'] = df['dm']**2 - df['de']**2
		df[['dt','dm','de','dm2_de2','cat']].to_csv(data_path+'corrected/'+fname)
		print(fname)

if __name__ == '__main__':
	p = Pool(n_cores)
	result = p.map(subtract_dm, np.array_split(fnames, n_cores))
