import pandas as pd
import numpy as np
from multiprocessing import Pool
from funcs.analysis import *

wdir = '/disk1/hrb/python/'
obj  = 'qsos'
ID   = 'uid'

# obj  = 'calibStars'
# ID   = 'uid_s'
# time_key = 'mjd'

# Use this to read in all data
def reader(n_subarray):
    return pd.read_csv(wdir+'data/merged/{}/lc_{}_{}.csv'.format(obj, band, n_subarray), nrows=None, index_col = ID, dtype = {'catalogue': np.uint8, 'mag_ps': np.float32, 'magerr': np.float32, 'mjd': np.float64, ID: np.uint32})

def loc_uids(self, lower, upper, width=4, step=1): 
    # width is number of processes per core
    uids = self.df.index.unique()
    bounds = calc_bounds(lower, upper, width, step)
    uid_dict = [{'{:06d}_{:06d}'.format(lower,upper):uids[(lower<uids)&(uids<=upper)] for lower, upper in zip(bound,bound[1:])} for bound in bounds]
    return uid_dict

def calc_bounds(lower, upper, width, step):
	return [np.arange(lower*1e4,(upper+1)*1e4,step*1e4, dtype='uint32')[i:(i+width+1)] for i in range(0, int((upper-lower)/step), width)]

def savedtdm(uid_dict):
    for key, uids in uid_dict.items():
        print('computing: {}'.format(key))
        df_batch = dr.save_errors(uids)
        print('saving:    {}'.format(key))
        df_batch.to_csv(wdir+'analysis/{}/computed/dtdm/raw/errors_raw_{}_{}.csv'.format(obj,dr.band,key),index=False)

lower = 101
upper = 117
width = 4
step  = 1
n_cores = int((upper-lower)/(step*width))

print('num cores: ',n_cores)

for i, chunk in enumerate(calc_bounds(lower, upper, width, step)):
	print('core {}: {}'.format(i+1,chunk))

keypress = input('press c to continue')
if keypress == 'c':
	pass
else:
	exit()

band = 'r'
dr = analysis(band, ID)
dr.read_in(reader, redshift=False)
# dr.residual({1: 0.049, 3: -0.0099}) # leaving this out for now as we can compute it later

if __name__ == '__main__':
    p = Pool(n_cores)
    p.map(savedtdm, loc_uids(dr, lower, upper, width, step))

