import pandas as pd
import numpy as np
import os

def split_save(oid_batch,n,i):
	oid_batch_half = np.array_split(oid_batch,2)
	df = pd.DataFrame(columns=['oid', 'mjd', 'mag','magerr','filtercode','magzp','clrcoeff','clrcounc'])
	for j, oids in enumerate(oid_batch_half):
		url = 'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?{}&FORMAT=csv&BAD_CATFLAGS_MASK=32768'.format(''.join(['&ID='+str(oid) for oid in oids])[1:])
		try:
			df_sub = pd.read_csv(url, usecols = ['oid', 'mjd', 'mag','magerr','filtercode','magzp','clrcoeff','clrcounc'])
			df = df.append(df_sub)
		except:
			print('error downloading batch {}, {}/{}'.format(n, i, n_request))
			unable_to_download += 'lc_{:01d}_{:03d}'.format(n,i)
	df.to_csv('lcs/lc_{:01d}_{:03d}.csv'.format(n,i), index=False)
	print('batch {}, {}/{}: split save complete'.format(n, i, n_request))

def obtain_ztf_lightcurves(n):
	oids = np.array_split(np.loadtxt('/disk1/hrb/python/data/surveys/ztf/calibStars/calibStars_oids.txt',dtype=np.uint64), n_cores)[n]
	for i, oid_batch in enumerate(np.array_split(oids, n_request)):
		if not os.path.isfile('lcs/lc_{:01d}_{:03d}.csv'.format(n,i)):
			url = 'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?{}&FORMAT=csv&BAD_CATFLAGS_MASK=32768'.format(''.join(['&ID='+str(oid) for oid in oid_batch])[1:])
			try:
				print('batch {}: requesting {}/{}'.format(n, i, n_request))
				df = pd.read_csv(url, usecols = ['oid', 'mjd', 'mag','magerr','filtercode','magzp','clrcoeff','clrcounc','airmass'])
				df.to_csv('lcs/lc_{:01d}_{:03d}.csv'.format(n,i), index=False)
				print('batch {}: saving     {}/{}'.format(n, i, n_request))
			except:
				print('batch {}, {}/{}: URL too long, splitting and saving'.format(n, i, n_request))
				try:
					split_save(oid_batch,n,i)
				except:
					print('Cannot save',n,i)

			print('batch {}: finished'.format(n))
			print(unable_to_download)
		else:
			print('lc_{:01d}_{:03d}.csv already exists'.format(n,i))

unable_to_download = []
n_request = 900 #number of requests per core. We require n s.t. len(calibStars_oids.txt)/(n_request * n_cores) < 350. 350 is approx the max no. oids we can ask for in http request.
n_cores   = 4
from multiprocessing import Pool
if __name__ == '__main__':
	p = Pool(4)
	p.map(obtain_ztf_lightcurves,[0,1,2,3])
