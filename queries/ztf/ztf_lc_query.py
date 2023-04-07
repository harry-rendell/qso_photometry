import pandas as pd
import numpy as np
import os

# Note, we could improve this by writing to csv in mode='a' which appends data. ie separate data by filtercode and save to respective csv's.

def split_save(oid_batch,n,i):
	oid_batch_half = np.array_split(oid_batch,2)
	# Note: clrcounc provides uncertainty in clrcoeff. Could be used to propogate errors in transformation but this is not currently implemented.
	# On further inspection, the max and median of clrcounc are on the order 1e-2 and 1e-5 respectively across our data, which is too small to worry about given mag uncertainties
	df = pd.DataFrame(columns=['oid','mjd','mag','magerr','filtercode','clrcoeff','limitmag'])
	for j, oids in enumerate(oid_batch_half):
		url = base_url.format(''.join(['&ID='+str(oid) for oid in oids])[1:])
		try:
			df_sub = pd.read_csv(url, usecols = ['oid','mjd','mag','magerr','filtercode','clrcoeff','limitmag'])
			df = df.append(df_sub)
		except:
			print('error downloading batch {}, {}/{}'.format(n, i, n_request))
			unable_to_download += 'lc_{:01d}_{:04d}'.format(n,i)
	df.to_csv(output_folder+'raw_lcs/lc_{:01d}_{:04d}.csv'.format(n,i), index=False)
	print('batch {}, {}/{}: split save complete'.format(n, i, n_request))

def obtain_ztf_lightcurves(n):
	oids = np.array_split(pd.read_csv(ztf_oids_fname, squeeze=True, index_col='uid', dtype=np.uint64).values, n_cores)[n]
	for i, oid_batch in enumerate(np.array_split(oids, n_request)):
		if not os.path.isfile(output_folder+'raw_lcs/lc_{:01d}_{:04d}.csv'.format(n,i)):
			url = base_url.format(''.join(['&ID='+str(oid) for oid in oid_batch])[1:])
			try:
				print('batch {}: requesting {}/{}'.format(n, i, n_request))
				df = pd.read_csv(url, usecols = ['oid', 'mjd', 'mag','magerr','filtercode','magzp','clrcoeff','clrcounc','airmass'])
				df.to_csv(output_folder+'raw_lcs/lc_{:01d}_{:04d}.csv'.format(n,i), index=False)
				print('batch {}: saving     {}/{}'.format(n, i, n_request))
			except Exception as e:
				print('response:',e)
				print('batch {}, {}/{}: URL too long, splitting and saving'.format(n, i, n_request))
				try:
					split_save(oid_batch,n,i)
				except:
					print('Cannot save',n,i)

			print('batch {}: finished'.format(n))
			print(unable_to_download)
		else:
			print('lc_{:01d}_{:04d}.csv already exists'.format(n,i))

unable_to_download = []
n_request = 1500 #total number of requests per core. We require n s.t. len(ztf_oids.txt)/(n_request * n_cores) < 350. 350 is approx the max no. oids we can ask for in http request.
n_cores   = 4
obj = "qsos"
ztf_oids_fname  = "/disk1/hrb/python/queries/ztf/{}/ztf_oids.csv".format(obj)
output_folder = "/disk1/hrb/python/data/surveys/ztf/{}/dr6/".format(obj)
base_url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?{}&FORMAT=csv&BAD_CATFLAGS_MASK=32768"

from multiprocessing import Pool
if __name__ == '__main__':
	p = Pool(4)
	p.map(obtain_ztf_lightcurves,[0,1,2,3])
