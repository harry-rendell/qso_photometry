import pandas as pd
import numpy as np
import argparse
import os
import sys
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import colour_transform
from urllib.error import HTTPError

def request_and_save_photometry(oid_batch, suffix):
	base_url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?{}&FORMAT=csv&BAD_CATFLAGS_MASK=32768"
	url = base_url.format(''.join(['&ID='+str(oid) for oid in oid_batch['oid']])[1:])
	df = pd.read_csv(url, usecols = ['oid','mjd','mag','magerr','filtercode','clrcoeff','limitmag'], dtype=cfg.COLLECTION.ZTF.dtypes)
	df = df.merge(oid_batch, on='oid', how='left', sort=False).sort_values(['uid','mjd'])
	for band in 'gri':
		transformed_data = color_transform.transform_ztf_to_ps(df[df['filtercode']=='z'+band], OBJ, band)
		transformed_data[SAVE_COLS].to_csv(output_folder+'{}_band/lc_{:01d}.csv'.format(band, suffix % 4), index=False, header=False, mode='a')

def split_save(oid_batch, suffix, n_chunks):
	# split into n chunks and save individually
	for oid_batch_halved in np.array_split(oid_batch, n_chunks):
		request_and_save_photometry(oid_batch_halved, suffix)

def obtain_ztf_lightcurves(suffix):
	oids = np.array_split(pd.read_csv(ztf_oids_fname, usecols=['uid','oid'], dtype=np.uint64), n_workers)[suffix]
	for i, oid_batch in enumerate(np.array_split(oids, n_requests)):
		for n_chunks in [1,2,4,8]:
			try:
				print('batch {}, {}/{} requesting     in {} chunk(s)'.format(suffix, i+1, n_requests, n_chunks), flush=True)
				split_save(oid_batch, suffix, n_chunks)
				print('batch {}, {}/{} success saving in {} chunk(s)'.format(suffix, i+1, n_requests, n_chunks), flush=True)
				success = True
				break
			except HTTPError as e:
				print('batch {}, {}/{} HTTPError      in {} chunk(s)'.format(suffix, i+1, n_requests, n_chunks), flush=True)
				success = False
		if not success:
			print('ERROR: batch {}, {}/{} - Unable to save despite splitting into 8 chunks'.format(suffix, i+1, n_requests), flush=True)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
	parser.add_argument("--n_workers", type=int, required=True, help="Number of workers to use")
	args = parser.parse_args()
	print('args:',args)

	OBJ = arg.object
	SAVE_COLS = ['uid','oid','mjd','mag','magerr','mag_orig','clrcoeff','limitmag']

	total_requests = 6000
	n_workers  = args.n_workers
	n_requests = total_requests // n_workers

	ztf_oids_fname  = cfg.USER.W_DIR + "python/pipeline/queries/ztf/{}/ztf_oids.csv".format(OBJ)
	output_folder = cfg.USER.W_DIR + "data/surveys/ztf/{}/dr6/".format(OBJ)

	# clear previous saved lightcurves and write header
	for band in 'gri':
		for i in range(4):
			with open(output_folder+'{}_band/lc_{:01d}.csv'.format(band, i), 'w') as file:
				file.write(','.join(SAVE_COLS) + '\n')

	with Pool(n_workers) as pool:
		pool.map(obtain_ztf_lightcurves,range(n_workers))
