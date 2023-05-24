import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse, lightcurve_statistics, pairwise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band", type=str, required=True, help="filterband for analysis")
    parser.add_argument("--n_cores", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--n_skiprows", type=int, help="Number of chunks of n_rows to skip when reading in photometric data")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    if OBJ == 'qsos':
        ID = 'uid'
        mjd_key = 'mjd_rf'
    else:
        ID = 'uid_s'
        mjd_key = 'mjd'

    nrows = args.n_rows
    skiprows = args.n_skiprows
    if args.n_skiprows and nrows is not None:
        skiprows = nrows * args.n_skiprows
    
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    # keyword arguments to pass to our reading function
    kwargs = {'dtypes': {**cfg.PREPROC.lc_dtypes, **cfg.PREPROC.dtdm_dtypes},
              'nrows': nrows,
              'skiprows': skiprows,
              'basepath': cfg.D_DIR + f'merged/{OBJ}/clean/',
              'usecols': [ID,mjd_key,'mag','magerr','band','sid'],
              'ID':ID,
              'mjd_key':mjd_key}
              
    start = time.time()
    
    for band in args.band:
        bool_arr = pd.read_csv(cfg.D_DIR + f'catalogues/{OBJ}/sets/clean_{band}.csv', index_col=ID, comment='#')
        restricted_set = bool_arr.index[ bool_arr['vac'].values & np.any(bool_arr[['sdss','ps']].values, axis=1)]
        del bool_arr

        kwargs['band'] = band
        kwargs['subset'] = restricted_set

        data_io.dispatch_function(pairwise.groupby_save_pairwise, chunks=None, max_processes=cfg.USER.N_CORES, **kwargs)

    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
