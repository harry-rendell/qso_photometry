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
    parser.add_argument("--band", type=str, required=True, help="single filterband for analysis")
    parser.add_argument("--n_cores", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
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
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    log_or_lin = 'log'
    max_t = cfg.PREPROC.MAX_DT_REST_FRAME[OBJ][args.band]
    n_points = 20

    if log_or_lin.startswith('log'):
        mjd_edges = np.logspace(0, np.log10(max_t), n_points+1)
    elif log_or_lin.startswith('lin'):
        mjd_edges = np.linspace(0, max_t, n_points+1)


    # keyword arguments to pass to our reading function
    kwargs = {'dtypes': cfg.PREPROC.dtdm_dtypes,
              'nrows': nrows,
              'basepath': cfg.D_DIR + f'merged/{OBJ}/clean/dtdm_{args.band}',
              'usecols': [ID,'dt','dm','de','dsid'],
              'ID':ID,
              'band':args.band,
              'mjd_key':mjd_key,
              'log_or_lin':log_or_lin,
              'inner':False,
              'max_t':max_t,
              'features':['n', 'mean weighted a', 'mean weighted b', 'SF cwf a', 'SF cwf b', 'SF cwf p', 'SF cwf n', 'skewness', 'kurtosis'],
              'n_points':n_points,
              'mjd_edges':mjd_edges}
    

    start = time.time()

    results = data_io.dispatch_function(pairwise.calculate_stats_looped, chunks=None, max_processes=cfg.USER.N_CORES, concat_output=False, **kwargs)
    n_chunks = len(results)
    # repack results
    results = {key:np.array([result[key] for result in results]) for key in kwargs['features']}

    pooled_results = {key:np.zeros(shape=(kwargs['n_points'], 2)) for key in kwargs['features']}
    pooled_results['n'] = np.zeros(shape=(kwargs['n_points']), dtype='uint64')
    
    for key in results.keys():
        if key != 'n':
            if results['n'].sum()>0:
                pooled_mean = np.average(results[key][:,:,0], weights=results['n'], axis=0)
                pooled_var  = np.average(results[key][:,:,1], weights=results['n'], axis=0) + np.average((results[key][:,:,0]-pooled_mean)**2, weights=results['n'], axis=0)
                if key.startswith('SF'):
                    pooled_results[key][:,0] = pooled_mean ** 0.5 # Square root to get SF instead of SF^2
                    pooled_results[key][:,1] = pooled_var ** 0.5 # Square root to get std instead of var
                else:
                    pooled_results[key][:,0] = pooled_mean
                    pooled_results[key][:,1] = pooled_var
        else:
            pooled_results[key] = results[key].sum(axis=0)

    for key in pooled_results.keys():
        np.savetxt(cfg.D_DIR + 'computed/{}/dtdm_stats/{}/pooled_{}.csv'.format(OBJ, kwargs['log_or_lin'], key.replace(' ','_')), pooled_results[key])

    np.savetxt(cfg.D_DIR + 'computed/{}/dtdm_stats/{}/mjd_edges.csv'.format(OBJ, kwargs['log_or_lin']), mjd_edges)

    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
