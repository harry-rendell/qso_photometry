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
    parser.add_argument("--bands", type=str, required=True, help="one or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--frame", type=str, help=("OBS or REST to specify rest frame or observer frame time. \n"
                                                   "Defaults to rest frame for Quasars and observer time for Stars.\n"))
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
    
    #TODO: make these both arguments
    n_points = 20
    log_or_lin = 'log'



    # keyword arguments to pass to our reading function
    kwargs = {'dtypes': cfg.PREPROC.dtdm_dtypes,
              'nrows': nrows,
              'usecols': [ID,'dt','dm','de','dsid'],
              'ID':ID,
              'mjd_key':mjd_key,
              'log_or_lin':log_or_lin,
              'inner':False,
              'features':['n', 'mean weighted a', 'mean weighted b', 'SF cwf a', 'SF cwf b', 'SF cwf p', 'SF cwf n', 'skewness', 'kurtosis'],
              'n_points':n_points}

    for band in args.bands:
        # set the maximum time to use for this band
        if args.frame:
            max_t = cfg.PREPROC.MAX_DT[args.frame][OBJ][band]
        elif OBJ == 'qsos':
            max_t = cfg.PREPROC.MAX_DT['REST']['qsos'][band]
        elif OBJ == 'calibStars':
            max_t = cfg.PREPROC.MAX_DT['OBS']['calibStars'][band]
        
        # create time bins given the maximum time
        if log_or_lin.startswith('log'):
            mjd_edges = np.logspace(0, np.log10(max_t), n_points+1)
        elif log_or_lin.startswith('lin'):
            mjd_edges = np.linspace(0, max_t, n_points+1)

        # add these back into the kwargs dictionary
        kwargs['band'] = band
        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/clean/dtdm_{band}'
        kwargs['max_t'] = max_t
        kwargs['mjd_edges'] = mjd_edges

        start = time.time()
        print('band:',band)

        # create output directories
        output_dir = os.path.join(cfg.D_DIR, f'computed/{OBJ}/dtdm_stats/all/{log_or_lin}/{band}')
        print(f'creating output directory if it does not exist: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

        results = data_io.dispatch_function(pairwise.calculate_stats_looped, chunks=None, max_processes=cfg.USER.N_CORES, concat_output=False, **kwargs)
        n_chunks = len(results)
        # repack results
        results = {key:np.array([result[key] for result in results]) for key in kwargs['features']}

        pooled_results = {key:np.zeros(shape=(n_points, 2)) for key in kwargs['features']}
        pooled_results['n'] = np.zeros(shape=(n_points), dtype='uint64')
        
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
            np.savetxt(os.path.join(output_dir, f"pooled_{key.replace(' ','_')}.csv"), pooled_results[key])
        np.savetxt(os.path.join(output_dir, 'mjd_edges.csv'), mjd_edges)

        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
