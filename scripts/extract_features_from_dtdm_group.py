import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse, lightcurve_statistics, pairwise, binning
from module.classes.analysis import analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band", type=str, required=True, help="single filterband for analysis")
    parser.add_argument("--property", type=str, required=True, help="QSO property to use for splitting features")
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


    dr = analysis(ID, OBJ, 'r')
    dr.read_vac(catalogue_name='dr16q_vac')
    dr.vac = parse.filter_data(dr.vac, cfg.PREPROC.VAC_BOUNDS, dropna=False)

    bounds_z = np.array([-3.5,-1.5,-1,-0.5,0,0.5,1,1.5,3.5])
    groups, bounds_values = binning.calculate_groups(dr.vac[args.property], args.property, bounds = bounds_z)
    n_groups = len(groups)
    if log_or_lin.startswith('log'):
        mjd_edges = np.logspace(0, np.log10(15767), n_points+1)
    elif log_or_lin.startswith('lin'):
        mjd_edges = np.linspace(0, max_t, n_points+1)

    # n_points=15 # number of points to plot
    # if log_or_lin.startswith('log'):
    #     self.mjd_edges = np.logspace(0, 4.01, n_points+1) # TODO add max t into argument
    # elif log_or_lin.startswith('lin'):
    #     self.mjd_edges = np.linspace(0, 10232.9, n_points+1)

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
              'mjd_edges':mjd_edges,
              'groups':groups,
              'property':args.property}

    start = time.time()

    results = data_io.dispatch_function(pairwise.calculate_stats_looped_key, chunks=None, max_processes=cfg.USER.N_CORES, concat_output=False, **kwargs)
    n_chunks = len(results)
    # repack results
    results = {key:np.array([result[key] for result in results]) for key in kwargs['features']}

    pooled_results = {key:np.zeros(shape=(kwargs['n_points'], n_groups, 2)) for key in kwargs['features']}
    pooled_results['n'] = np.zeros(shape=(kwargs['n_points'], n_groups), dtype='uint64')
    

    for key in results.keys():
        if key != 'n':
            # Two options. Could iterate over the group_idx, leaving pooled_mean = 0 in cases where the weights sum to zero
            # or, use 15767 (for Lbol) as the highest ∆t so that we can use the code below. There is a large discrepancy between max dt for differently grouped
            # qsos, because LboL is correlated with redshift (observational bias)
            pooled_mean = np.average(results[key][:,:,:,0], weights=results['n'], axis=0)
            pooled_var  = np.average(results[key][:,:,:,1], weights=results['n'], axis=0) + np.average((results[key][:,:,:,0]-pooled_mean)**2, weights=results['n'], axis=0) # this second term should be negligible since the SF of each chunk should be the same. # Note that the pooled variance is an estimate of the *common* variance of several populations with different means.
            if key.startswith('SF'):
                pooled_results[key][:,:,0] = pooled_mean ** 0.5 # Square root to get SF instead of SF^2
                pooled_results[key][:,:,1] = pooled_var ** 0.5 # Square root to get std instead of var
            else:
                pooled_results[key][:,:,0] = pooled_mean
                pooled_results[key][:,:,1] = pooled_var
        else:
            pooled_results[key] = results[key].sum(axis=0)

    save_dir = cfg.D_DIR + 'computed/{}/dtdm_stats/{}/{}/'.format(OBJ, args.property, kwargs['log_or_lin'])
    os.makedirs(save_dir, exist_ok=True)
    for key in pooled_results.keys():
        for group_idx in range(n_groups):
            np.savetxt(save_dir + 'pooled_{}_{}.csv'.format(key.replace(' ','_'), group_idx), pooled_results[key][:, group_idx])
    np.savetxt(save_dir + 'mjd_edges.csv', mjd_edges)
    np.savetxt(save_dir + 'bounds_values.csv', bounds_values)

    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
