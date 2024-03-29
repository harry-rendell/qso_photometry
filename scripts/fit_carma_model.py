import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io
from module.modelling import carma
from module.assets import load_grouped_tot
from functools import partial

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="One or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows",  type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Whether to do a dry run (i.e. don't save the output)")
    parser.add_argument("--survey",   type=str, help="name of survey to restrict data to. If left blank, then all surveys are used.")
    parser.add_argument("--model",  type=str, required=True, help ="Model to fit to the data. Options are drw or dho.")
    parser.add_argument("--nobs_min",type=int, help="Minimum number of observations required to fit a model.")
    parser.add_argument("--best_phot", action='store_true', help="Use this flag to only save the best subset of the data")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    if OBJ == 'qsos':
        ID = 'uid'
        mjd_key = 'mjd_rf'
        # mjd_key = 'mjd'
    else:
        ID = 'uid_s'
        mjd_key = 'mjd'

    if args.best_phot:
        phot_str = 'best_phot'
    else:
        phot_str = 'clean'

    nrows = args.n_rows
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    # keyword arguments to pass to our reading function
    kwargs = {'obj':OBJ,
              'dtypes': cfg.PREPROC.lc_dtypes,
              'nrows': nrows,
              'usecols': [ID,mjd_key,'mag','magerr','band','sid'],
              'ID':ID,
              'mjd_key':mjd_key}

    if args.survey:
        kwargs['sid'] = cfg.PREPROC.SURVEY_IDS[args.survey]
    else:
        args.survey = 'all'

    for band in args.band:
        # add these back into the kwargs dictionary
        if args.nobs_min:
            n_tot = load_grouped_tot(OBJ, band, usecols=['n_tot']).squeeze()
            uid_subset = n_tot[(n_tot > args.nobs_min).values].index
            kwargs['subset'] = uid_subset
        else:
            args.nobs_min = 0

        kwargs['band'] = band
        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/{phot_str}/'

        start = time.time()
        print('band:',band)

        if args.model == 'drw':
            f = partial(data_io.groupby_apply_dispatcher, carma.apply_drw_fit)
        elif args.model == 'dho':
            f = partial(data_io.groupby_apply_dispatcher, carma.apply_dho_fit)
        else:
            raise ValueError('Invalid model selected. Options are drw or dho.')

        results = data_io.dispatch_function(f, max_processes=cfg.USER.N_CORES, concat_output=True, **kwargs)
        results.to_csv(cfg.D_DIR + f'computed/{OBJ}/features/{args.model}_fits_{band}_{args.survey}_{args.nobs_min}_{phot_str}_rest_frame.csv')
        # results.to_csv(cfg.D_DIR + f'computed/{OBJ}/features/{args.model}_fits_{band}_{args.survey}_{args.nobs_min}_{phot_str}.csv')
        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))