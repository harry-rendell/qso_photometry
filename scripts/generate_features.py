import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse, lightcurve_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band", type=str, required=True, help="filterband for analysis")
    parser.add_argument("--survey", type=str, required=True, help="SDSS, PS or ZTF")
    parser.add_argument("--n_cores", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--n_skiprows", type=int, help="Number of chunks of n_rows to skip when reading in photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Number of chunks of n_rows to skip when reading in photometric data")
    args = parser.parse_args()
    print('args:',args)


    OBJ = args.object.lower()
    BAND   = args.band.lower()
    SURVEY = args.survey.lower()

    ID = 'uid' if (OBJ == 'qsos') else 'uid_s'
    
    nrows = args.n_rows
    skiprows = args.n_skiprows
    if args.n_skiprows and nrows is not None:
        skiprows = nrows * args.n_skiprows
    
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
              'nrows': nrows,
              'skiprows': skiprows,
              'basepath': cfg.USER.W_DIR+'data/surveys/{}/{}/clean/{}_band/'.format(SURVEY, OBJ, BAND),
              'ID':ID}

    df = data_io.dispatch_reader(kwargs, multiproc=True)


    chunks = parse.split_into_non_overlapping_chunks(df, cfg.USER.N_CORES)
    kwargs = {'dtypes':cfg.PREPROC.stats_dtypes}
    grouped = data_io.dispatch_function(lightcurve_statistics.groupby_apply_features, chunks, kwargs)
    if args.dry_run:
        print(grouped)
    else:
        output_folder = cfg.USER.W_DIR+'data/surveys/{}/{}/clean/{}_band/'.format(SURVEY, OBJ, band) 
        grouped.to_csv(output_folder + 'features.csv')
