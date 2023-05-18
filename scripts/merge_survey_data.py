import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band", type=str, required=True, help="filterband for analysis")
    parser.add_argument("--survey", type=str, required=True, help="SDSS, PS or ZTF")
    parser.add_argument("--n_cores", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Use this flag to print output instead of saving")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%X %x'))
    print('args:',args)
    
    OBJ = args.object
    ID = 'uid' if (OBJ == 'qsos') else 'uid_s'

    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    SAVE_COLS = [ID,'mjd','mag','magerr','band','sid']

    for survey in args.survey.lower().split(' '):
        for band in args.band.lower():
            nrows  = None
            kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
                      'nrows': nrows,
                      'ID':ID,
                      'basepath': cfg.USER.D_DIR + 'surveys/{}/{}/clean/{}_band/'.format(survey, OBJ, band)}
            
            df = data_io.dispatch_reader(kwargs, multiproc=True, max_processes=args.n_cores)
            df['band'] = np.array(band, dtype=np.dtype(('U',1)))
            df['sid'] = np.array(cfg.PREPROC.SURVEY_IDS[survey], dtype=np.uint8)
            
            for uid, chunk in zip(*parse.split_into_non_overlapping_chunks(df, args.n_cores, bin_size=15000, return_bin_edges=True)):
                if not chunk.empty:
                    output_name = cfg.USER.D_DIR + 'merged/{}/clean/lc_{:06d}_{:06d}.csv'.format(OBJ,uid[0],uid[1])
                    if not os.path.exists(output_name):
                        with open(output_name, 'w') as file:
                            file.write(','.join(SAVE_COLS) + '\n')
                    chunk.to_csv(output_name, mode='a', header=False)