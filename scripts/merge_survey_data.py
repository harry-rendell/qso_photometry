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
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Use this flag to print output instead of saving")
    args = parser.parse_args()
    print('args:',args)
    
    OBJ = args.object.lower()
    ID     = 'uid'
    SAVE_COLS = ['uid','mjd','mag','magerr','band','sid']

    for SURVEY in ['sdss','ps','ztf']:
        for BAND in 'gri':
            nrows  = None
            kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
                      'nrows': nrows,
                      'ID':ID,
                      'basepath': cfg.USER.D_DIR + 'surveys/{}/{}/clean/{}_band/'.format(SURVEY, OBJ, BAND)}
            
            df = data_io.dispatch_reader(kwargs, multiproc=True)
            df['band'] = np.array(BAND, dtype=np.dtype(('U',1)))
            df['sid'] = np.array(cfg.PREPROC.SURVEY_IDS[SURVEY], dtype=np.uint8)
            
            for uid, chunk in zip(*parse.split_into_non_overlapping_chunks(df, 36, bin_size=15000)):
                if not chunk.empty:
                    output_name = cfg.USER.D_DIR + 'merged/{}/clean/lc_{:06d}_{:06d}.csv'.format(OBJ,uid[0],uid[1])
                    if not os.path.exists(output_name):
                        with open(output_name, 'w') as file:
                            file.write(','.join(SAVE_COLS) + '\n')
                    chunk.to_csv(output_name, mode='a', header=False)