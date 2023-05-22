import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse

def multiproc_save():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band", type=str, required=True, help="filterband for analysis")
    parser.add_argument("--survey", type=str, required=True, help="SDSS, PS or ZTF")
    parser.add_argument("--n_chunks", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Use this flag to print output instead of saving")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    ID = 'uid' if (OBJ == 'qsos') else 'uid_s'

    if OBJ == 'qsos':
        SAVE_COLS = [ID,'mjd','mjd_rf','mag','magerr','band','sid']
        redshifts = pd.read_csv(cfg.USER.D_DIR + 'catalogues/qsos/dr14q/dr14q_redshift.csv').set_index(ID)
    else:
        SAVE_COLS = [ID,'mjd','mag','magerr','band','sid']
    uid_ranges, _ = parse.split_into_non_overlapping_chunks(None, 106, bin_size=5000, return_bin_edges=True)
    for uid_range in uid_ranges:
        output_name = cfg.USER.D_DIR + f'merged/{OBJ}/clean/lc_{uid_range}.csv'
        with open(output_name, 'w') as file:
            file.write(','.join(SAVE_COLS) + '\n')

    start = time.time()
    for survey in args.survey.lower().split(' '):
        print(survey)
        for band in args.band.lower():
            print(band)
            nrows  = None
            kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
                      'nrows': nrows,
                      'ID':ID,
                      'basepath': cfg.USER.D_DIR + f'surveys/{survey}/{OBJ}/clean/{band}_band/',
					  'usecols':[ID,'mjd','mag','magerr']}
            
            df = data_io.dispatch_reader(kwargs, multiproc=True, max_processes=32)
            df['band'] = np.array(band, dtype=np.dtype(('U',1)))
            df['sid'] = np.array(cfg.PREPROC.SURVEY_IDS[survey], dtype=np.uint8)
            if OBJ == 'qsos':
                df = df.join(redshift, on=ID)
                df['mjd_rf'] = df['mjd']/(1+df['z'])
            # this can be parallelised - give each chunk to a core. Change output of split_into_non_overlapping_chunks to output {:06d}_{:06d}
            #   then pass to data_io.dispatch_writer with mode='a'
            uid_ranges, chunks = parse.split_into_non_overlapping_chunks(df, 106, bin_size=5000, return_bin_edges=True)
            kwargs = {'basepath':cfg.USER.D_DIR + f'merged/{OBJ}/clean/',
                      'mode':'a',
					  'savecols':SAVE_COLS[1:]}
            data_io.dispatch_writer(chunks, kwargs=kwargs, max_processes=32, fname_suffixes=uid_ranges)

    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
