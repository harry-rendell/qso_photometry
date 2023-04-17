import pandas as pd
import numpy as np
import time
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
    args = parser.parse_args()
    
    OBJ = args.object.lower()
    BAND   = args.band.lower()
    SURVEY = args.survey.lower()

    ID = 'uid' if (OBJ == 'qsos') else 'uid_s'
    wdir   = cfg.USER.W_DIR
    nrows = args.n_rows
    skiprows = None if nrows == None else nrows * args.n_skiprows
    
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
          'nrows': nrows,
          'skiprows': skiprows,
          'basepath': wdir+'data/surveys/{}/{}/unclean/{}_band/'.format(SURVEY, OBJ, BAND),
          'ID':ID}

    df = data_io.dispatch_reader(kwargs, multiproc=True)

    # Remove obviously bad data
    parse.filter_data(df, bounds=cfg.PREPROC.FILTER_BOUNDS, dropna=True, inplace=True)

    # Read in grouped data
    grouped = pd.read_csv(wdir+'data/surveys/{}/{}/unclean/{}_band/grouped.csv'.format(SURVEY, OBJ, BAND), usecols=[ID, 'mag_med','mag_std']).set_index(ID)
    uids = grouped.index[grouped['mag_med']<cfg.PREPROC.LIMIT_MAG[SURVEY.upper()][BAND]]
    df = df[df.index.isin(uids)]

    # Discretise mjd
    df['mjd_floor'] = np.floor(df['mjd']).astype('uint32')

    # Separate observations into ones which share another observation on the same night (multi_obs) and those that do not (single_obs)
    mask = df.reset_index()[[ID,'mjd_floor']].duplicated(keep=False).values
    single_obs = df[~mask].drop(columns='mjd_floor')
    multi_obs = df[mask]
    del df

    # Join lightcurve median and std onto dataframe
    multi_obs = multi_obs.join(grouped[['mag_med','mag_std']], on=ID)
    
    chunks = parse.split_into_non_overlapping_chunks(multi_obs, cfg.USER.N_CORES)
    kwargs = {'dtypes':cfg.PREPROC.lc_dtypes}

    start = time.time()
    averaged = data_io.dispatch_function(lightcurve_statistics.groupby_apply_average, chunks, kwargs)
    end   = time.time()
    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(end-start)))

    # Add the data from nights with multiple observations back on
    df = single_obs.append(averaged).sort_values([ID,'mjd'])
    df.index = df.index.astype('uint32')

    # Add comment to start of csv file
    comment = """# CSV of photometry transformed to PS with preprocessing and cleaning.
# mag      : transformed photometry in PanSTARRS photometric system
# mag_orig : original photometry in native {} photometric system.\n""".format(SURVEY.upper())

    # keyword arguments to pass to our writing function
    kwargs = {'comment':comment,
              'basepath':cfg.USER.W_DIR + 'data/surveys/{}/{}/clean/{}_band/'.format(SURVEY, OBJ, BAND)}

    chunks = parse.split_into_non_overlapping_chunks(df, 4)
    data_io.dispatch_writer(chunks, kwargs)

