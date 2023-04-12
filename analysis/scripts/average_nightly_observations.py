import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.append('..')
print(os.getcwd())
from funcs.config import cfg
from funcs.preprocessing import data_io, parse, lightcurve_statistics

if __name__ == '__main__':
    OBJ    = 'qsos'
    ID     = 'uid'
    BAND   = 'r'
    wdir   = cfg.USER.W_DIR
    nrows  = None
    skiprows = None if nrows == None else nrows * 0
    SURVEY = 'ztf'
    print('using {} cores'.format(cfg.USER.N_CORES))

    kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
          'nrows': nrows,
          'skiprows': skiprows,
          'basepath': wdir+'data/surveys/{}/{}/unclean/{}_band/'.format(SURVEY, OBJ, BAND),
          'ID':ID}

    df = data_io.dispatch_reader(kwargs, multiproc=True)

    # Remove obviously bad data
    bounds={'mag':(15,25),'magerr':(0,2)}
    parse.filter_data(df, bounds=bounds, dropna=True, inplace=True)

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
    np.savetxt(wdir+'data/surveys/{}/{}/unclean/{}_band/average_nightly_obs_log.txt'.format(SURVEY, OBJ, BAND), np.array(log), fmt='%s')

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

