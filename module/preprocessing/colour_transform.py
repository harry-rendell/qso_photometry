import numpy as numpy
import pandas as pd
from ..config import cfg

def transform_ztf_to_ps(df, obj, band):
    """
    Add a column onto df with magnitudes transformed to the PS system
    """
    ID = df.index.name
    colors = pd.read_csv(cfg.USER.D_DIR + 'computed/{}/colors_sdss.csv'.format(obj), usecols=[ID,'mean_gr','mean_ri']).set_index(ID)
    df = df.join(colors, how='inner', on=ID).rename({'mag':'mag_orig'}, axis=1) #merge colors onto ztf df   
    if (band == 'r') | (band == 'g'):
        df['mag'] = (df['mag_orig'] + df['clrcoeff']*df['mean_gr']).astype(cfg.COLLECTION.ZTF.dtypes.mag)
    elif band == 'i':
        df['mag'] = (df['mag_orig'] + df['clrcoeff']*df['mean_ri']).astype(cfg.COLLECTION.ZTF.dtypes.mag)
    else:
        raise Exception('Unrecognised band: '+band)
    return df[['mjd', 'mag', 'mag_orig', 'magerr']].dropna(subset=['mag']) # There are some NaN entries in colors_sdss.csv

def transform_sdss_to_ps(df, color='g-r', system='tonry'):
    """
    Add a column onto df with magnitudes transformed to the PS system.
    There are few options of published transformations available. Here we use ones from Tonry 2012.
    TODO: Move transformations to data/assets (unversioned).
    """
    color_transf = pd.read_csv(cfg.USER.W_DIR+'pipeline/transformations/transf_to_ps_{}.txt'.format(system), sep='\s+', index_col=0)
    df = df.rename({'mag':'mag_orig'}, axis=1)
    df['mag'] = 0
    for band in 'griz':
        a0, a1, a2, a3 = color_transf.loc[band].values
        # Convert to PS mags
        slidx = pd.IndexSlice[:, band]
        x = df.loc[slidx, color]
        df.loc[slidx, 'mag'] = df.loc[slidx, 'mag_orig'] + a0 + a1*x + a2*(x**2) + a3*(x**3)
    return df[['mjd', 'mag', 'mag_orig', 'magerr']].dropna(subset=['mag'])