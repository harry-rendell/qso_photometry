import numpy as numpy
import pandas as pd
from ..config import cfg

def transform_ztf_to_ps(df, obj, band):
    """
    Add a column onto df with magnitudes transformed to the PS system
    """
    ID = df.index.name
    colors = pd.read_csv(cfg.USER.W_DIR+'data/computed/{}/colors_sdss.csv'.format(obj), usecols=[ID,'mean_gr','mean_ri']).set_index(ID)
    df = df.join(colors, how='left', on=ID).rename({'mag':'mag_orig'}, axis=1) #merge colors onto ztf df   
    if (band == 'r') | (band == 'g'):
        df['mag'] = (df['mag_orig'] + df['clrcoeff']*df['mean_gr']).astype(cfg.COLLECTION.ZTF_dtypes.mag)
    elif band == 'i':
        df['mag'] = (df['mag_orig'] + df['clrcoeff']*df['mean_ri']).astype(cfg.COLLECTION.ZTF_dtypes.mag)
    else:
        raise Exception('Unrecognised band: '+band)
    return df[['mjd', 'mag', 'mag_orig', 'magerr']]