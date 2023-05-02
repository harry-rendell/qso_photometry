import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from module.config import cfg
from module.preprocessing import data_io
import pandas as pd

def read_in(self, catalogue_of_properties = None, redshift=True, cleaned=True, nrows=None):
    """
    Read in raw data

    Parameters
    ----------
    reader : function
            used for reading data
    catalogue_of_properties : dataframe
    """
    
    # Default to 4 cores
    # Use the path below for SSA
    # basepath = cfg.USER.D_DIR + 'merged/{}/{}_band/with_ssa/'
    if cleaned:
        basepath = cfg.USER.D_DIR + 'merged/{}/{}_band/'.format(self.obj, self.band)
    else:
        basepath = cfg.USER.D_DIR + 'merged/{}/{}_band/unclean/'.format(self.obj, self.band)

    kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
              'nrows': nrows,
              'basepath': basepath, # we should make this path more general so it is consistent between surveys
              'ID': self.ID}

    self.df = data_io.dispatch_reader(kwargs, multiproc=True)

    # Remove objects with a single observation.
    self.df = self.df[self.df.index.duplicated(keep=False)]
    
    if redshift:
        # add on column for redshift. Use squeeze = True when reading in a single column.
        self.redshifts = pd.read_csv(cfg.USER.D_DIR + 'catalogues/qsos/dr14q/dr14q_uid_desig_z.csv', index_col=self.ID, usecols=[self.ID,'z'], squeeze=True).rename('redshift')
        self.df = self.df.join(self.redshifts, how = 'left', on=self.ID)
        self.df['mjd_rf'] = self.df['mjd']/(1+self.df['redshift'])

    self.df = self.df.sort_values([self.ID, 'mjd'])
    assert self.df.index.is_monotonic, 'Index is not sorted'