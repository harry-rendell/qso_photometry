import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from module.config import cfg
from module.preprocessing import data_io, parse
import pandas as pd

def read_merged_photometry(self, catalogue_of_properties = None, redshift=True, cleaned=True, nrows=None):
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
    # basepath = cfg.D_DIR + 'merged/{}/{}_band/with_ssa/'
    if cleaned:
        basepath = cfg.D_DIR + 'merged/{}/{}_band/'.format(self.obj, self.band)
    else:
        basepath = cfg.D_DIR + 'merged/{}/{}_band/unclean/'.format(self.obj, self.band)

    kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
              'nrows': nrows,
              'basepath': basepath, # we should make this path more general so it is consistent between surveys
              'ID': self.ID}

    self.df = data_io.dispatch_reader(kwargs, multiproc=True)

    # Remove objects with a single observation.
    self.df = self.df[self.df.index.duplicated(keep=False).values]
    
    if redshift:
        # add on column for redshift. Use squeeze = True when reading in a single column.
        self.redshifts = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr14q/dr14q_uid_desig_z.csv', index_col=self.ID, usecols=[self.ID,'z'], squeeze=True).rename('redshift')
        self.df = self.df.join(self.redshifts, how = 'left', on=self.ID)
        self.df['mjd_rf'] = self.df['mjd']/(1+self.df['redshift'])

    self.df = self.df.sort_values([self.ID, 'mjd'])
    assert self.df.index.is_monotonic, 'Index is not sorted'

def read_grouped(self):
    """
    This method assumes we are reading data for a single band
    TODO: we could iterate over self.band if multiple and return the respective data
    """
    if not hasattr(self, 'band'):
        raise Exception('Must specify a band to read grouped data.')
    self.sdss = pd.read_csv(cfg.D_DIR + f'surveys/sdss/{self.obj}/clean/{self.band}_band/grouped.csv', index_col=0)
    self.ps   = pd.read_csv(cfg.D_DIR + f'surveys/ps/{self.obj}/clean/{self.band}_band/grouped.csv', index_col=0)
    self.ztf  = pd.read_csv(cfg.D_DIR + f'surveys/ztf/{self.obj}/clean/{self.band}_band/grouped.csv', index_col=0)
    self.ssa  = pd.read_csv(cfg.D_DIR + f'surveys/supercosmos/{self.obj}/clean/{self.band}_band/grouped.csv', index_col=0)
    self.tot  = pd.read_csv(cfg.D_DIR + f'merged/{self.obj}/clean/grouped_{self.band}.csv', index_col=0)

def read_vac(self, catalogue_name='dr16q_vac'):
    if self.obj == 'calibStars':
        raise Exception('Stars have no redshift data')
    
    if catalogue_name == 'dr12_vac':
        # cols = z, Mi, L5100, L5100_err, L3000, L3000_err, L1350, L1350_err, MBH_MgII, MBH_CIV, Lbol, Lbol_err, nEdd, sdss_name, ra, dec, uid
        fpath = 'catalogues/qsos/dr12q/SDSS_DR12Q_BH_matched.csv'
        prop_range_all = {'Mi':(-30,-20),
                          'mag_mean':(15,23.5),
                          'mag_std':(0,1),
                          'redshift':(0,5),
                          'Lbol':(44,48),
                          'nEdd':(-3,0.5)}        

    elif catalogue_name == 'dr14_vac':
        # cols = ra, dec, uid, sdssID, plate, mjd, fiberID, z, pl_slope, pl_slope_err, EW_MgII_NA, EW_MgII_NA_ERR, FWHM_MgII_NA, FWHM_MgII_NA_ERR, FWHM_MgII_BR, FWHM_MgII_BR_ERR, EW_MgII_BR, EW_MgII_BR_ERR, MBH_CIV, MBH_CIV_ERR, MBH, MBH_ERR, Lbol
        fpath = 'catalogues/qsos/dr14q/dr14q_spec_prop_matched.csv'
        prop_range_all = {'mag_mean':(15,23.5),
                          'mag_std':(0,1),
                          'redshift':(0,5),
                          'Lbol':(44,48)}

    elif catalogue_name == 'dr16q_vac':
        # cols = ra, dec, redshift_vac, Lbol, Lbol_err, MBH_HB, MBH_HB_err, MBH_MgII, MBH_MgII_err, MBH_CIV, MBH_CIV_err, MBH, MBH_err, nEdd, nEdd_err
        fpath = 'catalogues/qsos/dr16q/dr16q_vac_shen_matched.csv'

    else:
        raise Exception('Unrecognised value-added catalogue')

    vac = pd.read_csv(os.path.join(cfg.D_DIR,fpath), index_col=self.ID)
    vac = vac.rename(columns={'z':'redshift_vac'});
    if catalogue_name == 'dr16q_vac':
        # Note, in dr16q, bad nEdd entries are set to 0 (exactly) so we can remove those.
        vac['nEdd'] = vac['nEdd'].where((vac['nEdd']!=0).values)
    self.vac = vac

def read_redshifts(self):
    if self.obj == 'calibStars':
        raise Exception('Stars have no redshift data')
    redshift = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr14q/dr14q_redshift.csv').set_index(ID)





