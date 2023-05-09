"""
This file uses easydict to set configurables which may be fetched in the module code
"""

from easydict import EasyDict as edict
__C = edict()
# Configurables may be fetched using cfg
cfg = __C

# imports
import numpy as np
import os

# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

#------------------------------------------------------------------------------
# User settings
#------------------------------------------------------------------------------
__C.USER = edict()

# Working directory
__C.USER.W_DIR = os.path.join(__C.ROOT_DIR, 'qso_photometry', '')

# Data directory
__C.USER.D_DIR = os.path.join(__C.ROOT_DIR, 'data', '')

# Results directory.
__C.USER.R_DIR = os.path.join(__C.USER.W_DIR, 'res', '')

# Set below to True to use multiple cores during computationally intensive tasks.
# Single core is not currently well supported, may cause errors when setting this to False.
__C.USER.USE_MULTIPROCESSING = True
# Choose how many cores to use
__C.USER.N_CORES = 4


#------------------------------------------------------------------------------
# Data collection configurables
#------------------------------------------------------------------------------
__C.COLLECTION = edict()
# These dtypes should be checked against the datatypes provided when fetching lightcurve data
# float32 - float
# float64 - double

__C.COLLECTION.SDSS = edict()
# Datatypes
# mag converted from real[4] with similar precision
# magerr converted from real[4] with similar precision
__C.COLLECTION.SDSS.dtypes = {
							 **{'uid'    : np.uint32,  'uid_s'  : np.uint32,
								'objID'  : np.uint64,  'mjd'    : np.float32,
								'ra'     : np.float64, 'ra_ref' : np.float64,
								'dec'    : np.float64, 'dec_ref': np.float64,
								'get_nearby_distance': np.float32},
							 **{band + 'psf'   : np.float64 for band in 'ugriz'},
							 **{band + 'psferr': np.float64 for band in 'ugriz'},
						 }

__C.COLLECTION.PS = edict()
# Datatypes
__C.COLLECTION.PS.dtypes = {
					'objID'     : np.uint64, 
					'obsTime'   : np.float32, # converted from float[8] with reduced precision
					'psfFlux'   : np.float64, # converted from float[8] with similar precision. Since this is flux we use double precision.
					'psfFluxErr': np.float64, # converted from float[8] with similar precision.
					'mjd'       : np.float32,
					'mag'       : np.float32,
					'magerr'    : np.float32,
					'uid'       : np.uint32,
					'uid_s'     : np.uint32
						 }

__C.COLLECTION.ZTF = edict()
# Datatypes
__C.COLLECTION.ZTF.dtypes = {
					'oid'     : np.uint64, # note, uint32 is not large enough for ztf oids
					'clrcoeff': np.float32,
					'limitmag': np.float32,
					'mjd'     : np.float32, # reduced from float64
					'mag'     : np.float32,
					'magerr'  : np.float32, 
					'uid'     : np.uint32,
					'uid_s'   : np.uint32
						 }

__C.COLLECTION.CALIBSTAR_dtypes = {
							'ra'      : np.float64,
							'dec'     : np.float64,
							'n_epochs': np.uint32,
							**{'mag_mean_'+b    : np.float32 for b in 'gri'},
							**{'mag_mean_err_'+b: np.float32 for b in 'gri'}
								}


# 'uid': np.uint32,
# 'uid_s':np.uint32,
# 'catalogue': np.uint8,
#------------------------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------------------------
__C.PREPROC = edict()

# Datatypes
__C.PREPROC.lc_dtypes = {'mjd'     : np.float32,
						 'mag'     : np.float32,
						 'mag_orig': np.float32,
						 'magerr'  : np.float32,
						 'uid'     : np.uint32,
						 'uid_s'   : np.uint32,
						 'sid'     : np.uint8}

__C.PREPROC.stats_dtypes = {'n_tot': np.uint16, # Increase this to uint32 if we think we will have more than 2^16 (65,536) observations for a single object
						    **{x:np.float32 for x in ['mjd_min','mjd_max','mjd_ptp',
													  'mag_min','mag_max','mag_mean','mag_med',
													  'mag_std',
													  'mag_mean_native','mag_med_native',
													  'mag_opt_mean','mag_opt_mean_flux','magerr_opt_std',
													  'magerr_max','magerr_mean','magerr_med']}}

__C.PREPROC.dtdm_dtypes = {'uid'	: np.uint32,
						   'uid_s' 	: np.uint32,
						   'dm' 	: np.float32,
						   'dm' 	: np.float32,
						   'de'		: np.float32,
						   'dm2_de2': np.float32,
						   'dsid'	: np.uint8}

# maybe not needed as the types stay consistent
__C.PREPROC.pairwise_dtypes = {'uid': np.uint32,
							   'dt' :np.float32,
							   'dm' :np.float32,
							   'de' :np.float32
							   }

# Limiting magnitudes
__C.PREPROC.LIMIT_MAG = edict()

# https://www.sdss4.org/dr16/imaging/other_info/
# 5 sigma limiting magnitudes
__C.PREPROC.LIMIT_MAG.SDSS = {
							'u': 22.15,
							'g': 23.13,
							'r': 22.70,
							'i': 22.20,
							'z': 20.71
							}

# https://outerspace.stsci.edu/display/PANSTARRS/PS1+FAQ+-+Frequently+asked+questions
# 5 sigma limiting magnitudes
__C.PREPROC.LIMIT_MAG.PS = {
							'g': 23.3,
							'r': 23.2,
							'i': 23.1,
							'z': 22.3,
							'y': 21.4
							}

# limitingmag can be fetched on a per-observation basis but below is an average
# Note, using limitmag to filter out observations may be biased as we are selectively removing
# 	dimmer observations.
# 5 sigma limiting magnitudes
__C.PREPROC.LIMIT_MAG.ZTF = {
							'g': 20.8,
							'r': 20.6,
							'i': 19.9
							}

# Magnitude error threshold
__C.PREPROC.MAG_ERR_THRESHOLD = 0.198

# Bounds to use on parse.filter_data in average_nightly_observations.py when removing bad data.
__C.PREPROC.FILTER_BOUNDS = {'mag':(15,25),'magerr':(0,2)}

__C.PREPROC.SURVEY_IDS =   {'sss_r1': 1,
							'sss_r2': 3,
						 	'sdss': 5,
							'ps': 7,
						 	'ztf': 11}

#------------------------------------------------------------------------------
# Analysis and results
#------------------------------------------------------------------------------
__C.RES = edict()






#------------------------------------------------------------------------------
# Figures
#------------------------------------------------------------------------------
__C.FIG = edict()

# Path to style files. Empty string at end ensures trailing slash
__C.FIG.STYLE_DIR = os.path.join(__C.USER.W_DIR, 'res', 'styles', '')

