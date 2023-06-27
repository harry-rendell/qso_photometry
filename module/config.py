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

#------------------------------------------------------------------------------
# Path variables
#------------------------------------------------------------------------------
# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Working directory
__C.W_DIR = os.path.join(__C.ROOT_DIR, 'qso_photometry', '')

# Data directory
__C.D_DIR = os.path.join(__C.ROOT_DIR, 'data', '')

# Results directory.
__C.RES_DIR = os.path.join(__C.W_DIR, 'res', '')

# Path to thesis folder
__C.THESIS_DIR = os.path.join(__C.ROOT_DIR, 'thesis_hrb', '')

#------------------------------------------------------------------------------
# User settings
#------------------------------------------------------------------------------
__C.USER = edict()

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
# 5σ limiting magnitudes
__C.PREPROC.LIMIT_MAG.SDSS = {
							'u': 22.15,
							'g': 23.13,
							'r': 22.70,
							'i': 22.20,
							'z': 20.71
							}

# https://outerspace.stsci.edu/display/PANSTARRS/PS1+FAQ+-+Frequently+asked+questions
# 5σ limiting magnitudes
# Note, these are actually the 3pi stacked 5σ depths... Should really be using single epoch 5σ depths
__C.PREPROC.LIMIT_MAG.PS = {
							'g': 23.3,
							'r': 23.2,
							'i': 23.1,
							'z': 22.3,
							'y': 21.4

							# Below are the single epoch 5σ depths
							# 'g': 22.0,
							# 'r': 21.8,
							# 'i': 21.5,
							# 'z': 20.9,
							# 'y': 19.7

							}



# limitingmag can be fetched on a per-observation basis but below is an average
# Note, using limitmag to filter out observations may be biased as we are selectively removing
# 	dimmer observations.
# 5σ limiting magnitudes
__C.PREPROC.LIMIT_MAG.ZTF = {
							'g': 20.8,
							'r': 20.6,
							'i': 19.9
							}

# https://arxiv.org/abs/1607.01189, peacock_ssa
# 4σ limiting magnitudes, using the smaller of the two between UKST and POSS2.
__C.PREPROC.LIMIT_MAG.SUPERCOSMOS = {
									# 4σ
									'g': 21.17,
									'r': 20.30,
									'i': 18.90
									}
									# 5σ
									# 'g': 20.26,
									# 'r': 19.78,
									# 'i': 18.38
# Table from paper above.
# Band    | 5σ    | 4σ    |
# --------|-------|-------|
# UKST  B | 20.79 | 21.19 |
# UKST  R | 19.95 | 20.30 |
# UKST  I | 18.56 | 19.94 |
# POSS2 B | 20.26 | 21.17 |
# POSS2 R | 19.78 | 20.35 |
# POSS2 I | 18.38 | 18.90 |

# Magnitude error threshold
__C.PREPROC.MAG_ERR_THRESHOLD = 0.198

# Bounds to use on parse.filter_data in average_nightly_observations.py when removing bad data.
__C.PREPROC.FILTER_BOUNDS = {'mag':(15,25),'magerr':(0,2)}

__C.PREPROC.SURVEY_IDS =   {'supercosmos':3,
						 	'sdss': 5,
							'ps': 7,
						 	'ztf': 11}

__C.PREPROC.VAC_BOUNDS = {'z':(0,5),
						  'redshift':(0,5),
						  'Lbol':(44,48),
						  'Lbol_err':(0,1),
						  'MBH_HB':(6,12),
						  'MBH_HB_err':(0,1),
						  'MBH_MgII':(6,12),
						  'MBH_MgII_err':(0,1),
						  'MBH_CIV':(6,12),
						  'MBH_CIV_err':(0,1),
						  'MBH':(6,12),
						  'MBH_err':(0,1),
						  'nEdd':(-3,1),
						  'nEdd_err':(0,2),
						  'Mi:':(-30,-20),
						  'mag_mean':(15,23.5)}

# Max ∆t for quasars and stars in rest frame, rounded up to the nearest integer.
# Calculated from mjd_ptp_rf.max() from clean/grouped_{band}.csv
__C.PREPROC.MAX_DT_REST_FRAME = {'qsos':{'g': 13794, 'r': 24765, 'i': 13056},
								 'calibStars':{'g':np.nan, 'r':np.nan, 'i':np.nan}}

# Do the same except with each black hole property

# Max ∆t for quasras and stars in observer frame frame, rounded up to the nearest integer
# Calculated from mjd_ptp.max() from clean/grouped_{band}.csv
__C.PREPROC.MAX_DT_OBS_FRAME  = {'qsos':{'g':16513, 'r':26702, 'i':14698},
								 'calibStars':{'g':15122, 'r':26062, 'i':12440}}



#------------------------------------------------------------------------------
# Colour transformations
#------------------------------------------------------------------------------
__C.TRANSF = edict()
__C.TRANSF.SSA = edict()

# https://arxiv.org/abs/1607.01189, peacock_ssa
__C.TRANSF.SSA.PEACOCK = {'g_north': ('g-r',[-0.078, +0.134]),
		                  'g_south': ('g-r',[-0.058, +0.102]),
		                  'r2_north':('g-r',[+0.012, -0.054]),
		                  'r2_south':('g-r',[-0.002, -0.022]),
		                  'i_north': ('r-i',[+0.008, -0.024]),
		                  'i_south': ('r-i',[+0.022, -0.092])}

# https://arxiv.org/abs/astro-ph/0701508, Ivezic2007_photometric_standardisation
__C.TRANSF.SSA.IVEZIC = {'g_north':  ('g-r', [+0.2628, -0.7952, +1.0544, +0.0268]),
						 'g_south':  ('g-r', [+0.2628, -0.7952, +1.0544, +0.0268]),

						 'r1':       ('r-i', [-0.0107, +0.0050, -0.2689, -0.1540]),
			             'r2_north': ('r-i', [-0.0107, +0.0050, -0.2689, -0.1540]),
			             'r2_south': ('r-i', [-0.0107, +0.0050, -0.2689, -0.1540]),
			             
			             'i_north':  ('r-i', [-0.0307, +0.1163, -0.3341, -0.3584]),
			             'i_south':  ('r-i', [-0.0307, +0.1163, -0.3341, -0.3584])}

#------------------------------------------------------------------------------
# Analysis and results
#------------------------------------------------------------------------------
__C.RES = edict()






#------------------------------------------------------------------------------
# Figures
#------------------------------------------------------------------------------
__C.FIG = edict()

# Path to style files. Empty string at end ensures trailing slash
__C.FIG.STYLE_DIR = os.path.join(__C.RES_DIR, 'styles', '')


