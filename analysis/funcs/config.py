"""
This file uses easydict to set configurables which may be fetched in the module code
"""

from easydict import EasyDict as edict
__C = edict()
# Configurables may be fetched using cfg
cfg = __C

# imports
import numpy as np


#------------------------------------------------------------------------------
# User settings
#------------------------------------------------------------------------------
__C.USER = edict()

# Working directory
__C.USER.W_DIR = '/disk1/hrb/python/'

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
__C.COLLECTION.SDSS.dtypes = {
					'mjd': np.float32,
					'mag': np.float32, # converted from real[4] with similar precision
					'magerr': np.float32, # converted from real[4] with similar precision
					'uid': np.uint32,
					'uid_s': np.uint32,
					'g-r': np.float32,
					'r-i': np.float32,
					'i-z': np.float32
						 }

__C.COLLECTION.PS = edict()
# Datatypes
__C.COLLECTION.PS.dtypes = {
					'objID': np.uint64, 
					'obsTime': np.float32, # converted from float[8] with reduced precision
					'psfFlux': np.float64, # converted from float[8] with similar precision. Since this is flux we use double precision.
					'psfFluxErr': np.float64, # converted from float[8] with similar precision.
					'mjd': np.float32,
					'mag': np.float32,
					'magerr': np.float32,
					'uid': np.uint32,
					'uid_s': np.uint32
						 }

__C.COLLECTION.ZTF = edict()
# Datatypes
__C.COLLECTION.ZTF.dtypes = {
					'oid': np.uint64, # note, uint32 is not large enough for ztf oids
					'clrcoeff': np.float32,
					'clrcounc': np.float32,
					'mjd': np.float32, # reduced from float64
					'mag': np.float32,
					'magerr': np.float32, 
					'uid': np.uint32,
					'uid_s': np.uint32
						 }


# 'uid': np.uint32,
# 'uid_s':np.uint32,
# 'catalogue': np.uint8,
#------------------------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------------------------
__C.PREPROC = edict()

# Datatypes
__C.PREPROC.dtypes = {
					'mjd': np.float32,
					'mag': np.float32,
					'mag_orig': np.float32,
					'magerr': np.float32,
					'uid': np.uint32,
					'uid_s': np.uint32,
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
# 5 sigma limiting magnitudes
__C.PREPROC.LIMIT_MAG.ZTF = {
							'g': 20.8,
							'r': 20.6,
							'i': 19.9
							}

# Magnitude error threshold
__C.PREPROC.MAG_ERR_THRESHOLD = 0.198


#------------------------------------------------------------------------------
# Analysis
#------------------------------------------------------------------------------
__C.ANALYSIS = edict()
