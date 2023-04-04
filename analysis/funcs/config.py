"""
This file uses easydict to set configurables which may be fetched in the module code
"""

from easydict import EasyDict as edict

__C = edict()
# Configurables may be fetched using cfg
cfg = __C

#
# User settings
#
__C.USER = edict()

# Working directory
__C.USER.W_DIR = '/disk1/hrb/python/'

# Set below to True to use multiple cores during computationally intensive tasks.
# Single core is not currently well supported, may cause errors when setting this to False.
__C.USER.USE_MULTIPROCESSING = True

# Choose how many cores to use
__C.USER.N_CORES = 4

#
# Collection
#
#__C.COLLECTION = edict()

#
# Preprocessing
#
__C.PREPROC = edict()

__C.PREPROC.MAG_ERR_THRESHOLD = 0.198

#
# Analysis
#
#__C.ANALYSIS = edict()