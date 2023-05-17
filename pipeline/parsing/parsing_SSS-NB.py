# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress
from astropy.time import Time
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
from module.config import cfg
from module.preprocessing import parse, data_io
from module.preprocessing.colour_transform import ssa_transform
# import matplotlib
# font = {'size' : 18}
# matplotlib.rc('font', **font)
# -

pd.set_option('max_colwidth', 100, 'max_columns',100)
# Available cols:
# ['surveyID', 'surveyName', 'systemID', 'fieldOfView', 'decMin', 'decMax', 'numFields', 'telescope', 'telAperture', 'telLong', 'telLat', 'telHeight', 'plateScale', 'colour', 'waveMin', 'waveMax', 'waveEff', 'magLimit', 'epochMin', 'epochMax', 'epTsys', 'equinox', 'eqTsys', 'surveyRef']
pd.read_csv(cfg.USER.D_DIR + 'surveys/supercosmos/survey_table.csv', usecols= ['surveyID', 'surveyName', 'fieldOfView', 'decMin', 'decMax',
                                                                               'numFields', 'telescope', 'telAperture',
                                                                               'colour', 'waveMin', 'waveMax', 'waveEff',
                                                                               'magLimit', 'epochMin', 'epochMax']).set_index('surveyID')

# + active=""
# Note: surveyID = 4 has only 1 datapoint for qsos and none for stars, so we leave it out.
#
# r band (R):
# * 2 SERC-R/AAO-R (south)
# * 5,9 POSSI-E(N) (north), POSSI-E(S) (south) (should be the same instrumentation therefore, same transformations.)
# * 7 POSSII-R (south)
#
# g band (B, blue):
# * 1 SERC-J/EJ (south)
# * 6 POSSII-B (north)
#
# i band (near-infrared)
# * 3 SERC-I (south)
# * 8 POSSII-I (north)
#
#
# -

# ---
# # Cell below is for acquiring SSA data

# +
PARSE_SECONDARY = True
SAVE_COORDS_IN_CHUNKS = True
SAVE_SECONDARY = False

# ID = 'uid_s'
# OBJ = 'calibStars'
ID = 'uid'
OBJ = 'qsos'

if PARSE_SECONDARY:
    def remove_nonmatches(df):
        df_sorted = df.reset_index().sort_values('distance')
        mask = df_sorted.duplicated(subset=[ID,'mjd','filterID','surveyID'], keep='first').values
        return df_sorted[~mask].sort_values([df.index.name,'mjd','distance']).set_index(df.index.name)

    if SAVE_COORDS_IN_CHUNKS:
        """
        Run this block to save our coordinates in a size and format that ssa.roe.ac.uk/xmatch.html can handle
        """
        coords = pd.read_csv(cfg.USER.D_DIR + 'catalogues/{}/{}_subsample_coords.csv'.format(OBJ, OBJ), comment='#', dtype={'ra':np.float32, 'dec':np.float32})
        for i, chunk in enumerate(np.array_split(coords,5)):
            chunk.to_csv(cfg.USER.D_DIR + 'surveys/supercosmos/{}/coord_chunks/chunk_{}.csv'.format(OBJ, i+1), index=False, header=False, columns=[ID, 'ra', 'dec'], sep=' ')
    
    if SAVE_SECONDARY:
        # Note that surveyID in detection and plate match up completely
        results_path = cfg.USER.D_DIR + 'surveys/supercosmos/{}/results/'.format(OBJ)
        n_files = len([f for f in os.listdir(results_path) if f.endswith('_results.csv')])
        detection = pd.concat([pd.read_csv(results_path + 'chunk_{}_results.csv'.format(i+1)) for i in range(n_files)])

        # The query to generate the table below is in queries/ssa.sql
        plate = pd.read_csv(cfg.USER.D_DIR + 'surveys/supercosmos/plate_table.csv', usecols = ['plateID','fieldID','filterID','utDateObs'])
        merged = detection.merge(plate, on='plateID').sort_values('up_name')

        # Convert arcmin to arcsec
        merged['distance'] *= 60

        # Remove rows with no matches
        merged = merged[merged['objID'] != 0]

        # Convert YYYY-MM-DD HH:MM:SS to MJD
        times = merged['utDateObs'].to_list()
        merged['mjd'] = Time(times).mjd

        merged['filterID'] = merged['filterID'].str.strip()
        merged = merged.rename(columns={'up_name':ID})
        merged = merged[[ID,'distance','smag','surveyID','filterID','mjd']].set_index(ID)
        merged = remove_nonmatches(merged)
        merged = merged[merged['distance']<1.5]

        # Plot out distance matching
        fig, ax = plt.subplots(1,1, figsize=(15,5))
        ax.hist(merged['distance'], bins=100);


        merged.astype({'smag':np.float32, 'surveyID':np.uint8, 'mjd':np.uint32}).to_csv(cfg.USER.D_DIR + 'surveys/supercosmos/{}/ssa_secondary.csv'.format(OBJ))
# -

# some pre-determined coefficients for transformations. 
p_r1 = np.array([-0.09991162, -0.19239214])
p_r2 = np.array([-0.21768903, -0.15050923])
p_dict = {'r1':p_r1, 'r2':p_r2}
p_r1_ivezic = np.array([-0.0107, +0.0050, -0.2689, -0.1540]) # Data from ivezic: https://arxiv.org/abs/astro-ph/0701508

# # Plot transformations
# ---

# + jupyter={"outputs_hidden": true}
for obj in ['calibStars', 'qsos']:
    for ssa_band in ['r1','r2']:
        print('band:',ssa_band,' - ','object:',obj)
        ssa = ssa_transform(obj, 'r', ssa_band, 'ps')
        ssa.read()
        # color_name = 'g-r'
        color_name = 'r-i'
        # color_name = 'i-z'
        ssa.color_transform(p=p_dict[ssa_band], color_name=color_name)
        ssa.hist_1d()
        ssa.hist_2d(color_name=color_name)
        # ssa.color_transform(p=p_r1_ivezic, color_name=color_name)
        # ssa.hist_1d()
        # ssa.hist_2d(color_name=color_name)
        ssa.mag_correlation()
# -

# # Transform and save data
# ---
# May be run without the cells above

# +
# some pre-determined coefficients for transformations. 
p_r1 = np.array([-0.09991162, -0.19239214])
p_r2 = np.array([-0.21768903, -0.15050923])
p_dict = {'r1':p_r1, 'r2':p_r2}
p_r1_ivezic = np.array([-0.0107, +0.0050, -0.2689, -0.1540]) # Data from ivezic: https://arxiv.org/abs/astro-ph/0701508

obj = 'calibStars'
ID = 'uid' if obj == 'qsos' else 'uid_s'
r1 = ssa_transform(obj, 'r', 'r1', 'ps')
r1.read()
color_name = 'r-i'
r1.color_transform(p=p_dict['r1'], color_name=color_name)

r2 = ssa_transform(obj, 'r', 'r2', 'ps')
r2.read()
color_name = 'r-i'
r2.color_transform(p=p_dict['r2'], color_name=color_name)

r1.df['mag'] = r1.mag_ssa_transf
r2.df['mag'] = r2.mag_ssa_transf
dfr = pd.concat([r2.df, r1.df]).sort_values([ID,'mjd'])
dfr['magerr'] = 0.06751*dfr['mag'] - 1.08
dfr['magerr'][(dfr['mag']<18.5).values] = 0.168935

# Add comment to start of csv file
comment =  ("# CSV of photometry with no other preprocessing or cleaning.\n"
            "# mag : photometry in native PanSTARRS photometric system")

band='r'
chunks = parse.split_into_non_overlapping_chunks(dfr.astype({'mag': np.float32, 'mag_orig': np.float32, 'magerr': np.float32}), 4)
# keyword arguments to pass to our writing function
kwargs = {'comment':comment,
          'basepath':cfg.USER.D_DIR + 'surveys/supercosmos/{}/unclean/{}_band/'.format(obj, band),
          'savecols':['mjd','mag','mag_orig','magerr']}

# data_io.dispatch_writer(chunks, kwargs)
# -
r1.df

r2.df


