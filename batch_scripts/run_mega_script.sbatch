#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=qso_sg
#SBATCH --mem=50G
#SBATCH -t 8:00:00
#SBATCH --mail-user=harry.rendell@ed.ac.uk

# Note, we can run this as a normal batch script using bash <filename>

# Load anaconda. Omit this if not running on cuillin
# source ~/.bashrc_old

# use same number of cores as cpus-per-task above
n_cores=32
surveys="supercosmos sdss ps ztf"
bands='gri'

# #------------------------------------------------------------------------------
# # Save grouped data (using uncleaned photometry)
# #------------------------------------------------------------------------------
# python scripts/save_grouped.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_save_grouped_qsos.log
# python scripts/save_grouped.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_save_grouped_calibStars.log


# #------------------------------------------------------------------------------
# # Average observations in the same night
# #------------------------------------------------------------------------------
# python scripts/average_nightly_observations.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_average_nightly_observations_qsos.log
# python scripts/average_nightly_observations.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_average_nightly_observations_calibStars.log


# #------------------------------------------------------------------------------
# # Save grouped data (using cleaned photometry)
# #------------------------------------------------------------------------------
# python scripts/save_grouped.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores --cleaned >& batch_scripts/logs/log_save_grouped_clean_qsos.log
# python scripts/save_grouped.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores --cleaned >& batch_scripts/logs/log_save_grouped_clean_calibStars.log

#------------------------------------------------------------------------------
# Merge survey data and save to to data/merged/clean/
#   NOTE 1: This appends instead of overwrites. Remove previously saved data if it is no longer needed.
#   NOTE 2: Don't forget to sort the data using sort_uid_mjd.sh after computing creating output (bash script below)
#------------------------------------------------------------------------------
# python scripts/merge_survey_data.py --object='qsos'       --band=$bands --survey="$surveys" --n_chunks=$n_cores >& batch_scripts/logs/log_merge_survey_data_qsos.log
# bash scripts/sort_uid_mjd.sh data/merged/qsos/clean
# python scripts/merge_survey_data.py --object='calibStars' --band=$bands --survey='supercosmos sdss ps ztf' --n_chunks=$n_cores >& batch_scripts/logs/log_merge_survey_data_calibStars.log
# bash scripts/sort_uid_mjd.sh data/merged/calibStars/clean


#------------------------------------------------------------------------------
# Save grouped data (using cleaned, merged photometry) and save to data/merged/clean/
#------------------------------------------------------------------------------
# python scripts/save_grouped_merged.py --object='qsos'       --band=$bands --n_cores=$n_cores >& batch_scripts/logs/log_save_grouped_merged_qsos.log
# python scripts/save_grouped_merged.py --object='calibStars' --band=$bands --n_cores=$n_cores >& batch_scripts/logs/log_save_grouped_merged_calibStars.log


#------------------------------------------------------------------------------
# Calculate and save dtdm. 
#   NOTE: This appends instead of overwrites. Remove previously saved data if it is no longer needed.
# ------------------------------------------------------------------------------
# python scripts/compute_dtdm_merged.py --object='qsos' --band=$bands --n_cores=$n_cores >& batch_scripts/logs/log_compute_dtdm_merged_qsos.log
# python scripts/compute_dtdm_merged.py --object='calibStars' --band=$bands --n_cores=$n_cores >& batch_scripts/logs/log_compute_dtdm_merged_calibStars.log


#------------------------------------------------------------------------------
# Extract and save features from dtdm
# ------------------------------------------------------------------------------
# python scripts/extract_features_from_dtdm.py --object='qsos'       --band=$bands --n_cores=$n_cores --name='log_40' --n_bins=40  >& batch_scripts/logs/log_extract_features_from_dtdm_qsos.log
# python scripts/extract_features_from_dtdm.py --object='calibStars' --band=$bands --n_cores=$n_cores --name='log_30' --n_bins=30 >& batch_scripts/logs/log_extract_features_from_dtdm_calibStars.log


#------------------------------------------------------------------------------
# Extract and save features from grouped dtdm (only applied to qsos)
# ------------------------------------------------------------------------------
# python scripts/extract_features_from_dtdm_group.py --object='qsos' --band=$bands --property='Lbol' --n_cores=$n_cores --name='log_30' --n_bins=30 >& batch_scripts/logs/log_extract_features_from_dtdm_group_qsos_lbol.log
# python scripts/extract_features_from_dtdm_group.py --object='qsos' --band=$bands --property='MBH' --n_cores=$n_cores --name='log_30' --n_bins=30 >& batch_scripts/logs/log_extract_features_from_dtdm_group_qsos_mbh.log
# python scripts/extract_features_from_dtdm_group.py --object='qsos' --band=$bands --property='nEdd' --n_cores=$n_cores --name='log_30' --n_bins=30 >& batch_scripts/logs/log_extract_features_from_dtdm_group_qsos_nedd.log

#------------------------------------------------------------------------------
# Bin dtdm data
# TODO: Rerun with inner in kwargs
# ------------------------------------------------------------------------------
# python scripts/dtdm_bin.py --object='calibStars' --band=$bands --n_cores=10 --n_bins_T=25  >& batch_scripts/logs/log_dtdm_bin_calibStars.log
# python scripts/dtdm_bin.py --object='qsos'       --band=$bands --n_cores=10 --n_bins_T=25  >& batch_scripts/logs/log_dtdm_bin_qsos.log
python scripts/dtdm_bin.py --object='calibStars' --band=$bands --n_cores=20 --n_bins_T=20 --inner --name='inner'  >& batch_scripts/logs/log_dtdm_bin_calibStars_inner.log
python scripts/dtdm_bin.py --object='qsos'       --band=$bands --n_cores=20 --n_bins_T=20 --inner --name='inner'  >& batch_scripts/logs/log_dtdm_bin_qsos_inner.log

#------------------------------------------------------------------------------
# Extract and save features from light curves
#------------------------------------------------------------------------------
# python scripts/generate_features.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_generate_features_qsos.log
# python scripts/generate_features.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_generate_features_calibStars.log