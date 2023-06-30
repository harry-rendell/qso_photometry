#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=qso_sg
#SBATCH --mem=42G
#SBATCH -t 99:00:00
#SBATCH --mail-user=harry.rendell@ed.ac.uk

# Note, we can run this as a normal batch script using bash <filename>

# Load anaconda. Omit this if not running on cuillin
# source ~/.bashrc_old

# use same number of cores as cpus-per-task above
n_cores=38
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
#   NOTE: This appends instead of overwrites. Remove previously saved data if it is no longer needed.
#------------------------------------------------------------------------------
# python scripts/merge_survey_data.py --object='qsos'       --band=$bands --survey="$surveys" --n_chunks=$n_cores >& batch_scripts/logs/log_merge_survey_data_qsos.log
# python scripts/merge_survey_data.py --object='calibStars' --band=$bands --survey='supercosmos sdss ps ztf' --n_chunks=$n_cores >& batch_scripts/logs/log_merge_survey_data_calibStars.log

#------------------------------------------------------------------------------
# Save grouped data (using cleaned, merged photometry) and save to data/merged/clean/
#------------------------------------------------------------------------------
# python scripts/save_grouped_merged.py --object='qsos'       --band=$bands --n_cores=$n_cores >& batch_scripts/logs/log_save_grouped_merged_qsos.log
# python scripts/save_grouped_merged.py --object='calibStars' --band=$bands --n_cores=$n_cores >& batch_scripts/logs/log_save_grouped_merged_calibStars.log

#------------------------------------------------------------------------------
# Calculate dtdm
#------------------------------------------------------------------------------
python scripts/compute_dtdm_merged.py --object='qsos' --band='gi' --n_cores=$n_cores >& batch_scripts/logs/log_compute_dtdm_merged_qsos.log
# python scripts/compute_dtdm_merged.py --object='calibStars' --band='gi' --n_cores=$n_cores >& batch_scripts/logs/log_compute_dtdm_merged_calibStars.log

#------------------------------------------------------------------------------
# Save features
#------------------------------------------------------------------------------
# python scripts/generate_features.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_generate_features_qsos.log
# python scripts/generate_features.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_generate_features_calibStars.log
