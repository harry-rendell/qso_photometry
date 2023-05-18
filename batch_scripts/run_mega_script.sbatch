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
n_cores=32
surveys='sdss ps ztf supercosmos'
bands='gri'

#------------------------------------------------------------------------------
# Save grouped data (using uncleaned photometry)
#------------------------------------------------------------------------------
python scripts/save_grouped.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores > batch_scripts/logs/log_save_grouped_qsos.txt
python scripts/save_grouped.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores > batch_scripts/logs/log_save_grouped_calibStars.txt


#------------------------------------------------------------------------------
# Average observations in the same night
#------------------------------------------------------------------------------
python scripts/average_nightly_observations.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_average_nightly_observations_qsos.txt
python scripts/average_nightly_observations.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores >& batch_scripts/logs/log_average_nightly_observations_calibStars.txt


#------------------------------------------------------------------------------
# Save grouped data (using cleaned photometry)
#------------------------------------------------------------------------------
python scripts/save_grouped.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores --cleaned > batch_scripts/logs/log_save_grouped_clean_qsos.txt
python scripts/save_grouped.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores --cleaned > batch_scripts/logs/log_save_grouped_clean_calibStars.txt

#------------------------------------------------------------------------------
# Merge survey data and save to to data/merged/clean/
#------------------------------------------------------------------------------
python scripts/merge_survey_data.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores > batch_scripts/logs/log_merge_survey_data_qsos.txt
python scripts/merge_survey_data.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores > batch_scripts/logs/log_merge_survey_data_calibStars.txt

#------------------------------------------------------------------------------
# Save grouped data (using cleaned, merged photometry) and save to data/merged/clean/
#------------------------------------------------------------------------------
python scripts/save_grouped_merged.py --object='qsos'       --band=$bands --n_cores=$n_cores > batch_scripts/logs/log_save_grouped_merged_qsos.txt
python scripts/save_grouped_merged.py --object='calibStars' --band=$bands --n_cores=$n_cores > batch_scripts/logs/log_save_grouped_merged_calibStars.txt

#------------------------------------------------------------------------------
# Save features
#------------------------------------------------------------------------------
python scripts/generate_features.py --object='qsos'       --band=$bands --survey=$surveys --n_cores=$n_cores > batch_scripts/logs/log_generate_features_qsos.txt
python scripts/generate_features.py --object='calibStars' --band=$bands --survey=$surveys --n_cores=$n_cores > batch_scripts/logs/log_generate_features_calibStars.txt