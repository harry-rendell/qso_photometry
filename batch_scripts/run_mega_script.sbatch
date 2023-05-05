#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=qso_sg
#SBATCH --mem=42G
#SBATCH -t 99:00:00
#SBATCH --mail-user=harry.rendell@ed.ac.uk

# Note, we can run this as a normal batch script using bash <filename>

# The work directory
cd /home/hrb/project/qso_photometry

# Load anaconda. Omit this if not running on cuillinmodule load anaconda
module load anaconda
source activate astroconda

n_cores=16

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
python scripts/save_grouped.py --object='qsos'       --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores > batch_scripts/logs/log_save_grouped_qsos.txt
python scripts/save_grouped.py --object='calibStars' --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores > batch_scripts/logs/log_save_grouped_calibStars.txt


#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
python scripts/average_nightly_observations.py --object='qsos'       --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores >& batch_scripts/logs/log_average_nightly_observations_qsos.txt
python scripts/average_nightly_observations.py --object='calibStars' --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores >& batch_scripts/logs/log_average_nightly_observations_calibStars.txt


#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
python scripts/save_grouped.py --object='qsos'       --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores --cleaned > batch_scripts/logs/log_save_grouped_clean_qsos.txt
python scripts/save_grouped.py --object='calibStars' --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores --cleaned > batch_scripts/logs/log_save_grouped_clean_calibStars.txt

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
python scripts/merge_survey_data.py --object='qsos'       --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores > batch_scripts/logs/log_merge_survey_data_qsos.txt
python scripts/merge_survey_data.py --object='calibStars' --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores > batch_scripts/logs/log_merge_survey_data_calibStars.txt

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
python scripts/save_grouped_merged.py --object='qsos'       --band='gri' --n_cores=$n_cores > batch_scripts/logs/log_save_grouped_merged_qsos.txt
python scripts/save_grouped_merged.py --object='calibStars' --band='gri' --n_cores=$n_cores > batch_scripts/logs/log_save_grouped_merged_calibStars.txt

#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
python scripts/generate_features.py --object='qsos'       --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores > batch_scripts/logs/log_generate_features_qsos.txt
python scripts/generate_features.py --object='calibStars' --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores > batch_scripts/logs/log_generate_features_calibStars.txt