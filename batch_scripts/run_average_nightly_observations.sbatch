#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=qso_avg
#SBATCH --mem=40G
#SBATCH -t 99:00:00
#SBATCH --mail-user=harry.rendell@ed.ac.uk

# Note, we can run this as a normal batch script using bash <filename>

# Load anaconda. Omit this if not running on cuillin
# source ~/.bashrc_old

# use same number of cores as cpus-per-task above
n_cores=4
python scripts/average_nightly_observations.py --object='qsos'       --band='gri' --survey='supercosmos' --n_cores=$n_cores --dry_run >& batch_scripts/logs/log_average_nightly_observations_qsos.log
python scripts/average_nightly_observations.py --object='calibStars' --band='gri' --survey='supercosmos' --n_cores=$n_cores --dry_run >& batch_scripts/logs/log_average_nightly_observations_calibStars.log