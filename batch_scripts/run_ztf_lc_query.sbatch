#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=qso_avg
#SBATCH --mem=42G
#SBATCH -t 99:00:00
#SBATCH --mail-user=harry.rendell@ed.ac.uk

# Note, we can run this as a normal batch script using bash <filename>

# Load anaconda. Omit this if not running on cuillin
# source ~/.bashrc_old

python scripts/ztf_lc_query.py --object='qsos'       --n_workers=16 > batch_scripts/logs/log_ztf_lc_query.log
python scripts/ztf_lc_query.py --object='calibStars' --n_workers=16 > batch_scripts/logs/log_ztf_lc_query.log
