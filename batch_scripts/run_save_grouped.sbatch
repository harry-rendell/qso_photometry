#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=qso_avg
#SBATCH --mem=42G
#SBATCH -t 99:00:00

# Note, we can run this as a normal batch script using bash <filename>

# The work directory
cd /home/hrb/qso_photometry/python

# Load anaconda. Omit this if not running on cuillinmodule load anaconda
source activate astroconda

python scripts/save_grouped.py --object='qsos' --band='r' --survey='ztf' --n_cores=4 --n_rows=100000 --dry_run --cleaned > sbatch_scripts/log_save_grouped.txt