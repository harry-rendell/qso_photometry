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

python scripts/ztf_lc_query.py --object='qsos' --n_cores=4 --n_requests=1500  > batch_scripts/logs/log_ztf_lc_query.txt
