#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=qso_avg
#SBATCH --mem=40G
#SBATCH -t 99:00:00

# Note, we can run this as a normal batch script using bash <filename>

# The work directory
cd /home/hrb/qso_photometry/python

# Load anaconda. Omit this if not running on cuillin
module load anaconda
source activate astroconda

# use same number of cores as cpus-per-task above
n_cores=16

# test 
# python scripts/average_nightly_observations.py --object='qsos' --band='r' --survey='ztf'  --n_cores=4 --n_rows=10000 --dry_run >& batch_scripts/logs/log_average_nightly_observations_test.txt

python scripts/average_nightly_observations.py --object='qsos' --band='gri' --survey='sdss ps ztf' --n_cores=$n_cores >& batch_scripts/logs/log_average_nightly_observations.txt