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
python scripts/save_grouped.py --object='qsos' --band='gri' --survey='sdss ps ztf'  --n_cores=$n_cores --cleaned > batch_scripts/logs/log_save_grouped.txt