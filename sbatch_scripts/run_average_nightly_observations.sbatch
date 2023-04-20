#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=qso_avg
#SBATCH --mem=42G
#SBATCH -t 99:00:00

# The work directory
cd /home/hrb/qso_photometry/python

# Run the executable
module load anaconda
source activate astroconda
# lower upper width step
# python scripts/average_nightly_observations.py --object='qsos' --band='r' --survey='ps'   --n_cores=16  >& sbatch_scripts/log_average_nightly_observations_ps_r.txt
python scripts/average_nightly_observations.py --object='qsos' --band='r' --survey='ztf'  --n_cores=16 --dry_run  >& sbatch_scripts/log_average_nightly_observations_ztf_r.txt

# python scripts/average_nightly_observations.py --object='qsos' --band='g' --survey='ps'   --n_cores=16  >& sbatch_scripts/log_average_nightly_observations_ps_g.txt
# python scripts/average_nightly_observations.py --object='qsos' --band='g' --survey='ztf'  --n_cores=16  >& sbatch_scripts/log_average_nightly_observations_ztf_g.txt

# python scripts/average_nightly_observations.py --object='qsos' --band='i' --survey='ps'   --n_cores=16  >& sbatch_scripts/log_average_nightly_observations_ps_i.txt
# python scripts/average_nightly_observations.py --object='qsos' --band='i' --survey='ztf'  --n_cores=16  >& sbatch_scripts/log_average_nightly_observations_ztf_i.txt

# python scripts/average_nightly_observations.py --object='qsos' --band='r' --survey='ztf'  --n_cores=4 --n_rows=100000 --dry_run  >& sbatch_scripts/log_average_nightly_observations_ztf_r.txt