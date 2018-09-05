#!/bin/bash

# Remember:
# Submit via: sbatch ...
# Check if running via: sqs -u zoglauer

#SBATCH -J 511Response

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2
#SBATCH --qos=savio_debug

#SBATCH -t 00:01:30

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

#SBATCH --signal=2@60

#SBATCH --mail-user=zog@ssl.berkeley.edu
#SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

echo "Loading modules..."
module load gcc/4.8.5 cmake python/3.6 cuda tensorflow

echo "Sourcing MEGAlib..."
source ${HOME}/Software/MEGAlib/bin/source-megalib.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mcosima -z -w -p 1 -r -t ${OMP_NUM_THREADS} Ling.source
#cosima -z Ling.source

wait
