#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Sim

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal

#SBATCH -t 02:00:00

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

#SBATCH --signal=2@60

##SBATCH --mail-user=XYZ@berkeley.edu
##SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mcosima -z -w -t `nproc` PairIdentification.source

wait
