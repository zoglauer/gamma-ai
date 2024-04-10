#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Sim

#SBATCH --account=fc_cosi
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_debug

#SBATCH --chdir=/global/scratch/users/zoglauer/Sims/EnergyLossEstimate

#SBATCH -t 00:10:00

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --signal=2@60

#SBATCH --mail-user=ethan.chen@berkeley.edu
#SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

wait