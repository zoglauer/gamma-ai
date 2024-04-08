#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Sim

#SBATCH --account=fc_cosi
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_debug

#SBATCH --chdir=/global/scratch/users/zoglauer/Sims/EnergyLossEstimate

# This should give us 1,000,000 events 
#SBATCH --time=00:10:00

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --signal=2@60

##SBATCH --mail-user=XYZ@berkeley.edu
# Copy above line below with your email instead of XYZ, and uncomment (### -> #)
#SBATCH --mail-user=ethan.chen@berkeley.edu

#SBATCH --mail-type=ALL

echo "Starting submit on host ${HOST}..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

. /global/home/groups/fc_cosi/MEGAlib/bin/source-megalib.sh

mcosima -z -w -t `nproc` /global/scratch/users/zoglauer/MachineLearning/energylossestimate/Sim_2MeV_5GeV_flat.source 

wait