#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Python

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal

#SBATCH -t 02:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

#SBATCH --signal=2@60

##SBATCH --mail-user=ruoxi.shang@berkeley.edu
##SBATCH --mail-type=ALL


echo "Starting submit on host ${HOSTNAME}..."

echo "Loading modules..."
module load gcc/4.8.5 cmake python/3.6 cuda tensorflow


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -u run.py -f 1MeV_50MeV_flat.p1.inc18166611.id1.sim.gz



wait
