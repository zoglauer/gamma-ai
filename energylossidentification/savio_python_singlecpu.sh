#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Python

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2_htc
#SBATCH --qos=savio_normal

#SBATCH -t 00:10:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --signal=2@60

##SBATCH --mail-user=
##SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

echo "Loading modules..."
module load gcc/4.8.5 cmake python/3.6 cuda tensorflow


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 run.py -f Ling.seq3.quality.root -o Results -a TMVA:BDT

wait
