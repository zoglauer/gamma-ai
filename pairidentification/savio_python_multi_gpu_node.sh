#!/bin/bash

# Remember:
# Submit via: sbatch ...
# Check via: squeue -u $USER

#SBATCH -J Python

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

#SBATCH -t 72:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --signal=2@60

# --> CHANGE TO YOUR EMAIL

#SBATCH --mail-user=harrisoncostantino@berkeley.edu



#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting analysis on host ${HOSTNAME} with job ID ${SLURM_JOB_ID}..."

echo "Loading modules..."
module purge
module load gcc/4.8.5 cmake python/3.6 tensorflow/1.12.0-py36-pip-gpu blas

echo "Starting execution..."

# --> ADAPT THE FILENAME
python3 -u PairIdentification.py -f PairIdentification.p1.sim.gz -m 30000


echo "Waiting for all processes to end..."
wait

