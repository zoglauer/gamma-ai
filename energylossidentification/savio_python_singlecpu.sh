#!/bin/bash

# Remember:
# Submit via: sbatch ...
# Check if running via: sqs -u zoglauer

#SBATCH -J 511Response

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2_htc
#SBATCH --qos=savio_debug

#SBATCH -t 00:10:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --signal=2@60

#SBATCH --mail-user=zog@ssl.berkeley.edu
#SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

echo "Loading modules..."
module load gcc/4.8.5 cmake python/3.6 cuda tensorflow


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 run.py -f Ling.seq3.quality.root -o Results -a TMVA:BDT

wait
