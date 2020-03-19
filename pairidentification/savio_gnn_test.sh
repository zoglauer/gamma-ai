#!/bin/bash

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

#SBATCH --mail-user= volkovskyi@berkeley.edu

python3 PairIdentificationGNN.py --maxevents 10000 -b 50 --n_iters 100
