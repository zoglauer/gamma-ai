#!/bin/bash

#SBATCH -J Python

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

#SBATCH -t 72:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

#SBATCH --signal=2@60

# --> CHANGE TO YOUR EMAIL
#SBATCH --mail-user=volkovskyi@berkeley.edu

module load pytorch/1.0.0-py36-cuda9.0
python3 -u PairIdentificationGNN.py --maxevents 20000 -b 32 --n_iters 100
