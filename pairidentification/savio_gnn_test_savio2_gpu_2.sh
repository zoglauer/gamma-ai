#!/bin/bash

#SBATCH -J Python

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

#SBATCH -t 01:00:00

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4

#SBATCH --signal=2@60


# --> CHANGE TO YOUR EMAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danielshin@berkeley.edu

python3 -u PairIdentificationGNN.py --maxevents 2000 -b 200 --n_iters 100

