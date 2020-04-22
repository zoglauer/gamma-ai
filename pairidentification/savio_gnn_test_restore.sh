#!/bin/bash

#SBATCH -J Python

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2_gpu
#SBATCH --qos=savio_normal

#SBATCH -t 00:20:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

#SBATCH --signal=2@60

# --> CHANGE TO YOUR EMAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danielshin@berkeley.edu

python3 -u PairIdentificationGNN.py --maxevents 2000 -b 200 --n_iters 10 --restore "saved_model_state.pt"

