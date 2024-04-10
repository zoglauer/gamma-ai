#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Sim

#SBATCH --account=fc_cosi
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal

#SBATCH --chdir=/global/scratch/users/zoglauer/Sims/EnergyLossEstimate

#SBATCH -t 00:10:00

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --signal=2@60

#SBATCH --mail-user=ethan.chen@berkeley.edu
#SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

. /global/home/groups/fc_cosi/MEGAlib/bin/source-megalib.sh

echo "Starting execution..."

# --> ADAPT THE FILENAME
python3 -u /global/scratch/users/zoglauer/MachineLearning/energylossestimate/event_extractor.py -m 1000 -f 2MeV_5GeV_flat.p1.sim.gz
mv 2MeV_5GeV_flat.p1.data EnergyLossEstimate.1k.data 

echo "Waiting for all processes to end..."
wait

wait