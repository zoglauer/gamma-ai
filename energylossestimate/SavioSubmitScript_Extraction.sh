#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Sim

#SBATCH --account=fc_cosi
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_normal

#SBATCH --chdir=/global/scratch/users/zoglauer/Sims/EnergyLossEstimate

#SBATCH -t 72:00:00

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --signal=2@60

##SBATCH --mail-user=XYZ@berkeley.edu
##SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

. /global/home/groups/fc_cosi/MEGAlib/bin/source-megalib.sh

echo "Starting execution..."

# --> ADAPT THE FILENAME
python3 -u /global/scratch/users/zoglauer/MachineLearning/energylossestimate/event_extractor.py -m 1000 -f 2MeV_5GeV_flat.p1.sim.gz
mv 2MeV_5GeV_flat.p1.data EnergyLossEstimate.1k.data 
python3 -u /global/scratch/users/zoglauer/MachineLearning/energylossestimate/event_extractor.py -m 10000 -f 2MeV_5GeV_flat.p1.sim.gz
mv 2MeV_5GeV_flat.p1.data EnergyLossEstimate.10k.data
python3 -u /global/scratch/users/zoglauer/MachineLearning/energylossestimate/event_extractor.py -m 100000 -f 2MeV_5GeV_flat.p1.sim.gz
mv 2MeV_5GeV_flat.p1.data EnergyLossEstimate.100k.data
python3 -u /global/scratch/users/zoglauer/MachineLearning/energylossestimate/event_extractor.py -m 10000000 -f 2MeV_5GeV_flat.p1.sim.gz
mv 2MeV_5GeV_flat.p1.data EnergyLossEstimate.all.data

echo "Waiting for all processes to end..."
wait



wait
