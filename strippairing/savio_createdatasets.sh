#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Sim

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal

#SBATCH -t 70:00:00

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

#SBATCH --signal=2@60

##SBATCH --mail-user=
##SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ID=$RANDOM
mcosima -z -w -p ${ID} -r -t ${OMP_NUM_THREADS} StripPairing.source
mresponsecreator -m sf -g ../detectormodel/COSILike.geo.setup -c StripPairing.revan.cfg -r StripPairing -f StripPairing.p${ID}.inc*.id1.sim.gz
responsemanipulator -j StripPairing.p${ID}

wait
