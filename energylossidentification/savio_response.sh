#!/bin/bash

# Remember:
# Submit via: sbatch ...

#SBATCH -J Ling_Response

#SBATCH --account=fc_cosi
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal

#SBATCH -t 00:10:00

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

#SBATCH --signal=2@60

##SBATCH --mail-user=
##SBATCH --mail-type=ALL


echo "Starting submit on host ${HOST}..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mresponsecreator -m qf -g ../detectormodel/COSILike.geo.setup -c Ling.revan.cfg -r Ling -f Ling.p1.inc*.id1.sim.gz

wait
