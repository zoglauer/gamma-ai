echo "Starting execution..."

# --> ADAPT THE FILENAME
python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a g -m 200000

python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a e -m 200000


python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a g -m 400000

python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a e -m 400000


python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a g -m 800000

python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a e -m 800000


python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a g -m 1600000

python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a e -m 1600000


python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a g -m 3200000

python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a e -m 3200000


python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a g -m 6400000

python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a e -m 6400000


python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a g -m 10000000

python3 -u ComptonTrackIdentificationGNN.py -f ComptonTrackIdentification_LowEnergy.p1.sim.gz -a g -m 10000000

echo "Waiting for all processes to end..."
wait
