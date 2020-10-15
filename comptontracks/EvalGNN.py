import os
import argparse

parser = argparse.ArgumentParser(description='Evaluate GNN separately on gamma and electron tracks on log scale.')
parser.add_argument('-l', '--low', default='5000', help='Starting num of events.')
parser.add_argument('-h', '--high', default='10000000', help='Ending num of events.')
parser.add_argument('-f', '--file', default='ComptonTrackIdentification_LowEnergy.p1.sim.gz', help='Simulation filepath.')

args = parser.parse_args()

low = int(args.low)
high = int(args.high)
file = args.file

print("=================== \nGNN Evaluation Script\n===================")

while True:
    print("Evaluating gamma tracks on {} events.".format(low))
    os.system("python3 -u ComptonTrackIdentificationGNN.py -f {} -a g -m {}".format(file, low))
    print("Evaluating electron tracks on {} events.".format(low))
    os.system("python3 -u ComptonTrackIdentificationGNN.py -f {} -a e -m {}".format(file, low))
    if low > high:
        break
    low *= 2

exit()

