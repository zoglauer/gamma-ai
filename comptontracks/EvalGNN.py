import os
import sys
import signal
import argparse

'''
Runs a script to evaluate GNN at number of events starting from 
-l flag to -m flag parameters. The rate is increased exponentially, x2 events of previous iteration each iteration.
All results are stored in the Results folder. -f flag to specify sim file (if sim.gz files are not in current dir).
'''

# FLAGS:
# -l for starting number of events.
# -m for ending number of events, use 0 if you only want to run on the -l number of events and none more.
# -r multiplication rate, i.e. runs on -l events, then -l*-r events, then -l*-r*-r events, etc. until exceeding -m.
# -f simulation file path if not stored in current dir.

# Take care of Ctrl-C
Interrupted = False
NInterrupts = 0
def signal_handler(signal, frame):
  global Interrupted
  Interrupted = True
  global NInterrupts
  NInterrupts += 1
  if NInterrupts >= 2:
    print("Aborting!")
    sys.exit(0)
  print("You pressed Ctrl+C - waiting for graceful abort, or press  Ctrl-C again, for quick exit.")
signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser(description='Evaluate GNN separately on gamma and electron tracks on log scale.')
parser.add_argument('-l', '--min', default='5000', help='Starting num of events.')
parser.add_argument('-m', '--max', default='10000000', help='Ending num of events.')
parser.add_argument('-r', '--rate', default='2', help='Multiplication rate.')
parser.add_argument('-f', '--file', default='ComptonTrackIdentification_LowEnergy.p1.sim.gz', help='Simulation filepath.')


args = parser.parse_args()

low = int(args.min)
high = int(args.max)
rate = int(args.rate)
file = args.file
split = 0.1

print("\n=================== \nGNN Evaluation Script\n===================")


while True:
    if low >= 1000000:
        split = 0.05
    elif low >= 10000000:
        split = 0.01
    print("\nEvaluating gamma tracks on {} events.".format(low))
    os.system("python3 -u ComptonTrackIdentificationGNN.py -f {} -a g -m {} -s {}".format(file, low, split))
    print("\nEvaluating electron tracks on {} events.".format(low))
    os.system("python3 -u ComptonTrackIdentificationGNN.py -f {} -a e -m {} -s {}".format(file, low, split))
    if low > high:
        break
    low *= rate

exit()

