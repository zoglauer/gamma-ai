# Aditya's New Version of the ShowerProfile file.

import pickle
import argparse
import os
import sys
from math import exp
import scipy
from scipy import optimize, special, spatial
import numpy as np
# from energylossestimate.ShowerProfile import shower_profile
from event_data import EventData
import time


parser = argparse.ArgumentParser(
    description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz',
                    help='File name used for training/testing')
parser.add_argument('-s', '--savefileto', default='shower_output/shower_events.pkl',
                    help='save file name for event data with shower profile estimates.')

args = parser.parse_args()

if args.filename != "":
    file_name = args.filename
if not os.path.exists(file_name):
    print(f"Error: The training data file does not exist: {file_name}")
    sys.exit(0)
print(f"CMD: Using file {file_name}")

with open(file_name, "rb") as file_handle:
    event_list = pickle.load(file_handle)

x_vals = []
y_vals = []
z_vals = []


# hit[0] = x cordinate
# hit[1] = y cordinate
# hit[2] = z cordinate_
# hit[3]


# shower_prifle the method takes in the event hits, does it for all the events, takes in alpha and beta has to be fitted first,
# used default values (has to be trained), shower_profile energy

# inputs for these models would be, trainingLoop?

for event in event_list:
    print(event)
    x_vals.extend([hit[0] for hit in event.hits])
    y_vals.extend([hit[1] for hit in event.hits])
    z_vals.extend([hit[2] for hit in event.hits])


# Now, we just need to

def shower_profile():
    print(1)
