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
parser.add_argument('-m', '--maxEvents', default=100000,
                    help='Max amount of events to occur')

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


# event.hits returns array of length 4, where the it's x, y, z and then some weird number.
# persumably the weird number might be the energy detected at that very moment, however I am not sure. Then this would make


# shower_prifle the method takes in the event hits, does it for all the events, takes in alpha and beta has to be fitted first,
# used default values (has to be trained), shower_profile energy

# inputs for these models would be, trainingLoop?

i = 0
for event in event_list:
    while i < 1:
        # print(list(event.hits))
        i += 1
    # print(event)
    x_vals.extend([hit[0] for hit in event.hits])
    y_vals.extend([hit[1] for hit in event.hits])
    z_vals.extend([hit[2] for hit in event.hits])


# Now, we just need to

#x_vals = [hit[0] for hit in event.hits for event in event_list]
x_vals_max, x_vals_min = max(x_vals) + 5, min(x_vals) - 5
#y_vals = [hit[1] for hit in event.hits for event in event_list]
y_vals_max, y_vals_min = max(y_vals) + 5, min(y_vals) - 5
#z_vals = [hit[2] for hit in event.hits for event in event_list]
z_vals_max, z_vals_min = max(z_vals) + 5, min(z_vals) - 5

# define x0 for tracker and calorimeter - units in cm
tracker_x0 = 0.1 * 1.86  # 90% is vacuum
calorimeter_x0 = 9.37

# Find 't' for radiation depth - NOTE: NEEDS TO BE UNIT CHECKED

# Create bins

# update? originally 0.5 mm width for z but...
x_step, y_step, z_step = 1, 1, 1
# [range(i, i+z_step, .1) for i in range(z_vals_min, z_vals_max, z_step)]
z_range = list(np.arange(z_vals_min, z_vals_max, z_step))
# [range(i, i+x_step, .1) for i in range(x_vals_min, x_vals_max, x_step)]
x_range = list(np.arange(x_vals_min, x_vals_max, x_step))
# [range(i, i+y_step, .1) for i in range(y_vals_min, y_vals_max, y_step)]
y_range = list(np.arange(y_vals_min, y_vals_max, y_step))


# useless value?
min_length = len(z_range)


coordinate_ranges = list(map(list, zip(x_range, y_range, z_range)))
bin_names = range(0, len(coordinate_ranges))
geometry = list(map(list, zip(bin_names, coordinate_ranges)))


def in_range(val, range_start, step):
    if range_start >= val and val < step + range_start:
        return True
    else:
        return False


def bin_find(hit, geometry):
    ''' Finds bin name for a given hit and given geometry.'''
    try:
        # print('xd')
        return hit[5]
    except IndexError:
        x, y, z = hit[0], hit[1], hit[2]
        for coords in geometry:
            # print(coords)
            # round(x, 1) in coords[1][0]
            x_right = in_range(x, coords[1][0], x_step)
            # round(y, 1) in coords[1][1]
            y_right = in_range(y, coords[1][1], y_step)
            # round(z, 1) in coords[1][2]
            z_right = in_range(z, coords[1][2], z_step)
            if x_right and y_right and z_right:
                return coords[0]


for event in event_list:
    bins = []
    # print(event)
    for hit in event.hits:
        # print(type(hit))
        # print(hit)
        bins.append(bin_find(hit, geometry))

        # print(event.hits.shape)
    event.hits = np.append(event.hits, np.array(bins).reshape(-1, 1), 1)


def calculate_t():
    return None


def shower_profile():
    calculate_t()
    # print(1)
