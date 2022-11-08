### Read in data, argparse, etc
from fileinput import filename
import pickle
import argparse
import os
import sys
from math import exp
import scipy
from scipy import optimize, special, spatial
import numpy as np
from event_data import EventData
import time
# import EventData as event
# what is passed in is the hits, and the
# what is returned should be the estimate of

start_time = time.time()

parser = argparse.ArgumentParser(
    description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz',
                    help='File name used for training/testing')
parser.add_argument('-s', '--savefileto', default='shower_output/shower_events.pkl',
                    help='save file name for event data with shower profile estimates.')

args = parser.parse_args()
if (args.filename == ""):
    print('xd')
else:
    print(args.filename)

if args.filename != "":
    file_name = args.filename
if not os.path.exists(file_name):
    print(f"Error: The training data file does not exist: {file_name}")
    sys.exit(0)
print(f"CMD: Using file {file_name}")

with open(file_name, "rb") as file_handle:
    event_list = pickle.load(file_handle)

event_list = event_list[0:2]
# Define geometry of system (cm)
# MAYBE should be made into a class in event_data that can be imported
si_x_min, si_x_max = -45.8, 45.8
si_y_min, si_y_max = -48.3, 48.3
si_z_min, si_z_max = 10.2, 45
# OVERALL max/min?

x_vals = []
y_vals = []
z_vals = []


# t_calculate provides the

# hit[0] = x cordinate
# hit[1] = y cordinate
# hit[2] = z cordinate
# hit[3] = ?
for event in event_list:
    x_vals.extend([hit[0] for hit in event.hits])
    y_vals.extend([hit[1] for hit in event.hits])
    z_vals.extend([hit[2] for hit in event.hits])


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

# Sort hits into bins


def in_range(val, range_start, step):
    if range_start >= val and val < step + range_start:
        return True
    else:
        return False


def bin_find(hit, geometry):
    ''' Finds bin name for a given hit and given geometry.'''
    x, y, z = hit[0], hit[1], hit[2]
    for coords in geometry:
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
    for hit in event.hits:
        # print(type(hit))
        bins.append(bin_find(hit, geometry))

        # print(event.hits.shape)
    event.hits = np.append(event.hits, np.array(bins).reshape(-1, 1), 1)

# add bins to hits as column 5
