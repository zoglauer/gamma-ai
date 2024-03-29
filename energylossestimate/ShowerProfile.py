### Read in data, argparse, etc
import pickle
import argparse
import os
import sys
from math import exp
import scipy
from scipy import optimize, special, spatial
import numpy as np
from EventData import EventData
import time

start_time = time.time()

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

# Define geometry of system (cm)
# MAYBE should be made into a class in event_data that can be imported
si_x_min, si_x_max = -45.8, 45.8
si_y_min, si_y_max = -48.3, 48.3
si_z_min, si_z_max = 10.2, 45
# OVERALL max/min?

x_vals = []
y_vals = []
z_vals = []

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
min_length = len(z_range)
# [range(i, i+x_step, .1) for i in range(x_vals_min, x_vals_max, x_step)]
x_range = list(np.arange(x_vals_min, x_vals_max, x_step))
# [range(i, i+y_step, .1) for i in range(y_vals_min, y_vals_max, y_step)]
y_range = list(np.arange(y_vals_min, y_vals_max, y_step))

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
    try:
        return hit[5]
    except IndexError:
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

# Find energy in each bin and calculate t accordingly


def t_calculate(hits, geometry):
    '''Finds t for each bin based off list of hits, then attaches t to appropriate hits.

    Inputs: list of hits for an event, geometry.
    Output: updated list of hits that now contains appropriate t for each hit.
    Method:
    - Generate bins, a dict where
     - key: sequential numbers based off of geometry
     - value: [measured_inside, t]
    - Find t for each bin
    - Go through original hit list and match hits to the appropriate bin and therefore appropriate t

    NEEDS refactoring
    '''
    # make a new class with additional info - x,y,z(vector), energy, t-val, geometry, boundaries
    #bins = dict(zip(len(geometry), [[0] for i in range(0, len(geometry))]))
    bins = {}
    keys = [geometry[i][0] for i in range(0, len(geometry))]
    values = [geometry[i][1] for i in range(0, len(geometry))]
    for i in range(0, len(geometry)):
        bins[keys[i]] = values[i]
    print(bins)

    for hit in hits:
        bins[hit[5]] += hit[4]
    for column in bins:
        t = bins[column] / (len(keys) * len(values) * len(z_vals) * tracker_x0)
        # t = bins[column] / (area * zbin_height * x0)
        bins[column].append(t)
    # match bin t back to hits
    for hit in hits:
        bin_name = bin_find(hit, geometry)
        t = bins[bin_name][1]
        hit[6] = t
    return hits
# l = 7
# l = 4
# 5 = 2
# 4 = 1
# 6 = 3
# how many values are supposed to be in hits?
# Store t for each hit in EventData


for event in event_list:
    event.hits = t_calculate(event.hits, geometry)

# Fit for alpha, beta


def shower_profile(event_hits, alpha, beta):
    '''Returns estimated gamma_energy based on inverse of shower profile eqn.

    Takes in:
    - event_hits, a list of hits associated with that event
      - t is calculated from xyz coordinates of event hits and geometry
    - alpha, a fit parameter calculated across events
    - beta, a fit parameter calculated across events
    Returns:
    - predicted 'true' gamma energy

    Method:
    - sum across hits(measured[hit n] / [alpha/beta/[t for hit]]) = gamma [event]
    '''
    gamma = special.gamma(alpha)
    gamma_for_hits = []
    for hit in event_hits:
        hit[4] = t
        hit[3] = measured_energy
        t_beta_alpha = (beta * t)**(alpha - 1) * beta * np.exp(-1 * beta * t)
        gamma_for_hits.append((measured_energy * gamma) / t_beta_alpha)
    return sum(gamma_for_hits)

# Create list of measured/t/true based on events


gamma_energies = [event.gamma_energy for event in event_list]
event_hits = [event.hits for event in event_list]

# TODO: set random seed an maybe pull from uniform dist. --> iterate over time to find best initial guess.
initial_guesses = .5, .5, 1
maxfev = 10000


def shower_profile_fit(f, event_hits, gamma_energies, initial_guesses, maxfev):
    '''Find alpha and beta for shower_profile().

    Takes in:
    - f, should be shower_profile.
    - event_hits, which contains by proxy measured energy and t
    - gamma_energies, a list of true energies
    - initial_guesses, best guess of what parameters are
    - maxfev, ?
    '''
    results, variance = optimize.curve_fit(
        f, (event_hits), gamma_energies, initial_guesses, maxfev=maxfev)
    alpha = results[0]
    beta = results[1]
    return alpha, beta, variance

# Call shower_profile_fit and save alpha/beta, variance


alpha, beta, variance = shower_profile_fit(
    shower_profile, event_hits, gamma_energies, initial_guesses, maxfev)
EventData.alpha = alpha
EventData.beta = beta
# dictionary for alpha/beta with bin [i.e. for the two different x0]?
# call shower_profile_fit twice for different alpha/beta for different regions?

# Add predicted shower energy to event data instances


def error(event):
    '''Returns error in shower_energy prediction for given event.'''
    return np.abs(event.gamma - event.shower_energy)/event.gamma  # percentage diff. Not pure abs.


errors = []

for event in event_list:
    event.shower_energy = shower_profile(event.hits, alpha, beta)
    errors.append(error(event))

avg_err = sum(errors)/len(event_list)
print("average MSE error:", avg_err)
print("fitted variance:", variance)

# save data/end program

print("--------------------------")
print(f"Added shower profile's predicted energy to {len(event_list)} events.")
print("Info: storing updated data.")

''' TODO FIX ARGS --> it errors because of restricted access to data sim file.
if args.savefileto != "":
    save_file = args.savefileto
if not os.path.exists(save_file):
    print(f"The savefile does not exist: {save_file}. Creating new...")
    with open(save_file, 'w') as fp:
        pass # write nothing.


with open(save_file, "wb") as file_handle:
    pickle.dump(event_list, file_handle)
print("Info: done.")
'''

end_time = time.time()
print('Total time elapsed:', end_time - start_time, 's.')


class Hit_Info:
    # make a new class with additional info - x,y,z(vector), energy, t-val, geometry, boundaries
    hits = [x_vals, y_vals, z_vals]
    boundaries = geometry
    t_val = t_calculate(hits, boundaries)
    gamma_energies
    event_hits


sys.exit(0)
