###read in data, argparse, etc
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

start_time = time.time()

parser = argparse.ArgumentParser(description=
        'Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz',
        help='File name used for training/testing')
parser.add_argument('-s', '--savefileto', default='shower_output/shower_events.pkl',
                    help='save file name for event data with shower profile estimates.')

args = parser.parse_args()

# PARAMS

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
#OVERALL max/min?
x_vals = [event.hits[0] for event in event_list]
x_val_max, x_val_min = max(x_vals) + 5, min(x_vals) - 5
y_vals = [event.hits[1] for event in event_list]
y_val_max, y_val_min = max(y_vals) + 5, min(y_vals) - 5
z_vals = [event.hits[2] for event in event_list]
z_val_max, z_val_min = max(z_vals) + 5, min(z_vals) - 5

#define x0 for tracker and calorimeter - units in cm
tracker_x0 = 0.1 * 1.86 #90% is vacuum
calorimeter_x0 = 9.37

### Find 't' for radiation depth - NOTE: NEEDS TO BE UNIT CHECKED

# Create bins

x_step, y_step, z_step = 1, 1, 1 #update? originally 0.5 mm width for z but...
z_range = range(z_vals_min, z_vals_max, z_step)
min_length = len(z_range)
x_range = #
z_range = #

coordinate_ranges = zip(x_range, y_range, z_range) #needs to end up with each tuple being x, y, z *range*
bin_names = range(0, len(coordinate_ranges))
geometry = zip(bin_names, coordinate_ranges)

# Sort hits into bins

def bin_find(hit, geometry):
    ''' Finds bin name for a given hit and given geometry.'''
    try:
        return hit[5]
    except IndexError:
        x, y, z = hit[0], hit[1], hit[2]
        for coords in geometry:
            x_right = round(x) in coords[1][0]
            y_right = round(y) in coords[1][1]
            z_right = round(z) in coords[1][2]
            if x_right and y_right and z_right:
                return coords[0]

for event in event_list:
    for hit in event.hits:
        hit[5] = bin_find(hit, geometry)

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
    bins = dict(zip(len(geometry), [[0] for i in range(0, len(geometry))]))
    for hit in hits:
        bins[hit[5]] += hit[4]
    for column in bins:
        t = bins[column] / (area * zbin_height * x0) 
        bins[column].append(t)
    # each t is now for each x y z bin
    # so we know from each hit, their xyz coords,
    # so we take the xzy bin t and apply it to each event hit
    for hit in hits:
        bin_name = bin_find(hit, geometry)
        t = bins[bin_name][1]
        hit[6] = t
    return hits

# Store t for each hit in EventData

for event in event_list:
    event.hits = t_calculate(event.hits, geometry)

### Fit for alpha, beta

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
    results, variance = optimize.curve_fit(f, (event_hits), gamma_energies, initial_guesses, maxfev=maxfev)
    alpha = results[0]
    beta = results[1]
    return alpha, beta, variance

# Call shower_profile_fit and save alpha/beta, variance

alpha, beta, variance = shower_profile_fit(shower_profile, event_hits, gamma_energies, initial_guesses, maxfev)
EventData.alpha = alpha
EventData.beta = beta
# dictionary for alpha/beta with bin [i.e. for the two different x0]?
# call shower_profile_fit twice for different alpha/beta for different regions?

### Add predicted shower energy to event data instances

def error(event):
    '''Returns error in shower_energy prediction for given event.'''
    return event.gamma - event.shower_energy

errors = []

for event in event_list:
    event.shower_energy = shower_profile(event.hits, alpha, beta)
    errors.append(error(event))

avg_err = sum(errs)/len(event_list)
print("average MSE error:", avg_err)
print("fitted variance:", variance)

### save data/end program

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

sys.exit(0)
