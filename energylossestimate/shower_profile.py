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

#define x0 for tracker and calorimeter - units in cm
tracker_x0 = 0.1 * 1.86 #90% is vacuum
calorimeter_x0 = 9.37

### func to find 't' for radiation depth - NOTE: NEEDS TO BE UNIT CHECKED

#create bins as follows: - NOTE: MAY WISH TO MORE FINELY SLICE X/Y
# z dimension: 0 to overall z dimension of detector, width of 0.5mm
# x dimension: 0 to overall x dimension of detector, 'widths' being:
# --> < silicon start, > silicon start and < silicon end, > silicon end
# y dimension: 0 to overall y dimension of detector, 'widths' being:
# --> < silicon start, > silicon start and < silicon end, > silicon end
# each bin should have:
# - (x, y, and z range)
# - energy_inside init. to 0
# - 't' init. to 0
# data structure - hashmap, where the key includes material/location
# function as value?
#dictionary: stores different trained parameters
# sort into xyz bins by numbering bins and adding that number to the
# hit list and then just do dictionary[bin number] += hit_energy

#go through passed in list of hits and:
# 1. find bin in which the hit goes based on (x,y,z)
# 2. add energy for that hit to the energy_inside for the bin

def t_calculate(hits, geometry):
    bins = {}
    #sorts the hits into bins that hold energy in each bin based on geometry
    for column in bins:
        t = bins[column] / (area * zbin_height * x0) 
        bins[i].append(t)
    # each t is now for each x y z bin
    # so we know from each hit, their xyz coords,
    # so we take the xzy bin t and apply it to each event hit
    for hit in hits:
        hits[hit, 4] = t
    return t#or hits??

### Fit for alpha, beta

def shower_profile(event_inputs, alpha, beta):
    '''Returns estimated gamma_energy based on inverse of shower profile eqn.

    Takes in:
    - event_inputs, a tuple of data from EventData: (measured_energy, t)
      - t is calculated from xyz coordinates of event hits and geometry
    - alpha, a fit parameter calculated across events
    - beta, a fit parameter calculated across events
    Returns:
    - predicted 'true' gamma energy
    '''
    measured_energy, t = event_inputs
    gamma = special.gamma(alpha)
    t_beta_alpha = (beta * t)**(alpha - 1) * beta * np.exp(-1 * beta * t)
    gamma_energy = (measured_energy * gamma) / t_beta_alpha
    return gamma_energy

#create list of measured/t/true based on events
#measured_energies = [event.measured_energy for event in event_list]
#gamma_energies = [event.gamma_energy for event in event_list]

#gamma[event1] = measured[hit 1] * some weight + measured[hit2] * some weight ...
# where some weight related to alpha beta and t somehow?
# some weight = alpha-beta part of the shower profile function which includes t.
# measured[for each hit] / [alpha/beta/[t for event]] = gamma energy[event 1]
# sum across hits(measured[hit n] / [alpha/beta/[t for hit]]) = gamma energy[event]

# store t for each hit in EventData, store shower_energy

def shower_profile_fit(f, measured_energies, event_ts, gamma_energies):
    '''Find alpha and beta for shower_profile().

    Takes in:
    - f, should be shower_profile.
    - 
    '''
    # scipy fit
    return alpha, beta

#call shower_profile_fit appropriately and save alpha/beta, variance
# store alpha/beta as class variable
# ^ separately in EventData there should be a dictionary to match appropriate
# alpha/beta with bin (? for later) [i.e. for the two different x0]

### add predicted shower energy to event data instances

def predicted_shower(event):
    '''Calculates shower energy prediction for true energy.

    Takes in:
    - an event
     - uses t, measured_energy, xyz of hits
     - alpha, beta (class variables selected based on xyz of event)
    '''
    event.t = #calculate
    event_inputs = (event.measured_energy, event.t)
    alpha = event.alpha[]#select
    beta = event.beta[]
    return shower_profile(event.inputs, alpha, beta)

def error(event):
    '''Returns error in shower_energy prediction for given event.'''
    return event.gamma - event.shower_energy

errors = []

for event in event_list:
    event.shower_energy = predicted_shower(event)
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
