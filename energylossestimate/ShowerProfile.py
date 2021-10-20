'''Finds alpha, beta parameters for shower profile, then estimates event energy.

1. Load data processed by event_extractor.py (pass in to argparser).

2. Fitting dE(t)/dt = P(t) = E * (B*t)**(a-1)*B*exp(-B*t)/Gamma(a):
    - we have from our data the given values for E and dE(t)/dt = P(t)
    - E = energy of the event in the beginning
    - P = energy measured by the calorimeter (hit energy)
    - scipy.special.gamma used for Gamma function

3. Curve fitting done using scipy.optimize.curve_fit:
 - Takes in measured_energy for E
 - In paper (see README), t = x / X0. Assumed X0 will be incorporated into beta since constant
  - X0 likely dependent on material differences that should be reflected in data set used.
  - Distance between first and last hit x, y, z coordinates is used.
 - Alpha and beta found via curve fitting.

4. Use alpha and beta found to predict total energy given measured energy
 - EventData class instances updated with this.
 - Class instances written into file produced by event_extractor.

5. Further steps:
- Create alpha and beta distributions for specific energies
 - Do by returning function with estimated alpha, beta (i.e., create HOF in EventData class)?
- Account for variation from individual events.
- Incorporate into event_extractor.py
'''

import pickle
import argparse
import os
import sys
from math import exp
import scipy
from scipy import optimize, special
import numpy as np
from event_data import EventData

parser = argparse.ArgumentParser(description=
        'Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz',
        help='File name used for training/testing')

args = parser.parse_args()

if args.filename != "":
    file_name = args.filename
if not os.path.exists(file_name):
    print(f"Error: The training data file does not exist: {file_name}")
    sys.exit(0)
print(f"CMD: Using file {file_name}")

with open(file_name, "rb") as file_handle:
    event_list = pickle.load(file_handle)
    #print(event_list)
    #print('event list type:', type(event_list[0]))

def shower_profile(xdat, alpha, beta):
    """Function that represents the shower profile.

    Takes in the event and predicts total gamma energy using alpha and beta to fit.
    Described in [source]
    shower_optimize() fits for alpha and beta.
    """
    #measured_energy = event.measured_energy
    #hits = event.hits
    measured_energy, start_pos, end_pos = xdat
    #start_pos = hits[0]
    #end_pos = hits[-1]
    distance = np.linalg.norm(end_pos - start_pos)
    gamma = special.gamma(alpha)
    numerator = (beta * distance)**(alpha - 1) * beta * exp(-1 * beta * distance)
    return measured_energy * (numerator / gamma)

def shower_optimize(f, events, total_energies=None, initial_guesses=None):
    """Finds alpha and beta for shower_profile().

    Pass in shower_profile() for f.
    Returns array with vals for alpha and beta and 2D array with variance.

    """
    measured_energies = [event.measured_energy for event in event_list]
    hits = [event.hits for event in event_list]
    start_pos = hits[0]
    end_pos = hits[-1] # ask auden: why -1?
    if events and total_energies == None:
        total_energies = [event.gamma_energy for event in event_list]
    else:
        raise ValueError
    if initial_guesses == None:
        initial_guesses = .5, .5
        
    print("XDAT INPUT SIZES")
    print("meng:", len(measured_energies), type(measured_energies))
    print("spos:", len(start_pos), type(start_pos))
    print("epos:", len(end_pos), type(end_pos))
    return optimize.curve_fit(f, (measured_energies, start_pos, end_pos), total_energies, initial_guesses)

gamma_energies = [event.gamma_energy for event in event_list]
initial_guesses = .5, .5 # TODO: set random seed an maybe pull from uniform dist. --> iterate over time to find best initial guess.
# event_energies = [event.measured_energy for event in event_list]
fitted_params, variance = shower_optimize(shower_profile, event_list, initial_guesses=initial_guesses) #, gamma_energies)
alpha = fitted_params[0]
beta = fitted_params[1]
print('alpha:', alpha)
print('beta:', beta)
for event in event_list:
    event.shower_energy = shower_profile(event, alpha, beta)

print(f"Added shower profile's predicted energy to {len(event_list)} events.")
print("Info: storing updated data.")

with open(file_name, "wb") as file_handle:
    pickle.dump(event_list, file_handle)
print("Info: done.")


sys.exit(0)
