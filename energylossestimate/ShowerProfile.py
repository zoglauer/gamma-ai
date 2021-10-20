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


''' TO TRAIN ON SAVIO: STEPS:
1. login and setup up cosimachine learning and megalib libr
2. run SavioSubmitScript_ShowerProfile.sh using sbatch.
3. to see output, look at slurm-$JOBID.out using vim (or vim extension via VS code).
4. printed outputs should be there.
5. change params for submit script as needed.
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
maxfev = 10000

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
    #measured_energy, x, y, z = xdat
    #pos = np.array((x, y, z))
    measured_energy, distance = xdat
    #start_pos = hits[0]
    #end_pos = hits[-1]
    #distance = np.linalg.norm(start_pos - end_pos)
    
    gamma = special.gamma(alpha)
    numerator = (beta * distance)**(alpha - 1) * beta * np.exp(-1 * beta * distance)
    return measured_energy * (numerator / gamma)


def build_xdat(events):
    measured_energies = np.array([event.measured_energy for event in event_list], dtype=np.float64)
    start_pos = np.array([event.hits[0, 0:3] for event in event_list], dtype=np.float64)
    end_pos = np.array([event.hits[-1, 0:3] for event in event_list], dtype=np.float64) 
    
    dist = np.linalg.norm(start_pos - end_pos, axis=1, ord=2)
    return (measured_energies, dist) 


def shower_optimize(f, events, total_energies=None, initial_guesses=None):
    """Finds alpha and beta for shower_profile().

    Pass in shower_profile() for f.
    Returns array with vals for alpha and beta and 2D array with variance.

    """
    #measured_energies = np.array([event.measured_energy for event in event_list])
    #start_pos = np.array([event.hits[0, 0:3] for event in event_list])
    #end_pos = np.array([event.hits[-1, 0:3] for event in event_list]) 
    
    #dist = np.linalg.norm(start_pos - end_pos, axis=1, ord=2)
    measured_energies, dist = build_xdat(events)
    #print('spos shape:', start_pos.shape)
    #print('epos shape:', end_pos.shape)
    #start_pos = pos[:, 0]
    #end_pos = pos[:, -1]
    #x_pos = hits[0, :]
    #y_pos = hits[1, :] # ask auden: why -1?
    #z_pos = hits[2, :]
    if events and total_energies == None:
        total_energies = [event.gamma_energy for event in event_list]
    else:
        raise ValueError
    if initial_guesses == None:
        initial_guesses = .5, .5
        
    print("XDAT - INPUT VAR META DATA :: ")
    print("meng:", len(measured_energies), type(measured_energies))
    #print("xpos:", len(x_pos), type(x_pos))
    #print("ypos:", len(y_pos), type(y_pos))
    #print("zpos:", len(z_pos), type(z_pos))
    #print("spos:", start_pos.shape, type(start_pos))
    #print("epos:", end_pos.shape, type(end_pos))
    print("dist:", dist.shape, type(dist))
    print("::::::::::::::::::::::::::::::::")
    return optimize.curve_fit(f, (measured_energies, dist), total_energies, initial_guesses, maxfev=maxfev)

gamma_energies = [event.gamma_energy for event in event_list]
initial_guesses = .5, .5 # TODO: set random seed an maybe pull from uniform dist. --> iterate over time to find best initial guess.
# event_energies = [event.measured_energy for event in event_list]
fitted_params, variance = shower_optimize(shower_profile, event_list, initial_guesses=initial_guesses) #, gamma_energies)
alpha = fitted_params[0]
beta = fitted_params[1]

print("========FITTED PARAMS ========")
print('alpha:', alpha)
print('beta:', beta)
print("==============================")

print('-------- AVG ERROR -------')
errs = []
for event in event_list:
    xdat = build_xdat([event]) # TODO: !! eventually we might batch this
    event.shower_energy = shower_profile(xdat, alpha, beta)
    errs.append(LA.norm(event.gamma_energy - event.shower_energy))
    #shower_profile(event, alpha, beta)

avg_err = sum(errs)/len(event_list)
print("average MSE error:", avg_err)


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
