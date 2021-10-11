'''
Some basic outline & pseudocode:
1. we still want to load the data. (thankfully we have this from the other model to pull from)
2. fitting dE(t)/dt = P(t) = E * (B*t)**(a-1)*B*exp(-B*t)/Gamma(a)
    - we have from our data the given values for E and dE(t)/dt = P(t)
    - E = energy of the event in the beginning
    - P = energy measured by the calorimeter (hit energy)
    - Gamma is a statistical function provided by various python libraries!
        - see here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html
    
What we are essentially attempting here is to produce all the possible 
alphas (a) and Betas(b), rewriting the equation with a and b as our unknowns.

3. fitting the curve: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
it is pretty simple with scipy.optimize.curve_fit, where we add a function and the data and it will give us the unknown paramters!
function should take in the unknown params (a and b in our case) and E and P for x and y.
- our E and P data should be split up by magnitude (based on what Rhea did) so we should attempt this first
- ideally our final model will be able to recognize the magnitude and feed it into an array of its relevant fitted equation (i believe.)
 ** material differences --> maybe why Rhea took a different approach to the fitting than using a library **
4. return function & attempt plots :)
-  this is a more flexible part, we will probably have to play around with a small dataset to test out what is possible & efficient.
    
5. Further steps:

Create alpha and beta distributions for specific energies

Account for variation from individual events.
'''

import scipy
import pickle
from math import sqrt, exp
import numpy as np
from event_data.py import EventData

parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='EnergyEstimate.p1.sim.gz', help='File name used for training/testing')

if args.filename != "":
    file_name = args.filename
if not os.path.exists(file_name):
    print(f"Error: The training data file does not exist: {file_name}")
    sys.exit(0)
print(f"CMD: Using file {file_name}")

with open(file_name, "rb") as file_handle:
   event_list = pickle.load(file_handle)

def shower_profile(event, alpha, beta):
    """Function that represents the shower profile.

    Takes in the event and predicts total gamma energy using alpha and beta to fit.
    Described in [source]
    shower_optimize() fits for alpha and beta.
    """
    energy = event.measured_energy
    hits = event.hits
    start_pos = hits[0]
    end_pos = hits[-1]
    distance = np.linalg.norm(end_pos - start_pos)
    gamma = scipy.special.gamma(alpha)
    numerator = (beta * distance)**(alpha - 1) * beta * exp(-1 * beta * distance)
    return measured_energy * (numerator / gamma)

def shower_optimize(f, events, gamma_energies):
    """Finds alpha and beta for shower_profile().

    Pass in shower_profile() for f.
    Returns array with vals for alpha and beta and 2D array with variance.
    """
    return scipy.optimize.curve_fit(f, events, gamma_energies)

gamma_energies = [event.gamma_energy for event in event_list]
fitted_params, variance = shower_optimize(shower_profile, event_list, gamma_energies)
alpha = fitted_params[0]
beta = fitted_params[1]

for event in event_list:
    event.shower_energy = shower_profile(event, alpha, beta)

print(f"Added shower profile's predicted energy to {len(event_list)} events.")
print("Info: storing updated data.")

with open(file_name, "wb") as file_handle:
    pickle.dump(event_list, file_handle)
print("Info: done.")

sys.exit(0)
