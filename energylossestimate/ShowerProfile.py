import matplotlib

from showerProfileUtils import parseTrainingData
from DetectorGeometry import DetectorGeometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

start_time = time.time()

event_list = parseTrainingData()

# BOUNDARY CHECK

checks = []
for event in event_list:
    checks.extend([DetectorGeometry.verifyHit(hit) for hit in event.hits])

print("Percentage in bounds: ", 100 * sum(checks) / len(checks))
print("Number of hits out of bounds: ", len(checks) - sum(checks))

# TODO: 3d plot hits of event_to_analyze, fit line & identify outliers
r = random.randint(0, len(event_list))
event_to_analyze = event_list[r]

# matlob 3D scatter plot figure
fig = plt.figure()
ax = Axes3D(fig)

x_vals = []
y_vals = []
z_vals = []

for hit in event_to_analyze.hits:
    x_vals.append(hit[0])
    y_vals.append(hit[1])
    z_vals.append(hit[2])

ax.scatter(x_vals, y_vals, z_vals)
plt.savefig('3D plot of hits.png')

# OLD SHOWER PROFILE STUFF:
"""
import math
from scipy import special
from scipy.optimize import optimize
import sys
import numpy as np

event_list = event_list[0:1]

# print(inspect.getmembers(EventData, lambda a:not(inspect.isroutine(a))))

# Define geometry of system (cm)
# MAYBE should be made into a class in event_data that can be imported
si_x_min, si_x_max = -45.8, 45.8
si_y_min, si_y_max = -48.3, 48.3
si_z_min, si_z_max = 10.2, 45

# IMPORTANT VALUES THAT WILL BE USED FOR CALCUALTING T AND ENERGYLOSS

#radiation length in g*cm^-2
radiation_length = 21.82

# Units: meV
critical_energy = 40.19

# OVERALL max/min?

x_vals = []
y_vals = []
z_vals = []


# t_calculate provides the

# hit[0] = x cordinate
# hit[1] = y cordinate
# hit[2] = z cordinate
# hit[3] = measured energy
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

all_bins = []

for a in x_range:
    for b in y_range:
        for c in z_range:
            all_bins.append([a, b ,c])

bin_names = range(0, len(all_bins))
geometry = list(map(list, zip(bin_names, all_bins)))

# bin_name, then an array consisting of three arrays

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
    #means the hit was recorded outside the measured region
    return None

for event in event_list:
    bins = []
    for hit in event.hits:
        bins.append(bin_find(hit, geometry))
    event.hits = np.append(event.hits, np.array(bins).reshape(-1, 1), 1)
# add bins to hits as column 5

# Find energy in each bin and calculate t accordingly



bins = {}

for event in event_list:
    for hit in event.hits:
        if hit[4] in bins:
            bins[hit[4]].append(list(hit))
        else:
            bins[hit[4]] = [list(hit)]

t_values = {}

energy_values = {}


for bin in bins.keys():
    bins.get(bin)
    total_energy = 0
    counter = 0
    for i in bins.get(bin):
        total_energy += i[3]
        counter += 1
    energy_values[bin] = total_energy / counter


for bin in energy_values.keys():
    x_value = radiation_length * ((np.log(energy_values.get(bin)) * np.log(critical_energy)) / np.log(2))
    t = x_value / radiation_length
    t_values[bin] = t



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
    # '''
    gamma_for_hits = []
    for bin in energy_values.keys():
        t = t_values.get(bin)
        numerator = (energy_values.get(bin) * b) * (beta*t ** (alpha - 1) * math.e ** ((-1 * beta) * t))
        inverse = special.gamma(alpha) / numerator
        gamma_for_hits.append(inverse)
    # gamma = special.gamma(alpha)
    # gamma_for_hits = []
    # for hit in event_hits:
    #     t = t_values.get(hit[4])
    #     numerator = (hit[3] * b) * (beta*t ** (alpha - 1) * math.e ** ((-1 * beta) * t))
    #     energy = numerator / gamma
    #     # hit[3] = measured_energy
    #     t_beta_alpha = (beta * hit[6])**(alpha - 1) * \
    #         beta * np.exp(-1 * beta * hit[6])
    #     gamma_for_hits.append((hit[3] * gamma) / t_beta_alpha)
    print(gamma_for_hits)
    np.log()
    # return 2

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

# print("--------------------------")
# print(f"Added shower profile's predicted energy to {len(event_list)} events.")
# print("Info: storing updated data.")

# ''' TODO FIX ARGS --> it errors because of restricted access to data sim file.
# if args.savefileto != "":
#     save_file = args.savefileto
# if not os.path.exists(save_file):
#     print(f"The savefile does not exist: {save_file}. Creating new...")
#     with open(save_file, 'w') as fp:
#         pass # write nothing.


# with open(save_file, "wb") as file_handle:
#     pickle.dump(event_list, file_handle)
# print("Info: done.")
# '''

end_time = time.time()
print('Total time elapsed:', end_time - start_time, 's.')

sys.exit(0)"""
