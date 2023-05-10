from sklearn.exceptions import UndefinedMetricWarning
from Curve import Curve
from showerProfileUtils import parseTrainingData
from showerProfileDataUtils import toDataSpace, \
    boundaryCheck, zBiasedInlierAnalysis, showPlot, interpretAndDiscretize, savePlot
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np

start_time = time.time()

event_list = parseTrainingData()

# uncomment to check if all hits are within bounds ~ takes around 18sec
# boundaryCheck(event_list)
# print(f'Starting Inlier Plot. Time: {round(time.time() - start_time, 2)} seconds')

### SHOWER PROFILE
print(f'Starting Shower Analysis. Time: {round(time.time() - start_time, 2)} seconds')

curveFamily = []
curveEvents = []
analyzed_count = 0
time_thresh = 10
bin_size = 0.05

# ignore the weak RANSAC analyses (on the order of 10^0 weak sets, out of 10^3 total sets)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

event_list = event_list[:1000]
training_data = event_list[ : int(0.80 * len(event_list))]

# simulation stage - generate shower profiles
for event in training_data:

    geometricData, energyData = toDataSpace(event)
    inlierGeoData, inlierEnergyData, outlierGeoData = zBiasedInlierAnalysis(geometricData, energyData)

    if inlierGeoData is not None and len(inlierGeoData > 20):

        # uncomment to show the plot of each event
        # showPlot(plt, event)

        # uncomment to save the plot of each event
        # savePlot(plt, event, "event")

        t_expected, dEdt_expected = interpretAndDiscretize(inlierGeoData, inlierEnergyData, bin_size)
        gamma_energy = event.gamma_energy
        curve = Curve.fit(t_expected, dEdt_expected, gamma_energy, bin_size)
        if curve is not None:
            curveFamily.append(curve)
            curveEvents.append(analyzed_count)

    analyzed_count += 1

    # comment to ommitt time steps
    if (time.time() - start_time) > time_thresh:
        print(f'{analyzed_count} events so far! Time: {round(time.time() - start_time, 10)}')
        time_thresh += 10

print(f'Pseudo Simulation Stage Complete! {analyzed_count} Events')
print(f'Time: {round(time.time() - start_time, 2)} seconds')
print(f'Simulated events yielding curves: {len(curveEvents)}')

if len(curveEvents) < 0.20 * len(event_list):
    experiment_data = event_list[ -len(curveEvents): ]
else:
    experiment_data = event_list[ int(0.80 * len(curveEvents)) : ]

errors = {}
training_r_squared = {}
test_r_squared = {}
## experimental energy estimates
for event in experiment_data:

    geometricData, energyData = toDataSpace(event)
    inlierGeoData, inlierEnergyData, outlierGeoData = zBiasedInlierAnalysis(geometricData, energyData)

    if inlierGeoData is not None and len(inlierGeoData > 20):
        t, dEdt = interpretAndDiscretize(inlierGeoData, inlierEnergyData, bin_size)
        gamma_energy = event.gamma_energy

        curve = Curve.fit(t, dEdt, gamma_energy, bin_size, True)
        if curve is not None:
            closest = min(curveFamily, key=lambda c: c.compare(curve, bin_size))
            energy_estimate = closest.energy
            # print(f'Estimate: {energy_estimate}, True: {gamma_energy}')
            error = abs(energy_estimate - gamma_energy)
            errors[gamma_energy] = error
            training_r_squared[closest.r_squared] = error
            test_r_squared[curve.r_squared] = error

plt.scatter(list(errors.keys()), list(errors.values()))
plt.xlabel("Energy")
plt.ylabel("Error")
savePlot(plt, "ErrorEnergy")

plt.scatter(list(training_r_squared.keys()), list(training_r_squared.values()))
plt.xlabel("R_squared of Training Curve")
plt.ylabel("Error")
savePlot(plt, "ErrorRSquaredTraining")

plt.scatter(list(test_r_squared.keys()), list(test_r_squared.values()))
plt.xlabel("R_squared of Test Curve")
plt.ylabel("Error")
savePlot(plt, "ErrorRSquaredTest")

print(f'Average error: {np.mean(list(errors.values()))}')
# filter R^2 > 0.75
test_rs = list(test_r_squared.keys())
test_errors = list(test_r_squared.values())
filtered = [test_errors[i] for i in range(len(test_errors)) if test_rs[i] > 0.75]
print(f'Average error of test data with R^2 > 0.8: {np.mean(filtered)}')

