from sklearn.exceptions import UndefinedMetricWarning
from showerProfileUtils import parseTrainingData
from showerProfileDataUtils import toDataSpace, \
    boundaryCheck, zBiasedInlierAnalysis, showPlot, showerProfile, interpretAndDiscretize, savePlot
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

incident_energies = []
gamma_energies = []
measured_energies = []
analyzed_count = 0
time_thresh = 10

# ignore the weak RANSAC analyses (on the order of 10^0 weak sets, out of 10^3 total sets)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

for event in event_list[:50]:

    geometricData, energyData = toDataSpace(event)
    inlierGeoData, inlierEnergyData, outlierGeoData = zBiasedInlierAnalysis(geometricData, energyData)

    if inlierGeoData is not None and len(inlierGeoData > 20):

        # uncomment to show the plot of each event
        # showPlot(plt, event)

        # uncomment to save the plot of each event
        # savePlot(plt, event, "event")

        incident_energies.append(round(showerProfile(inlierGeoData, inlierEnergyData, 0.05, plt), 3))
        gamma_energies.append(event.gamma_energy)
        measured_energies.append(round(event.measured_energy, 3))

    analyzed_count += 1

    if (time.time() - start_time) > time_thresh:
        print(f'{analyzed_count} events so far! Time: {round(time.time() - start_time, 10)}')
        time_thresh += time_thresh

print(f'Gamma Distribution Attempt Complete! {analyzed_count} Events')
print(f'Time: {round(time.time() - start_time, 2)} seconds')
print([incident_energies[i] / gamma_energies[i] for i in range(len(incident_energies))])

plt.show()
