from sklearn.exceptions import UndefinedMetricWarning

from Curve import Curve
from showerProfileUtils import parseTrainingData
from showerProfileDataUtils import toDataSpace, \
    boundaryCheck, zBiasedInlierAnalysis, showPlot, interpretAndDiscretize, savePlot
import matplotlib.pyplot as plt
import time
import warnings

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
bin_size = 0.06

# ignore the weak RANSAC analyses (on the order of 10^0 weak sets, out of 10^3 total sets)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# simulation stage - generate shower profiles
for event in event_list[:9900]:

    geometricData, energyData = toDataSpace(event)
    inlierGeoData, inlierEnergyData, outlierGeoData = zBiasedInlierAnalysis(geometricData, energyData)

    if inlierGeoData is not None and len(inlierGeoData > 20):

        # uncomment to show the plot of each event
        # showPlot(plt, event)

        # uncomment to save the plot of each event
        # savePlot(plt, event, "event")

        t_expected, dEdt_expected = interpretAndDiscretize(inlierGeoData, inlierEnergyData, bin_size)
        gamma_energy = event.gamma_energy
        if t_expected is not None:
            curve = Curve.fit(t_expected, dEdt_expected, gamma_energy, bin_size)
            if curve is not None:
                curveFamily.append(curve)
                curveEvents.append(analyzed_count)

    analyzed_count += 1

    # comment to ommitt time steps
    if (time.time() - start_time) > time_thresh:
        print(f'{analyzed_count} events so far! Time: {round(time.time() - start_time, 10)}')
        time_thresh += time_thresh

print(f'Pseudo Simulation Stage Complete! {analyzed_count} Events')
print(f'Time: {round(time.time() - start_time, 2)} seconds')
print(f'Simulated events yielding curves: {len(curveEvents)}')

errors = []
## experimental energy estimates
for event in event_list[9990:]:

    geometricData, energyData = toDataSpace(event)
    inlierGeoData, inlierEnergyData, outlierGeoData = zBiasedInlierAnalysis(geometricData, energyData)

    if inlierGeoData is not None and len(inlierGeoData > 20):
        t, dEdt = interpretAndDiscretize(inlierGeoData, inlierEnergyData, bin_size)
        gamma_energy = event.gamma_energy

        if t is not None:
            curve = Curve.fit(t, dEdt, gamma_energy, bin_size, True)
            if curve is not None:
                energy_estimate = min(curveFamily, key=lambda c: c.compare(curve, bin_size)).energy
                # print(f'Estimate: {energy_estimate}, True: {gamma_energy}')
                errors.append(100 * (energy_estimate - gamma_energy) / gamma_energy)

print(f'Average error: {sum(errors) / len(errors)}')
plt.show()
