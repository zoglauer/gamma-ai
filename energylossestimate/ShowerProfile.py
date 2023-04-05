from sklearn.exceptions import UndefinedMetricWarning
from showerProfileUtils import parseTrainingData
from showerProfileDataUtils import toDataSpace, naiveShowerProfile, \
    boundaryCheck, combineEValues, sumAndExtend, zBiasedInlierAnalysis, plotSaveEvent
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

num_events = len(event_list)
x = []
Y = []
time_thresh = 10

# ignore the weak RANSAC analyses (on the order of 10^0 weak sets, out of 10^3 total sets)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

for i in range(num_events):

    event = event_list[i]
    geometricData, energyData = toDataSpace(event)
    inlierGeoData, inlierEnergyData, outlierGeoData = zBiasedInlierAnalysis(geometricData, energyData)

    if inlierGeoData is not None and len(inlierGeoData > 20):
        x, y = naiveShowerProfile(inlierGeoData, inlierEnergyData, 0.5)
        Y = sumAndExtend(Y, y)

        if (time.time() - start_time) > time_thresh:
            print(f'{i} events so far! Time: {round(time.time() - start_time, 3)}')
            time_thresh += time_thresh

    # uncomment to save the plot of each event
    # plotSaveEvent(plt, event, "event")

# average Y over the number of events
Y = list(map(lambda y_val : y_val / num_events, Y))

gdfig, ax2D = plt.subplots()
ax2D.scatter(x, Y)
ax2D.set_title('Energy Deposited v. Depth')
ax2D.set_xlabel('X [euclidean penetration in radiation lengths]')
ax2D.set_ylabel('E / (E0 * dX) [MeV]')
ax2D.set_xlim(left=0, right=25)

print(f'Gamma Distribution Attempt Complete! {num_events} Events')
print(f'Time: {round(time.time() - start_time, 2)} seconds')

plt.show()

"""
Questions for Dr. Z:
1. What should the y and x axes be?
2. How do we use this graph to calculate the total incident energy? Do we integrate the curve? What's next?
3. Since there are two critical energies and two radiation lengths, which should I use to normalize the Y and X 
axes, respectively? (currently I normalized the X axis by using the avg X0)
4. Is the critical energy used in the shower profile graph? (doesn't seem to be in the paper)

Lots of clarifications:
1. Use the current depth / radiation lengths to get the total penetration for the x - axis
    a. current depth = num_rad_lengths in tracker + num_rad_lengths in calorimeter
2. For the Y axis, plot dE/dX
3. Fit a gamma distribution to EACH curve fo EACH event
4. Find the incident energy, E0, using the formula in the paper
"""
