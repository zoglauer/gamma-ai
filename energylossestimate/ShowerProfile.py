from sklearn.exceptions import UndefinedMetricWarning
from showerProfileUtils import parseTrainingData
from showerProfileDataUtils import toDataSpace, naiveShowerProfile, \
    boundaryCheck, sumAndExtend, zBiasedInlierAnalysis, plotSaveEvent
import matplotlib.pyplot as plt
import time
import warnings

start_time = time.time()

event_list = parseTrainingData()

# uncomment to check if all hits are within bounds ~ takes around 18sec
# boundaryCheck(event_list)
# print(f'Starting Inlier Plot. Time: {round(time.time() - start_time, 2)} seconds')

### NAIVE SHOWER PROFILE APPROACH
print(f'Starting Shower Analysis. Time: {round(time.time() - start_time, 2)} seconds')

num_events = len(event_list)
x = []
Y = []
time_thresh = 10

# ignore the weak RANSAC analysis (on the order of 10^0 weak sets, out of 10^3 total sets)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

for i in range(num_events):

    event = event_list[i]
    geometricData, energyData = toDataSpace(event)
    inlierGeoData, inlierEnergyData, outlierGeoData = zBiasedInlierAnalysis(geometricData, energyData)

    if inlierGeoData is not None and len(inlierGeoData > 20):
        x, y = naiveShowerProfile(inlierGeoData, inlierEnergyData, 1)
        Y = sumAndExtend(Y, y)

        if (time.time() - start_time) > time_thresh:
            print(f'{i} events so far! Time: {round(time.time() - start_time, 3)}')
            time_thresh += time_thresh

    # uncomment to save the plot of each event
    # plotSaveEvent(plt, event, "event")

gdfig, ax2D = plt.subplots()
ax2D.scatter(x, Y)
ax2D.set_title('Energy Deposited v. Depth')
ax2D.set_xlabel('X [euclidean penetration in cm]')
ax2D.set_ylabel('E [MeV]')

print(f'Gamma Distribution Attempt Complete! {num_events} Events')
print(f'Time: {round(time.time() - start_time, 2)} seconds')

plt.show()

"""
for every hit along the regression line
hit[3] --> Energy
tracker or calorimeter --> X0
distance from hit1 - hit2 (X = rad length) --> X / X0 (t or c)
make a plot of deposition energy between hits & the change in radiation length (dE/dX - y, X - x)

dE/dX
  |
  |
  |
  0 --------------- X

note: for t intervals where a hit is not found on the regression line, use the closest nearby hit
"""
