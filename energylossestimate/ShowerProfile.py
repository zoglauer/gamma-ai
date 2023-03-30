from showerProfileUtils import parseTrainingData
from showerProfileDataUtils import pickEvent, toDataSpace, savePlot, naiveShowerProfile, \
    inlierAnalysis, boundaryCheck, sumAndExtend, zBiasedInlierAnalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

start_time = time.time()

event_list = parseTrainingData()

# uncomment to check if all hits are within bounds ~ takes around 18sec
# boundaryCheck(event_list)
# print(f'Starting Inlier Plot. Time: {round(time.time() - start_time, 2)} seconds')

# plot an event from the data set
event_to_analyze = pickEvent(False, event_list, lambda lst: 2)
D, E = toDataSpace(event_to_analyze)

# Matplotlib 3D scatter plot
fig = plt.figure()
ax = Axes3D(fig)

inlierD, inlierE, outlierD = zBiasedInlierAnalysis(D, E)

# scatter inlier data
ax.scatter(inlierD[:, 0], inlierD[:, 1], inlierD[:, 2], c='blue', label='Inliers')

# plot formattting
ax.legend(loc='upper left')
ax.set_title('Hits for a single selected event in the detector.')
ax.set_xlabel('Hit X (cm)')
ax.set_ylabel('Hit Y (cm)')
ax.set_zlabel('Hit Z (cm)')

# LINEAR REGRESSION
# @source adapted StackOverFlow https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d using chatGPT
datamean = inlierD.mean(axis=0)
uu, dd, vv = np.linalg.svd(inlierD - datamean)
linepts = vv[0] * np.mgrid[-40:40:2j][:, np.newaxis]
linepts += datamean
ax.plot3D(*linepts.T)

"""
# PLANE

regressor = LinearRegression()
regressor.fit(D[inlier_mask, :2], D[inlier_mask, 2])
coef_x, coef_y = regressor.coef_
intercept = regressor.intercept_

xx, yy = np.meshgrid(D[inlier_mask, 0], D[inlier_mask, 1])
zz = coef_x * xx + coef_y * yy + intercept
ax.plot_surface(xx, yy, zz, alpha=0.5)
"""

# comment to hide outlier data (in red on plot)
ax.scatter(outlierD[:, 0], outlierD[:, 1], outlierD[:, 2], c='red', label='Outliers')

print('Inlier Outlier Plot finished! (will display after all computations complete)')

# uncomment to save inlier/outlier plot to directory
# savePlot(plt, "showerProfilePlots", "consistent_hit_plot")

### NAIVE SHOWER PROFILE APPROACH
print(f'Starting Shower Analysis. Time: {round(time.time() - start_time, 2)} seconds')

num_events = 3
X = []
Y = []
for i in range(num_events):
    event = event_list[i]
    geometricData, energyData = toDataSpace(event)
    inlierGeoData, inlierEnergyData, outlierGeoData = zBiasedInlierAnalysis(geometricData, energyData)
    x, y = naiveShowerProfile(inlierGeoData, inlierEnergyData)
    X = x
    Y = sumAndExtend(Y, y)

gdfig, ax2D = plt.subplots()
ax2D.scatter(X, Y)
ax2D.set_title('Energy Deposited v. Depth')
ax2D.set_xlabel('X [penetration in cm]')
ax2D.set_ylabel('(1/Ec)E [units of critical energy]')
plt.xlim([0, 45]) # ignore noise

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