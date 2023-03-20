from showerProfileUtils import parseTrainingData
from showerProfileDataUtils import pickEvent, toDataSpace, boundaryCheck, savePlot
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
import numpy as np
import time

start_time = time.time()

event_list = parseTrainingData()

boundaryCheck(event_list)

# event selection & data space
event_to_analyze = pickEvent(False, event_list, lambda lst: len(lst)//2 + 10)
D, E = toDataSpace(event_to_analyze)

# Matplotlib 3D scatter plot
fig = plt.figure()
ax = Axes3D(fig)

# ransac model fit with test data
avg_distance = np.mean(pdist(D))
rs, mt = 2*avg_distance//5, len(D[:, 0])//2
ransac = RANSACRegressor(residual_threshold=rs, max_trials=mt)
xy = D[:, :2]
z = D[:, 2]
ransac.fit(xy, z)

# inlier and outlier data
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
inlierD = D[inlier_mask, :]
outlierD = D[outlier_mask, :]
inlierE = np.array(E).T[inlier_mask]

# scatter inlier data
ax.scatter(inlierD[:, 0], inlierD[:, 1], inlierD[:, 2], c='blue', label='Inliers')

# plot formattting
ax.legend(loc='upper left')
ax.set_title('Hits for a single randomly selected event in the detector.')
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

ax.scatter(outlierD[:, 0], outlierD[:, 1], outlierD[:, 2], c='red', label='Outliers')

print('Plot finished!')
print(f'Time: {round(time.time() - start_time, 2)} seconds')

plt.show()
# savePlot(plt, "showerProfilePlots", "consistent_hit_plot")

# TODO: ransac reg fit line from inlier dataset
# TODO: func (hit1, hit2) --> output distance, energy difference

"""
for every hit along the regression line
hit[4] --> Energy
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