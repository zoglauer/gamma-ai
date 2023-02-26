from showerProfileUtils import parseTrainingData
from DetectorGeometry import DetectorGeometry
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
import numpy as np
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

# PLOT RANDOM EVENT:

# random event selection
r = random.randint(0, len(event_list))
event_to_analyze = event_list[r]

# Matplotlib 3D scatter plot & RANSAC = outlier resistant regression model.
fig = plt.figure()
ax = Axes3D(fig)

# format hit x, y, z
x_vals = []
y_vals = []
z_vals = []

for hit in event_to_analyze.hits:
    x_vals.append(hit[0])
    y_vals.append(hit[1])
    z_vals.append(hit[2])

x_vals, y_vals, z_vals = map(np.array, [x_vals, y_vals, z_vals])
D = np.column_stack((x_vals, y_vals, z_vals))

"""
|       |
| | | | |
| x y z |
| | | | |
|       |
"""

# used to set residual threshold
distances = pdist(D)
avg_distance = np.mean(distances)

# ransac model fit with test data
ransac = RANSACRegressor(residual_threshold=2*avg_distance)

xy = D[:, :2]
z = D[:, 2]
ransac.fit(xy, z)

# axis labels
ax.set_title('Hits for a single randomly selected event in the detector.')
ax.set_xlabel('Hit X (cm)')
ax.set_ylabel('Hit Y (cm)')
ax.set_zlabel('Hit Z (cm)')

# TODO: fix ransac fit
"""
# generate points on the fitted line
line_x = np.column_stack((np.arange(D[:,0].min(), D[:,0].max(), 0.1),
                          np.zeros_like(np.arange(D[:,0].min(), D[:,0].max(), 0.1)),
                          np.zeros_like(np.arange(D[:,0].min(), D[:,0].max(), 0.1))))
line_y = ransac.predict(line_x)
line_z = np.zeros_like(line_y)

ax.plot(line_x, line_y, line_z, color='red')
"""

# inlier and outlier masks
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# inlier, outlier, data scatterplot
ax.scatter(x_vals, y_vals, z_vals, c='orange')
ax.scatter(D[inlier_mask, 0], D[inlier_mask, 1], D[inlier_mask, 2], c='blue', label='Inliers')
ax.scatter(D[outlier_mask, 0], D[outlier_mask, 1], D[outlier_mask, 2], c='red', label='outliers')

plt.savefig('3D_plot_of_hits_with_reg001.png')
