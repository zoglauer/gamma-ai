from showerProfileUtils import parseTrainingData
from DetectorGeometry import DetectorGeometry
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D
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

# TODO: 3d plot hits of event_to_analyze, fit line & identify outliers

# random event selection
r = random.randint(0, len(event_list))
event_to_analyze = event_list[r]

# Matplotlib 3D scatter plot & RANSAC = outlier resistant regression model.
fig = plt.figure()
ax = Axes3D(fig)
ransac = RANSACRegressor()

x_vals = []
y_vals = []
z_vals = []

for hit in event_to_analyze.hits:
    x_vals.append(hit[0])
    y_vals.append(hit[1])
    z_vals.append(hit[2])

D = np.column_stack((x_vals, y_vals, z_vals))
ransac.fit(D, np.zeros(len(x_vals)))

# axis labels
ax.set_title('Hits for a single randomly selected event in the detector.')
ax.set_xlabel('Hit X (cm)')
ax.set_ylabel('Hit Y (cm)')
ax.set_zlabel('Hit Z (cm)')

# generate points on the fitted line
line_x = np.arange(D[:,0].min(), D[:,0].max(), 0.1)[:, np.newaxis]
line_y = ransac.predict(line_x)
line_z = np.zeros_like(line_y)

# scatterplot and regression
ax.scatter(x_vals, y_vals, z_vals)
ax.plot(line_x, line_y, line_z, color='red')

plt.savefig('3D_plot_of_hits.png')
