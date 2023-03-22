import math
import numpy as np
from energylossestimate.DetectorGeometry import DetectorGeometry
from energylossestimate.showerProfileUtils import get_num_files
from sklearn.linear_model import RANSACRegressor

def savePlot(plt, directory, name):
    num_files = get_num_files(directory)
    plt.savefig(f"{directory}/{name}{num_files}.png")

def pickEvent(random, event_list, selection):
    """returns an event for analysis based on random or selection criteria"""
    event_to_analyze = event_list[selection(event_list)]
    if random:
        r = random.randint(0, len(event_list))
        event_to_analyze = event_list[r]
    return event_to_analyze

def boundaryCheck(events):
    """check if all hits are in the bounds of the detector"""
    checks = []
    for event in events:
        checks.extend([DetectorGeometry.verifyHit(hit) for hit in event.hits])

    print("Percentage in bounds: ", 100 * sum(checks) / len(checks))
    print("Number of hits out of bounds: ", len(checks) - sum(checks))

def toDataSpace(event):
    """ Returns data in the form:
           |         |
           | | | | | |
           | x y z E |
           | | | | | |
           |         |
    where x, y, z, and E are horizontally stacked column vectors.
    """

    x_vals = []
    y_vals = []
    z_vals = []
    energies = []

    for hit in event.hits:
        x_vals.append(hit[0])
        y_vals.append(hit[1])
        z_vals.append(hit[2])
        energies.append(hit[3])

    x_vals, y_vals, z_vals = map(np.array, [x_vals, y_vals, z_vals])
    D = np.column_stack((x_vals, y_vals, z_vals))

    return D, energies

def naiveShowerProfile(data, energies):
    """Use all inlier data to chart a rough gamma distribution."""

    x = data[:, 0].T
    y = data[:, 1].T
    z = data[:, 2].T

    dEdX = []
    X = []
    current_depth = 0

    for h in range(0, len(data[:, 0]) - 1):

        critical_Energy = DetectorGeometry.critE(x[h], y[h], z[h])
        radiation_Length = DetectorGeometry.radLength(x[h], y[h], z[h])
        distance = dist(x[h], y[h], z[h], x[h-1], y[h-1], z[h-1]) # [cm]
        # dX = distance / radiation_Length
        dX = distance

        # ignore hits in the same spot (no change in radiation length) or with no deposited Energy
        if dX > 0 and energies[h] > 0:
            dEdX.append(
                (1 / critical_Energy) * energies[h] / dX
            )
            current_depth += distance
            X.append(current_depth)

    return X, dEdX

def dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def computeAvgDistBetweenHits(data):

    x = data[:, 0].T
    y = data[:, 1].T
    z = data[:, 2].T

    totalDist = 0
    numHits = len(data[:, 0])

    for h in range(1, numHits):
        totalDist += dist(x[h], y[h], z[h], x[h-1], y[h-1], z[h-1])

    return totalDist / numHits

def inlierAnalysis(geoData, energyData):
    """Returns inliers of geometric data and the *corresponding* energy data based on linear regression /
    RANSAC analysis."""

    # ransac model fit with test data
    avg_distance = computeAvgDistBetweenHits(geoData)
    rs, mt = 0.26 * avg_distance, 100
    ransac = RANSACRegressor(residual_threshold=rs, max_trials=mt)
    xy = geoData[:, :2]
    z = geoData[:, 2]
    ransac.fit(xy, z)

    # inlier and outlier data
    inlier_mask = ransac.inlier_mask_
    inlierD = geoData[inlier_mask, :]
    inlierE = np.array(energyData).T[inlier_mask]

    # uncomment if you need outlier data
    # outlier_mask = np.logical_not(inlier_mask)
    # outlierD = geoData[outlier_mask, :]

    return inlierD, inlierE

def firstNonZeroItemIndex(lst):
    """returns index, item of first non-null item in lst"""

    index = 0
    for item in lst:
        if item:
            return index, item
        index += 1
    return None
