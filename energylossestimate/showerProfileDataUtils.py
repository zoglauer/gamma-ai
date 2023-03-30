import math
import numpy as np
from scipy.spatial.distance import pdist
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

    x_vals, y_vals, z_vals, energies = map(np.array, [x_vals, y_vals, z_vals, energies])
    D = np.column_stack((x_vals, y_vals, z_vals))

    # sort by depth (z) descending
    z = D[:, 2]
    indices = np.argsort(-z)

    return D[indices], energies[indices]

def naiveShowerProfile(data, energies):
    """Use all inlier data to chart a rough gamma distribution."""

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    bin_size = 1

    # binned energies, E @ 5 corresp. to all energy deposits <= 5cm depth
    E = {i : 0 for i in range(bin_size, 150, bin_size)}

    # dEdX = []
    # X = []
    current_depth = 0
    g0x, g0y, g0z = x[0], y[0], z[0]

    for h in range(1, len(data[:, 0])):

        critical_Energy = DetectorGeometry.critE(x[h], y[h], z[h])
        # radiation_Length = DetectorGeometry.radLength(x[h], y[h], z[h])
        distance = dist(x[h], y[h], z[h], x[h-1], y[h-1], z[h-1]) # [cm]
        # dX = distance / radiation_Length

        # ignore hits in the same spot or with no deposited energy
        if distance > 0 and energies[h] > 0:

            key = current_depth - (current_depth % bin_size) + bin_size
            E[key] = E[key] + (1 / critical_Energy) * energies[h] * 10**-6

            # dEdX.append( (1 / critical_Energy) * energies[h] )
            # X.append(current_depth)
            current_depth = dist(x[h], y[h], z[h], g0x, g0y, g0z)

    x = list(E.keys())
    y = list(E.values())

    return x, y

def dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def sumAndExtend(lst1, lst2):
    if not lst1:
        return lst2
    else:
        i = 0
        while i < len(lst1):
            lst1[i] += lst2[i]
            i+=1
        lst1.extend(lst2[i:])

    return lst1

def inlierAnalysis(geoData, energyData):
    """Returns inliers of geometric data and the *corresponding* energy data based on linear regression /
    RANSAC analysis."""

    # ransac model fit with test data
    xy = geoData[:, :2]
    z = geoData[:, 2]
    adistxy = np.mean(pdist(xy))
    rs, mt = 0.28 * adistxy, 100
    ransac = RANSACRegressor(residual_threshold=rs, max_trials=mt)
    ransac.fit(xy, z)

    # inlier data
    inlier_mask = ransac.inlier_mask_
    inlierD = geoData[inlier_mask]
    inlierE = energyData[inlier_mask]

    # outlier data
    outlier_mask = np.logical_not(inlier_mask)
    outlierD = geoData[outlier_mask]

    return inlierD, inlierE, outlierD

def zBiasedInlierAnalysis(geoData, energyData):
    """Biases inlier analysis by prioritizing straight lines coming in the downwards z direction.
    Assumes data is ordered by depth (z) descending.
    :return inlierGeoData, inlierEnergyData, outlierGeoData """

    xy = geoData[:, :2]
    z = geoData[:, 2]
    n = len(geoData)

    # model top half of z data
    upper_z = z[:n//4]
    upper_xy = xy[:n//4]

    ransac = RANSACRegressor()
    ransac.fit(upper_xy, upper_z)

    # filter rest of data via model
    z_pred = ransac.predict(xy)
    inliers = np.abs(z - z_pred) < 0.5 * np.std(z_pred)
    outliers = np.logical_not(inliers)

    return geoData[inliers], energyData[inliers], geoData[outliers]

def firstNonZeroItemIndex(lst):
    """returns index, item of first non-null item in lst"""

    index = 0
    for item in lst:
        if item:
            return index, item
        index += 1
    return None
