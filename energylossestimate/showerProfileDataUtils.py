import math
import numpy as np
from energylossestimate.DetectorGeometry import DetectorGeometry
from energylossestimate.showerProfileUtils import get_num_files

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

def naiveShowerProfile(energies, data):
    """Use all inlier data to chart a rough gamma distribution."""

    x = data[:, 0].T
    y = data[:, 1].T
    z = data[:, 2].T

    dEdX = []
    X = []
    current_depth = 0
    E0 = energies[0]

    for h in range(1, len(data[:, 0])):
        distance = dist(x[h], y[h], z[h], x[h-1], y[h-1], z[h-1])
        dX = distance / DetectorGeometry.radLengthForZ(z[h])
        # dEdX.append((energies[h] - energies[h-1]) / dX)
        dEdX.append( (energies[h] / E0) / dX )
        current_depth += dX
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

