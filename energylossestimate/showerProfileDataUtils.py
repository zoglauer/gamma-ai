import math
import numpy as np
from scipy.spatial.distance import pdist
from energylossestimate.DetectorGeometry import DetectorGeometry
from energylossestimate.showerProfileUtils import get_num_files
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

def savePlot(plt, event, filename):
    """Plots the given event on a 3d axis and saves it to showerProfilePlots with filename + enum. """
    plot(plt, event)
    num_files = get_num_files("showerProfilePlots")
    plt.savefig(f"showerProfilePlots/{filename}{num_files}.png")
    plt.close()

def showPlot(plt, event):
    """Shows the given event in 3d with inliers & outliers and linreg in interactive window. """
    plot(plt, event)
    plt.show()

def plot(plt, event):
    """Plot event data, inliers v outliers, linear regression. """

    geometricData, energyData = toDataSpace(event)
    inlierGeoData, inlierEnergyData, outlierGeoData = zBiasedInlierAnalysis(geometricData, energyData)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(inlierGeoData[:, 0], inlierGeoData[:, 1], inlierGeoData[:, 2], c='blue', label='Inliers')
    ax.scatter(outlierGeoData[:, 0], outlierGeoData[:, 1], outlierGeoData[:, 2], c='red', label='Outliers')

    # plot formattting
    ax.legend(loc='upper left')
    ax.set_title('Hits for a single selected event in the detector.')
    ax.set_xlabel('Hit X (cm)')
    ax.set_ylabel('Hit Y (cm)')
    ax.set_zlabel('Hit Z (cm)')

    # comment to remove linreg line from plot
    # Note: * is used to unpack the 2d array, where each row is a coordinate
    ax.plot3D(*linearRegressionLine(inlierGeoData).T)

    return ax

def linearRegressionLine(inlierGeoData):
    """LINEAR REGRESSION adapted from chatGPT3 and StackOverFlow
    https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d """
    datamean = inlierGeoData.mean(axis=0)
    uu, dd, vv = np.linalg.svd(inlierGeoData - datamean)
    linepts = vv[0] * np.mgrid[-50:50:2j][:, np.newaxis]
    linepts += datamean
    return linepts

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

def interpretAndDiscretize(data, energies, bin_size):
    """ Going in the downward z-dir, projects the euclidean distance vectors from hit to hit (data = inliers)
    and returns the energy deposition at each bin corresponding to ||proj|| / radiation_length @ current depth.
    """

    # linreg line points
    linepts = linearRegressionLine(data)
    linepts = linepts[np.argsort(-linepts[:, 2])]
    start_point = linepts[0, :]
    end_point = linepts[len(linepts) - 1, :]
    line_vec = end_point - start_point

    # binned energies, E @ bin_size corresp. to all energy deposits <= bin_size [cm] depth
    E = {i: 0 for i in np.arange(bin_size, 100, bin_size)}

    ### COMPLETELY DIFFERENT APPROACH
    """Move down the linear regression line and identify points that are on the horizontal plane 
    associated with that z depth or within a vertical error of some margin. Sum the energies of these 
    points, and track the depth from initial penetration along the line."""
    curr_vec = start_point
    dir_line = line_vec / np.linalg.norm(line_vec) # unit vector
    """dir_line is always length one. We are checking the space between two planes for points: the horizontal plane
    at z + vertical_margin and z - vertical_margin (notice these planes are centered at the horiz. plane @ z). 
    To avoid overlap of point checking, we want to only check half of the vertical distance from curr_vec to 
    curr_vec + dir_line above and below z, which is equivalently half of the magnitude of the projection of 
    dir_line onto the downward z vector."""
    vertical_margin = np.linalg.norm(proj(dir_line, np.array([0, 0, -1]))) / 2
    depth = 0 # in radiation lengths
    penned = False
    penetration = 0
    while penetration < np.linalg.norm(line_vec):
        zPlane = curr_vec[2]
        pts = pointsWithinZ(zPlane, vertical_margin, data, line_vec)
        if len(pts) != 0 and not penned:
            penned = True
            depth = 0 # true beginning of data
        plane_energy = sum([energies[i] for i in pts])
        radiation_length = DetectorGeometry.radLength(curr_vec[0], curr_vec[1], curr_vec[2])
        depth += np.linalg.norm(dir_line) / radiation_length
        curr_vec += dir_line # move down linreg line
        penetration += np.linalg.norm(dir_line)
        key = depth - (depth % bin_size) + bin_size
        E[key] = E[key] + plane_energy

    # remove placeholder zero E values
    trimmed_E = {K : E[K] for K in E if E[K] != 0}
    rad_lengths = list(trimmed_E.keys())
    event_energies = list(trimmed_E.values())

    return rad_lengths, event_energies

def proj(v, w):
    """returns the projection of v unto w, where v and w are vectors in R3. """
    return (np.dot(v, w) / np.dot(w, w)) * w

def vec(i, data):
    """returns vector for point at index (row) i in data (assuming data is formatted as in toDataSpace()"""

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    return np.array([x[i], y[i], z[i]])

def pointsWithinZ(z, e, data, line_vec):
    """returns all point indices within [z-e, z+e].
    points sorted by distance from line ascending
    keep in mind data is organized z descending"""
    pts = []
    distances = []
    i = 0
    v = vec(i, data)
    lower_bound, upper_bound = z - e, z + e
    while v[2] >= lower_bound and i < len(data) - 1:
        if v[2] <= upper_bound:
            pts.append(i)
            distances.append(np.linalg.norm(v - proj(v, line_vec)))
        i += 1
        v = vec(i, data)

    distances = np.array(distances)
    pts = np.array(pts)
    indices = np.argsort(distances)

    return pts[indices]

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

    try:
        ransac = RANSACRegressor()
        ransac.fit(upper_xy, upper_z)
    except ValueError as e:
        return None, None, None

    # filter rest of data via model
    z_pred = ransac.predict(xy)
    inliers = np.abs(z - z_pred) < 0.5 * np.std(z_pred)
    outliers = np.logical_not(inliers)

    return geoData[inliers], energyData[inliers], geoData[outliers]

