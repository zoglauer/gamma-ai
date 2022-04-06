import numpy as np
import os
import pickle
import argparse

from skspatial.objects import Line
from skspatial.objects import Points
from skspatial.plotting import plot_3d

MaxEvents = 100
t1 = -7
t2 = 7
filename = "EnergyLoss.10k.v1.data"

if filename != "":
    FileName = filename
if not os.path.exists(FileName):
    print("Error: The training data file does not exist: {}".format(FileName))
    sys.exit(0)

with open(FileName, "rb") as FileHandle:
   DataSets = pickle.load(FileHandle)

if len(DataSets) > MaxEvents:
  DataSets = DataSets[:MaxEvents]

this_data = []
this_point = []

for g in range(0, len(DataSets)):
    Event = DataSets[g]
    for h in range(0, Event.hits.shape[0]):
        X = Event.hits[h, 0]
        Y = Event.hits[h, 1]
        Z = Event.hits[h, 2]
        this_point = [X, Y, Z]
        this_data.append(this_point)

points = Points(this_data)
fit = Line.best_fit(points)
plot_3d(fit.plotter(t_1=t1, t_2=t2, c='k'), points.plotter(c='b', depthshade=False))
