###################################################################################################
#
# ComptonTrackingGNN.py
#
# Copyright (C) by Andreas Zoglauer & Pranav Nagarajan
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################



###################################################################################################

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

import random

import signal
import sys
import time
import math
import csv
import os
import argparse
from datetime import datetime
from functools import reduce

print("\nCompton Track Identification")
print("============================\n")



# Step 1: Input parameters
###################################################################################################


# Default parameters

UseToyModel = True

# Split between training and testing data
TestingTrainingSplit = 0.1

MaxEvents = 100000



OutputDirectory = "Results"

parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='ComptonTrackIdentification.p1.sim.gz', help='File name used for training/testing')
parser.add_argument('-m', '--maxevents', default='10000', help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainingsplit', default='0.1', help='Testing-training split')
parser.add_argument('-b', '--batchsize', default='128', help='Batch size')

args = parser.parse_args()

if args.filename != "":
  FileName = args.filename

if int(args.maxevents) > 1000:
  MaxEvents = int(args.maxevents)

if int(args.batchsize) >= 16:
  BatchSize = int(args.batchsize)

if float(args.testingtrainingsplit) >= 0.05:
   TestingTrainingSplit = float(args.testingtrainingsplit)



if os.path.exists(OutputDirectory):
  Now = datetime.now()
  OutputDirectory += Now.strftime("_%Y%m%d_%H%M%S")

os.makedirs(OutputDirectory)



###################################################################################################
# Step 2: Global functions
###################################################################################################


# Take care of Ctrl-C
Interrupted = False
NInterrupts = 0
def signal_handler(signal, frame):
  global Interrupted
  Interrupted = True
  global NInterrupts
  NInterrupts += 1
  if NInterrupts >= 2:
    print("Aborting!")
    sys.exit(0)
  print("You pressed Ctrl+C - waiting for graceful abort, or press  Ctrl-C again, for quick exit.")
signal.signal(signal.SIGINT, signal_handler)


# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from EventData import EventData

# Load MEGAlib into ROOT so that it is usable
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
M.PyConfig.IgnoreCommandLineOptions = True



###################################################################################################
# Step 3: Create some training, test & verification data sets
###################################################################################################


# Read the simulation file data:
DataSets = []
NumberOfDataSets = 0

if UseToyModel == True:
  for e in range(0, MaxEvents):
    Data = EventData()
    Data.createFromToyModel(e)
    DataSets.append(Data)

    NumberOfDataSets += 1
    if NumberOfDataSets > 0 and NumberOfDataSets % 1000 == 0:
      print("Data sets processed: {}".format(NumberOfDataSets))

else:
  # Load geometry:
  Geometry = M.MDGeometryQuest()
  if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
    print("Geometry " + GeometryName + " loaded!")
  else:
    print("Unable to load geometry " + GeometryName + " - Aborting!")
    quit()


  Reader = M.MFileEventsSim(Geometry)
  if Reader.Open(M.MString(FileName)) == False:
    print("Unable to open file " + FileName + ". Aborting!")
    quit()


  print("\n\nStarted reading data sets")
  while True:
    Event = Reader.GetNextEvent()
    if not Event:
      break

    if Event.GetNIAs() > 0:
      Data = EventData()
      if Data.parse(Event) == True:
        Data.center()

        if Data.hasHitsOutside(XMin, XMax, YMin, YMax, ZMin, ZMax) == False and Data.isOriginInside(XMin, XMax, YMin, YMax, ZMin, ZMax) == True:
          DataSets.append(Data)
          NumberOfDataSets += 1

          if NumberOfDataSets > 0 and NumberOfDataSets % 1000 == 0:
            print("Data sets processed: {}".format(NumberOfDataSets))

    if NumberOfDataSets >= MaxEvents:
      break


print("Info: Parsed {} events".format(NumberOfDataSets))



# Split the data sets in training and testing data sets

# The number of available batches in the inoput data
NBatches = int(len(DataSets) / BatchSize)
if NBatches < 2:
  print("Not enough data!")
  quit()

# Split the batches in training and testing according to TestingTrainingSplit
NTestingBatches = int(NBatches*TestingTrainingSplit)
if NTestingBatches == 0:
  NTestingBatches = 1
NTrainingBatches = NBatches - NTestingBatches

# Now split the actual data:
TrainingDataSets = []
for i in range(0, NTrainingBatches * BatchSize):
  TrainingDataSets.append(DataSets[i])


TestingDataSets = []
for i in range(0,NTestingBatches*BatchSize):
   TestingDataSets.append(DataSets[NTrainingBatches * BatchSize + i])


NumberOfTrainingEvents = len(TrainingDataSets)
NumberOfTestingEvents = len(TestingDataSets)

print(np.unique(np.array([event.unique for event in TestingDataSets])))

print("Info: Number of training data sets: {}   Number of testing data sets: {} (vs. input: {} and split ratio: {})".format(NumberOfTrainingEvents, NumberOfTestingEvents, len(DataSets), TestingTrainingSplit))




###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################


print("Info: Setting up the graph neural network...")

# Criterion for choosing to connect two nodes
radius = 25

# Checking if distance is within criterion
def distanceCheck(h1, h2):
    dist = np.sqrt(np.sum((h1 - h2)**2))
    return dist <= radius

# Utility function for keeping non-zero rows
def filterZero(tensor, association = False):
    if association:
        zero_vector = tf.constant([2], dtype = tensor.dtype)
        reduced_tensor = tf.reduce_min(tensor, 1)
    else:
        zero_vector = tf.zeros(shape=(1,1), dtype = tensor.dtype)
        reduced_tensor = tf.reduce_sum(tensor, 1)
    mask = tf.squeeze(tf.not_equal(reduced_tensor, zero_vector))
    nonzero_rows = tf.boolean_mask(tensor, mask)
    return nonzero_rows

# Utility function for removing padding
def removePad(tensor, association = False):
    rotated_tensor = tf.transpose(tensor)
    nonzero_cols = filterZero(rotated_tensor, association)
    rotated_tensor = tf.transpose(nonzero_cols)
    nonzero_rows = filterZero(rotated_tensor, association)
    return nonzero_rows

# Creates the graph representation for the detector
def CreateGraph(event, pad_size):

    A = np.zeros((len(event.X), len(event.X)))

    # Parse the event data
    assert len(event.X) == len(event.Y) \
           == len(event.Z) == len(event.E) \
           == len(event.Type) == len(event.Origin), "Event Data size mismatch."
    data = np.array(list(zip(event.X, event.Y, event.Z, event.E, event.Type, event.Origin)))
    hits = data[:, :3].astype(np.float)
    energies = data[:, 3].astype(np.float)
    types = data[:, 4]
    origins = data[:, 5].astype(np.int)

    # Fill in the adjacency matrix
    for i in range(len(hits)):
        for j in range(i+1, len(hits)):
            gamma_bool = (types[i] == 'g' and types[j] == 'g')
            compton_bool = (types[j] == 'eg' and origins[j] == 1)
            if gamma_bool or compton_bool or distanceCheck(hits[i], hits[j]):
                A[i][j] = A[j][i] = 1

    # Create the incoming matrix, outgoing matrix, and matrix of labels
    num_edges = int(np.sum(A))
    Ro = np.zeros((len(hits), num_edges))
    Ri = np.zeros((len(hits), num_edges))
    y = np.zeros(pad_size)
    # y = np.zeros((len(hits), len(hits)))

    # Fill in the incoming matrix, outgoing matrix, and matrix of labels
    counter = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j]:
                Ro[i, np.arange(num_edges)] = 1
                Ri[j, np.arange(num_edges)] = 1
                if i + 1 == origins[j]:
                    #y[i][j] = 1
                    y[counter] = 1
                counter += 1

    # Generate feature matrix of nodes
    X = data[:, :4].astype(np.float)

    # Fill in dictionary of node labels and positions
    nodes = {}
    for i in range(len(y)):
        nodes[i] = [i, i**2]

    # Visualization of graph of true edges
    # G = nx.from_numpy_matrix(y, create_using = nx.DiGraph)
    # nx.draw_networkx(G = G, pos = nodes, arrows = True, with_labels = True)
    # plt.show()

    # Padding to maximum dimension
    A = np.pad(A, [(0, pad_size - len(A)), (0, pad_size - len(A[0]))])
    Ro = np.pad(Ro, [(0, pad_size - len(Ro)), (0, pad_size - len(Ro[0]))], constant_values = 2)
    Ri = np.pad(Ri, [(0, pad_size - len(Ri)), (0, pad_size - len(Ri[0]))], constant_values = 2)
    X = np.pad(X, [(0, pad_size - len(X)), (0, 0)])
    #y = np.pad(y, [(0, pad_size - len(y)), (0, pad_size - len(y[0]))])

    return [A, Ro, Ri, X, y]


# Definition of edge network (calculates edge weights)
def EdgeNetwork(H, Ro, Ri, input_dim, hidden_dim):

    def create_B(H):
        bo = tf.transpose(Ro) @ H
        bi = tf.transpose(Ri) @ H
        B = tf.keras.layers.concatenate([bo, bi])
        return B

    B = tf.keras.layers.Lambda(lambda H: create_B(H))(H)
    layer_2 = tf.keras.layers.Dense(hidden_dim, activation = "tanh")(B)
    layer_3 = tf.keras.layers.Dense(1, activation = "sigmoid")(layer_2)

    return layer_3


# Definition of node network (computes states of nodes)
def NodeNetwork(H, Ro, Ri, edge_weights, input_dim, output_dim):

    def create_M(e):
        bo = tf.transpose(Ro) @ H
        bi = tf.transpose(Ri) @ H
        Rwo = Ro * tf.transpose(e)
        Rwi = Ri * tf.transpose(e)
        mi = Rwi @ bo
        mo = Rwo @ bi
        M = tf.keras.layers.concatenate([mi, mo, H])
        return M

    M = tf.keras.layers.Lambda(lambda e: create_M(e))(edge_weights)
    layer_4 = tf.keras.layers.Dense(output_dim, activation = "tanh")(M)
    layer_5 = tf.keras.layers.Dense(output_dim, activation = "tanh")(layer_4)

    return layer_5


# Definition of overall network (iterates to find most probable edges)
def SegmentClassifier(pad_size, input_dim = 4, hidden_dim = 16, num_iters = 3):

    # PLaceholders for association matrices and data matrix
    A = tf.keras.layers.Input(batch_shape = (pad_size, pad_size))
    Ro = tf.keras.layers.Input(batch_shape = (pad_size, pad_size))
    Ri = tf.keras.layers.Input(batch_shape = (pad_size, pad_size))
    X = tf.keras.layers.Input(batch_shape = (pad_size, input_dim))

    # Remove padding from input matrices
    #A_new = removePad(A)
    #Ro_new = removePad(Ro, True)
    #Ri_new = removePad(Ri, True)
    #X_new = tf.reshape(removePad(X), [-1, 4])

    # Application of input network (creates latent representation of graph)
    H = tf.keras.layers.Dense(hidden_dim, activation = "tanh")(X)
    H = tf.keras.layers.concatenate([H, X])

    # Application of graph neural network (generates probabilities for each edge)
    for i in range(num_iters):
        edge_weights = EdgeNetwork(H, Ro, Ri, input_dim + hidden_dim, hidden_dim)
        H = NodeNetwork(H, Ro, Ri, edge_weights, input_dim + hidden_dim, hidden_dim)
        H = tf.keras.layers.concatenate([H, X])

    output_layer = EdgeNetwork(H, Ro, Ri, input_dim + hidden_dim, hidden_dim)

    # Fill in adjacency matrix with probabilities
    # zero = tf.constant(0, dtype = A.dtype)
    # adjacency = tf.keras.backend.flatten(A)
    # indices = tf.cast(tf.where(tf.not_equal(adjacency, zero)), tf.int32)
    # updated = tf.scatter_nd(indices, output_layer, [pad_size*pad_size, 1])
    # output = tf.reshape(updated, [pad_size, pad_size])

    # Creation and compilation of model
    model = tf.keras.models.Model(inputs = [A, Ro, Ri, X], outputs = output_layer)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model.summary())

    return model


###################################################################################################
# Step 5: Training and evaluating the network
###################################################################################################


print("Info: Training and evaluating the network - to be written")

pad_size = 100
model = SegmentClassifier(pad_size)

for Batch in range(NTrainingBatches):
    for e in range(BatchSize):

        # Prepare graph for a set of simulated events (training)
        event = TrainingDataSets[Batch*BatchSize + e]
        A, Ro, Ri, X, y = CreateGraph(event, pad_size)

        # Fit the model to the data
        model.fit([A, Ro, Ri, X], y)

# for Batch in range(NTestingBatches):
#    for e in range(BatchSize):

       # Prepare graph for a set of simulated events (testing)
#       event = TestingDataSets[Batch*BatchSize + e]
#       A, Ro, Ri, X, y = CreateGraph(event, pad_size)

       # Generate predictions for a graph
#       predicted_edge_weights = model.predict([A, Ro, Ri, X])
#       print(predicted_edge_weights)


#input("Press [enter] to EXIT")
sys.exit(0)
