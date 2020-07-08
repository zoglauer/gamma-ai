###################################################################################################
#
# ComptonTrackIdentificationGNN.py
#
# Copyright (C) by Andreas Zoglauer & Pranav Nagarajan
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################



###################################################################################################
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Disabling GPU for testing CPU usage
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
import argparse
from datetime import datetime
from functools import reduce
from GraphRepresentation import GraphRepresentation

import time as t

print("\nCompton Track Identification")
print("============================\n")



# Step 1: Input parameters
###################################################################################################


# Default parameters

UseToyModel = False

TestingTrainingSplit = 0.1

epochs = 100

MaxEvents = 100000

OutputDirectory = "Results" + os.path.sep

# setup for debugging toy model, does nothing if False
ToyTest = False
if ToyTest:
    UseToyModel = True
    epochs = 1
#
Tuning = False

parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='ComptonTrackIdentification.p1.sim.gz', help='File name used for training/testing')
parser.add_argument('-g', '--geometry', default='$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup', help='Geometry with which the sim file was created')
parser.add_argument('-m', '--maxevents', default='10000', help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainingsplit', default='0.1', help='Testing-training split')
parser.add_argument('-b', '--batchsize', default='128', help='Batch size')
parser.add_argument('-e', '--epochs', default='100', help='Epochs')
parser.add_argument('-t', '--tuning', default='0', help='Hyperparameter tuning mode')

args = parser.parse_args()

if args.filename != "":
  FileName = args.filename

if args.geometry != "":
  GeometryName = args.geometry

if int(args.maxevents) >= 500:
  MaxEvents = int(args.maxevents)

if int(args.batchsize) >= 16:
  BatchSize = int(args.batchsize)

if float(args.testingtrainingsplit) >= 0.05:
   TestingTrainingSplit = float(args.testingtrainingsplit)

if args.tuning != "" and args.tuning != "0":
    Tuning = True

if args.epochs != "":
    epochs = int(args.epochs)

if os.path.exists(OutputDirectory):
  Now = datetime.now()
  OutputDirectory += Now.strftime("%Y%m%d_%H%M%S")

os.makedirs(OutputDirectory)

print(MaxEvents)

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

#
start = t.time()

# Read the simulation file data:
DataSets = []
NumberOfDataSets = 0
NumberOfEvents = 0

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
    M.SetOwnership(Event, True) # Python need owner shiop of the event in order to delete it
    NumberOfEvents += 1

    if Event.GetNIAs() > 0:
      Data = EventData()
      if Data.parse(Event) == True:
        # Data.center()

        # if Data.hasHitsOutside(XMin, XMax, YMin, YMax, ZMin, ZMax) == False and Data.isOriginInside(XMin, XMax, YMin, YMax, ZMin, ZMax) == True:
          DataSets.append(Data)
          NumberOfDataSets += 1

          if NumberOfDataSets > 0 and NumberOfDataSets % 1000 == 0:
              print("Data sets processed: {} (out of {} read events)".format(NumberOfDataSets, NumberOfEvents))

    if NumberOfDataSets >= MaxEvents:
      break
    
    if Interrupted == True:
      Interrupted = False
      NInterrupts -= 1
      break


print("Info: Parsed {} events".format(NumberOfDataSets))
print("Info: Time Elapsed for Data Loading: {}".format(t.time() - start))


#
start = t.time()

# Split the data sets in training and testing data sets

# The number of available batches in the input data
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
print("Info: Time Elapsed for Train/Test Split: {}".format(t.time() - start))



###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################


print("Info: Setting up the graph neural network...")

# Definition of edge network (calculates edge weights)
def EdgeNetwork(H, Ri, Ro, input_dim, hidden_dim):

    def create_B(H):
        bo = tf.transpose(Ro, perm = [0, 2, 1]) @ H
        bi = tf.transpose(Ri, perm = [0, 2, 1]) @ H
        B = tf.keras.layers.concatenate([bo, bi])
        return B

    B = tf.keras.layers.Lambda(lambda H: create_B(H))(H)
    layer_2 = tf.keras.layers.Dense(hidden_dim, activation = "tanh")(B)
    layer_3 = tf.keras.layers.Dense(1, activation = "sigmoid")(layer_2)

    return tf.squeeze(layer_3, axis = -1)


# Definition of node network (computes states of nodes)
def NodeNetwork(H, Ri, Ro, edge_weights, input_dim, output_dim):

    def create_M(e):
        bo = tf.transpose(Ro, perm = [0, 2, 1]) @ H
        bi = tf.transpose(Ri, perm = [0, 2, 1]) @ H
        Rwo = Ro * e[:, None]
        Rwi = Ri * e[:, None]
        mi = Rwi @ bo
        mo = Rwo @ bi
        M = tf.keras.layers.concatenate([mi, mo, H])
        return M

    M = tf.keras.layers.Lambda(lambda e: create_M(e))(edge_weights)
    layer_4 = tf.keras.layers.Dense(output_dim, activation = "tanh")(M)
    layer_5 = tf.keras.layers.Dense(output_dim, activation = "tanh")(layer_4)

    return layer_5


# Definition of overall network (iterates to find most probable edges)
def SegmentClassifier(input_dim = 4, hidden_dim = 64, num_iters = 5):

    # PLaceholders for association matrices and data matrix
    X = tf.keras.layers.Input(shape = (None, input_dim))
    Ri = tf.keras.layers.Input(shape = (None, None))
    Ro = tf.keras.layers.Input(shape = (None, None))

    # Application of input network (creates latent representation of graph)
    H = tf.keras.layers.Dense(hidden_dim, activation = "tanh")(X)
    H = tf.keras.layers.concatenate([H, X])

    # Application of graph neural network (generates probabilities for each edge)
    for i in range(num_iters):
        edge_weights = EdgeNetwork(H, Ri, Ro, input_dim + hidden_dim, hidden_dim)
        H = NodeNetwork(H, Ri, Ro, edge_weights, input_dim + hidden_dim, hidden_dim)
        H = tf.keras.layers.concatenate([H, X])

    output_layer = EdgeNetwork(H, Ri, Ro, input_dim + hidden_dim, hidden_dim)

    # Creation and compilation of model
    model = tf.keras.models.Model(inputs = [X, Ri, Ro], outputs = output_layer)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                  metrics = ['accuracy', tf.keras.metrics.Precision(thresholds = 0.55),
                             tf.keras.metrics.Recall(thresholds = 0.55)])
    print(model.summary())

    return model


###################################################################################################
# Step 5: Training the graph neural network
###################################################################################################

print("Info: Training the graph neural network...")

model = SegmentClassifier()

def data_generator():
    ct = 0
    while True:
        #
        start = t.time()
        ct += 1

        random_batch = np.random.randint(0, NTrainingBatches - 1)

        # Initialize vectorization of training data
        max_train_hits, max_train_edges = 0, 0
        train_X = []
        train_Ri = []
        train_Ro = []
        train_y = []

        for e in range(BatchSize):

            # Prepare graph for a set of simulated events (training)
            event = TrainingDataSets[random_batch*BatchSize + e]
            graphRepresentation = GraphRepresentation.newGraphRepresentation(event)
            graphData = graphRepresentation.graphData
            A, Ro, Ri, X, y = graphData
            max_train_hits = max(max_train_hits, len(X))
            max_train_edges = max(max_train_edges, len(y))
            train_X.append(X)
            train_Ri.append(Ri)
            train_Ro.append(Ro)
            train_y.append(y)

        # Padding to maximum dimension
        for i in range(len(train_X)):
            train_X[i] = np.pad(train_X[i], [(0, max_train_hits - len(train_X[i])), (0, 0)], mode = 'constant')
            train_Ri[i] = np.pad(train_Ri[i], [(0, max_train_hits - len(train_Ri[i])), (0, max_train_edges - len(train_Ri[i][0]))], mode = 'constant')
            train_Ro[i] = np.pad(train_Ro[i], [(0, max_train_hits - len(train_Ro[i])), (0, max_train_edges - len(train_Ro[i][0]))], mode = 'constant')
            train_y[i] = np.pad(train_y[i], [(0, max_train_edges - len(train_y[i]))], mode = 'constant')

        print("Info: Time Elapsed for Data Processing {}: {}".format(ct, t.time() - start))

        yield ([train_X, train_Ri, train_Ro], np.array(train_y))


model_start = t.time()
model.fit(data_generator(), steps_per_epoch = NTrainingBatches, epochs = epochs)
print("Info: Total Time Elapsed for Training: {}".format(t.time() - model_start))

###################################################################################################
# Step 6: Evaluating the graph neural network
###################################################################################################

print("Info: Evaluating the graph neural network...")

# Initialize vectorization of testing data
max_test_hits, max_test_edges = 0, 0
test_X = []
test_Ri = []
test_Ro = []
test_y = []
test_rep = []

for Batch in range(NTestingBatches):
    #
    start = t.time()
    for e in range(BatchSize):

        # Prepare graph for a set of simulated events (testing)
        event = TestingDataSets[Batch*BatchSize + e]
        graphRepresentation = GraphRepresentation.newGraphRepresentation(event)
        graphData = graphRepresentation.graphData
        A, Ro, Ri, X, y = graphData
        max_test_hits = max(max_test_hits, len(X))
        max_test_edges = max(max_test_edges, len(y))
        test_X.append(X)
        test_Ri.append(Ri)
        test_Ro.append(Ro)
        test_y.append(y)
        test_rep.append(graphRepresentation)

    print("Info: Time Elapsed for Test Batch {}: {}".format(Batch, t.time() - start))

#
start = t.time()

# Padding to maximum dimension
for i in range(len(test_X)):
    test_X[i] = np.pad(test_X[i], [(0, max_test_hits - len(test_X[i])), (0, 0)], mode = 'constant')
    test_Ri[i] = np.pad(test_Ri[i], [(0, max_test_hits - len(test_Ri[i])), (0, max_test_edges - len(test_Ri[i][0]))], mode = 'constant')
    test_Ro[i] = np.pad(test_Ro[i], [(0, max_test_hits - len(test_Ro[i])), (0, max_test_edges - len(test_Ro[i][0]))], mode = 'constant')
    test_y[i] = np.pad(test_y[i], [(0, max_test_edges - len(test_y[i]))], mode = 'constant')

print("Info: Time Elapsed for Test Padding: {}".format(t.time() - start))

# Generate predictions for a graph

#
start = t.time()

predictions = model.predict([test_X, test_Ri, test_Ro], batch_size = BatchSize)

print("Info: Time Elapsed for Prediction: {}".format(t.time() - start))

test_graph = test_rep[0]
test_graph.add_prediction(predictions[0])
test_graph.visualize_last_prediction()

# for i in range(len(predictions)):
#     test_rep[i].add_prediction(predictions[i])
#     test_rep[i].visualize_last_prediction()


#
start = t.time()

model.evaluate([test_X, test_Ri, test_Ro], np.array(test_y), batch_size = BatchSize)

print("Info: Time Elapsed for Evaluation: {}".format(t.time() - start))
