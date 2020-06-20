###################################################################################################
#
# ComptonTrackIdentificationGNN_PyTorch.py
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
import torch
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
from CERN_GNN import GNNSegmentClassifier

from GraphRepresentation import GraphRepresentation
#from GraphVisualizer import GraphVisualizer

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
parser.add_argument('-g', '--geometry', default='$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup', help='Geometry with which the sim file was created')
parser.add_argument('-m', '--maxevents', default='10000', help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainingsplit', default='0.1', help='Testing-training split')
parser.add_argument('-b', '--batchsize', default='128', help='Batch size')

args = parser.parse_args()

if args.filename != "":
  FileName = args.filename

if args.geometry != "":
  GeometryFileName = args.geometry

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
  if Geometry.ScanSetupFile(M.MString(GeometryFileName)) == True:
    print("Geometry " + GeometryFileName + " loaded!")
  else:
    print("Unable to load geometry " + GeometryFileName + " - Aborting!")
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
        #Data.center()

        #if Data.hasHitsOutside(XMin, XMax, YMin, YMax, ZMin, ZMax) == False and Data.isOriginInside(XMin, XMax, YMin, YMax, ZMin, ZMax) == True:
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

# Class and functions for graph representations are now in GraphRepresentation.py
# Initialize class with "GraphRepresentation(event)" where "event" is an EventData object.
# Optional parameters:


###################################################################################################
# Step 5: Training the graph neural network
###################################################################################################


print("Info: Training the graph neural network...")

# Initialize vectorization of training data
max_train_hits, max_train_edges = 0, 0
training_data = []

for Batch in range(NTrainingBatches):
    for e in range(BatchSize):

        # Prepare graph for a set of simulated events (training)
        event = TrainingDataSets[Batch*BatchSize + e]
        graphRepresentation = GraphRepresentation.newGraphRepresentation(event)
        graphData = graphRepresentation.graphData
        A, Ro, Ri, X, y = graphData
        max_train_hits = max(max_train_hits, len(X))
        max_train_edges = max(max_train_edges, len(y))
        training_data.append([[X, Ri, Ro], y])

# Padding to maximum dimension
for i in range(len(training_data)):
    training_data[i][0][0] = \
        np.pad(training_data[i][0][0], [(0, max_train_hits - len(training_data[i][0][0])), (0, 0)], mode = 'constant')
    training_data[i][0][1] = \
        np.pad(training_data[i][0][1], [(0, max_train_hits - len(training_data[i][0][1])),
                                                             (0, max_train_edges - len(training_data[i][0][1][0]))], mode = 'constant')
    training_data[i][0][2] = \
        np.pad(training_data[i][0][2], [(0, max_train_hits - len(training_data[i][0][2])),
                                                             (0, max_train_edges - len(training_data[i][0][2][0]))], mode = 'constant')
    training_data[i][1] = \
        np.pad(training_data[i][1], [(0, max_train_edges - len(training_data[i][1]))], mode = 'constant')

# Initialize data loader in PyTorch
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size = BatchSize)

# Initialize model, loss function, and optimizer
model = GNNSegmentClassifier(input_dim = 4, hidden_dim = 64, n_iters = 4)
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

loss_history = []
n_epochs = 100
for i in range(n_epochs):
    for (x, y) in train_dataloader:
        counter, sum_loss = 0, 0

        # Train the model using PyTorch
        model.zero_grad()
        prediction = model(x)
        loss = loss_function(prediction, y)
        loss.backward()
        sum_loss += loss
        counter += 1
        optimizer.step()

    # Keep track of average loss over each epoch
    loss_history.append((sum_loss / (counter + 0.0)).item())
    print(f"Epoch {i + 1} completed...")

print("\nLoss History (Training Set):")
print(loss_history, "\n")

###################################################################################################
# Step 6: Evaluating the graph neural network
###################################################################################################
print("Info: Evaluating the graph neural network...")

# Initialize vectorization of testing data
max_test_hits, max_test_edges = 0, 0
testing_data = []

for Batch in range(NTestingBatches):
    for e in range(BatchSize):

        # Prepare graph for a set of simulated events (testing)
        event = TestingDataSets[Batch*BatchSize + e]
        graphRepresentation = GraphRepresentation.newGraphRepresentation(event)
        graphData = graphRepresentation.graphData
        A, Ro, Ri, X, y = graphData
        max_test_hits = max(max_test_hits, len(X))
        max_test_edges = max(max_test_edges, len(y))
        testing_data.append([[X, Ri, Ro], y])

# Padding to maximum dimension
for i in range(len(testing_data)):
    testing_data[i][0][0] = np.pad(testing_data[i][0][0], [(0, max_test_hits - len(testing_data[i][0][0])), (0, 0)])
    testing_data[i][0][1] = np.pad(testing_data[i][0][1], [(0, max_test_hits - len(testing_data[i][0][1])),
                                                             (0, max_test_edges - len(testing_data[i][0][1][0]))])
    testing_data[i][0][2] = np.pad(testing_data[i][0][2], [(0, max_test_hits - len(testing_data[i][0][2])),
                                                             (0, max_test_edges - len(testing_data[i][0][2][0]))])
    testing_data[i][1] = np.pad(testing_data[i][1], [(0, max_test_edges - len(testing_data[i][1]))], mode = 'constant')

# Initialize data loader in PyTorch
test_dataloader = torch.utils.data.DataLoader(testing_data, batch_size = BatchSize)

# Evaluate the model using PyTorch
test_history = []
for (x, y) in test_dataloader:
    with torch.no_grad():

        prediction = model(x)
        loss = loss_function(prediction, y)
        accuracy = sum((prediction > 0.5) == (y > 0.5)) / (BatchSize + 0.0)

        accuracy, precision, recall = 0, 0, 0
        for (pred, out) in zip(prediction, y):
            accuracy += sum((pred > 0.5) == (out > 0.5)) / (len(out) + 0.0)
            true_pos, false_pos = 0, 0
            for i in range(len(pred)):
                if pred[i] > 0.5:
                    if out[i] == 1:
                        true_pos += 1
                    else:
                        false_pos += 1
            if true_pos + false_pos == 0:
                precision += 0.0
            else:
                precision += true_pos / (true_pos + false_pos)
            recall += true_pos / sum(out)

        accuracy = accuracy / (BatchSize + 0.0)
        precision = precision / (BatchSize + 0.0)
        recall = recall / (BatchSize + 0.0)

        test_history.append((loss.item(), accuracy.item(), precision, recall.item()))

# TODO: Figure out way to add predictions to graph representation
print("\nLoss, Accuracy, Precision, and Recall (Testing Set):")
print(test_history)

#input("Press [enter] to EXIT")
sys.exit(0)
