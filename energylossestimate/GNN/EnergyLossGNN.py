###################################################################################################
#
# EnergyLossGNN.py
#
# Copyright (C) by Andreas Zoglauer & Pranav Nagarajan & Rithwik Sudharsan
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################



###################################################################################################
import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

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

import pickle

import time as t

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten


print("\nEnergy Loss Estimation - GNN")
print("============================\n")



# Step 1: Input parameters
###################################################################################################


# Default parameters

UseToyModel = False

TestingTrainingSplit = 0.1

epochs = 100

MaxEvents = 500000

OutputDirectory = "Results" + os.path.sep

ExtractEvents = False
ExtractFileName = ""

# setup for debugging toy model, does nothing if False
ToyTest = False
#

Tuning = False

parser = argparse.ArgumentParser(description='Perform training and/or testing of the event clustering machine learning tools.')
parser.add_argument('-f', '--filename', default='/volumes/selene/users/rithwik/2MeV_5GeV_flat.inc1.id1.sim.gz', help='File name used for training/testing')
parser.add_argument('-g', '--geometry', default='$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup', help='Geometry with which the sim file was created')
parser.add_argument('-m', '--maxevents', default='500000', help='Maximum number of events to use')
parser.add_argument('-s', '--testingtrainingsplit', default='0.1', help='Testing-training split')
parser.add_argument('-b', '--batchsize', default='128', help='Batch size')
parser.add_argument('-e', '--epochs', default='100', help='Epochs')
parser.add_argument('-p', '--tuning', default='False', help='Hyperparameter tuning mode')
parser.add_argument('-a', '--acceptance', default='egpb', help='Which track types to accept: e:reject all EVENTS with electron track, g: reject all HITS with JUST gamma ray interaction, p: reject all EVENTS with positron track, b: reject all events with bremsstrahlung hits')
parser.add_argument('-t', '--testing', default='False', help='Toy testing mode')
parser.add_argument('-x', '--extract', default='', help='Only create an extracted sim file --- to speed up later runs.')
parser.add_argument('-d', '--save', default='True', help='Save results in a text file, in output dir.')
parser.add_argument('-v', '--viz', default='0.5', help='Edge visualization threshold.')

args = parser.parse_args()

if args.filename != "":
  FileName = args.filename

if args.geometry != "":
  GeometryName = args.geometry

if args.viz != "":
  viz_threshold = float(args.viz)
else:
    viz_threshold = 0.5

if int(args.maxevents) >= 0:
  MaxEvents = int(args.maxevents)

if int(args.batchsize) >= 16:
  BatchSize = int(args.batchsize)

if float(args.testingtrainingsplit) >= 0.05:
   TestingTrainingSplit = float(args.testingtrainingsplit)

if args.tuning == "True":
    Tuning = True

Acceptance = args.acceptance

if args.epochs != "":
    epochs = int(args.epochs)


if args.testing == "True":
    ToyTest = True

if ToyTest:
    UseToyModel = True
    if int(args.epochs) == 100:
        epochs = 5
    MaxEvents = 1000

Save = True
if args.save != "":
    Save = False


if args.extract != "":
  ExtractEvents = True
  ExtractFileName = args.extract

if os.path.exists(OutputDirectory):
  Now = datetime.now()
  OutputDirectory += Now.strftime("%Y_%m_%d_%H.%M.%S")

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

#
start = t.time()

# Read the simulation file data:
DataSets = []
NumberOfDataSets = 0
NumberOfEvents = 0

loadData = True
if os.path.exists('/volumes/selene/users/rithwik/gnn.data'):
    with open('/volumes/selene/users/rithwik/gnn.data', 'rb') as filehandle:
          DataSets = pickle.load(filehandle)
    NumberOfDataSets = len(DataSets)
    print("Loaded from Pickle! {} Events".format(NumberOfDataSets))
    if NumberOfDataSets >= MaxEvents:
      DataSets = DataSets[:MaxEvents]
      loadData = False

if loadData:
  DataSets = []
  NumberOfDataSets = 0
  NumberOfEvents = 0
  # Load geometry:
  @profile
  def dataLoader(NumberOfDataSets, NumberOfEvents, Interrupted):
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

    Writer = M.MFileEventsSim(Geometry)
    if ExtractEvents == True:
      if Writer.Open(M.MString(ExtractFileName), M.MFile.c_Write) == False:
        #print("Unable to open file " + FileName + ". Aborting!")
        quit()

      Writer.SetGeometryFileName(M.MString(GeometryName));
      Writer.SetVersion(25);
      Writer.WriteHeader();

    print("\n\nStarted reading data sets")
    pbar = tqdm(total=MaxEvents)
    while True:
      #print(" Event:   {}".format(NumberOfDataSets), end='\r')
      Event = Reader.GetNextEvent()
      if not Event:
        break
      M.SetOwnership(Event, True) # Python needs ownership of the event in order to delete it
      NumberOfEvents += 1

      if Event.GetNIAs() > 0:
        Data = EventData()
        # Data.setAcceptance(Acceptance)

        if Data.parse(Event) == True:
          # Data.center()
          # if Data.hasHitsOutside(XMin, XMax, YMin, YMax, ZMin, ZMax) == False and Data.isOriginInside(XMin, XMax, YMin, YMax, ZMin, ZMax) == True:
            DataSets.append(Data)
            NumberOfDataSets += 1
            pbar.update(1)

            if NumberOfDataSets > 0 and NumberOfDataSets % 1000 == 0:
              print("Data sets processed: {} (out of {} read events)".format(NumberOfDataSets, NumberOfEvents))

            if ExtractEvents == True:
              EventForWriter = Event.ToSimString()
              Writer.AddText(EventForWriter)

      if NumberOfDataSets >= MaxEvents:
        break

      if Interrupted == True:
        Interrupted = False
        NInterrupts -= 1
        break

    if ExtractEvents == True:
      Writer.CloseEventList();
      Writer.Close();
      quit()

    pbar.close()

    return NumberOfDataSets, NumberOfEvents, Interrupted
  
  NumberOfDataSets, NumberOfEvents, Interrupted = dataLoader(NumberOfDataSets, NumberOfEvents, Interrupted)

  if NumberOfDataSets == MaxEvents:
    with open('/volumes/selene/users/rithwik/gnn.data', 'wb') as filehandle:
          pickle.dump(DataSets, filehandle)
          print("Saved to pickle! {} Events".format(len(DataSets)))

  
print("Info: Parsed {} events".format(NumberOfDataSets))
dataload_time = t.time() - start
print("Time to load data: {}".format(dataload_time))

##### Stellar graph library usage begins here.
#
start = t.time()

# Split the data sets in training and testing data sets

# The number of available batches in the input data
NBatches = int(len(DataSets) / BatchSize)
if NBatches < 2:
  print("Not enough data!")
  quit()


X = []
y = []

for event in DataSets:
  graphRep = GraphRepresentation(event)
  X.append(graphRep.stellar_graph)
  y.append(graphRep.gamma_energy)

#X = pd.DataFrame(X)
y = pd.DataFrame(y)

generator = PaddedGraphGenerator(graphs=X)

train_graphs, test_graphs = \
  model_selection.train_test_split(y, train_size=TestingTrainingSplit)

NumberOfTrainingEvents = TestingTrainingSplit * len(DataSets)
NumberOfTestingEvents = len(DataSets) - NumberOfTrainingEvents

print("Info: Number of training data sets: {}   Number of testing data sets: {} (vs. input: {} and split ratio: {})".format(NumberOfTrainingEvents, NumberOfTestingEvents, len(DataSets), TestingTrainingSplit))
traintestsplit_time = t.time() - start



###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################


print("Info: Setting up the graph neural network...")

k = 35  # the number of rows for the output tensor
layer_sizes = [32, 32, 32, 1]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)

model.compile(
    optimizer=Adam(lr=0.0001), loss="mean_squared_error", metrics=["acc"],
)

gen = PaddedGraphGenerator(graphs=X)

train_gen = gen.flow(
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=50,
    symmetric_normalization=False,
)

test_gen = gen.flow(
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)


###################################################################################################
# Step 5: Training the graph neural network
###################################################################################################

print("Info: Training the graph neural network...")

history = model.fit(
    train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
)