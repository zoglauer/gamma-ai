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
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

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

import time as t

print("\nCompton Track Identification")
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
parser.add_argument('-f', '--filename', default='ComptonTrackIdentification_LowEnergy.p1.sim', help='File name used for training/testing')
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

if int(args.maxevents) >= 500:
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

  Writer = M.MFileEventsSim(Geometry)
  if ExtractEvents == True:
    if Writer.Open(M.MString(ExtractFileName), M.MFile.c_Write) == False:
      #print("Unable to open file " + FileName + ". Aborting!")
      quit()

    Writer.SetGeometryFileName(M.MString(GeometryName));
    Writer.SetVersion(25);
    Writer.WriteHeader();



  print("\n\nStarted reading data sets")
  while True:
    Event = Reader.GetNextEvent()
    if not Event:
      break
    M.SetOwnership(Event, True) # Python needs ownership of the event in order to delete it
    NumberOfEvents += 1

    if Event.GetNIAs() > 0:
      Data = EventData()
      Data.setAcceptance(Acceptance)

      EventForWriter = Event.ToSimString()

      if Data.parse(Event) == True:
        # Data.center()

        # if Data.hasHitsOutside(XMin, XMax, YMin, YMax, ZMin, ZMax) == False and Data.isOriginInside(XMin, XMax, YMin, YMax, ZMin, ZMax) == True:
          DataSets.append(Data)
          NumberOfDataSets += 1

          if NumberOfDataSets > 0 and NumberOfDataSets % 1000 == 0:
            print("Data sets processed: {} (out of {} read events)".format(NumberOfDataSets, NumberOfEvents))

          if ExtractEvents == True:
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

print("Info: Parsed {} events".format(NumberOfDataSets))
dataload_time = t.time() - start



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
traintestsplit_time = t.time() - start



###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################


print("Info: Setting up the graph neural network...")

# Definition of edge network (calculates edge weights)
def EdgeNetwork(H, Ri, Ro, input_dim, hidden_dim):

    def create_B(H):
        # Note: In numpy transposes are memory-efficient constant time operations as they simply return
        # a new view of the same data with adjusted strides. TensorFlow does not support strides,
        # so transpose returns a new tensor with the items permuted.
        bo = tf.transpose(a=Ro, perm = [0, 2, 1]) @ H
        bi = tf.transpose(a=Ri, perm = [0, 2, 1]) @ H
        B = tf.keras.layers.concatenate([bo, bi])
        return B

    B = tf.keras.layers.Lambda(create_B)(H)
    layer_2 = tf.keras.layers.Dense(hidden_dim, activation = "tanh")(B)
    layer_3 = tf.keras.layers.Dense(1, activation = "sigmoid")(layer_2)

    return tf.squeeze(layer_3, axis = -1)


# Definition of node network (computes states of nodes)
def NodeNetwork(H, Ri, Ro, edge_weights, input_dim, output_dim):

    def create_M(e):
        bo = tf.transpose(a=Ro, perm = [0, 2, 1]) @ H
        bi = tf.transpose(a=Ri, perm = [0, 2, 1]) @ H
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
    X = tf.keras.Input(shape = (None, input_dim))
    Ri = tf.keras.Input(shape = (None, None))
    Ro = tf.keras.Input(shape = (None, None))

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
                  metrics = ['accuracy', tf.keras.metrics.Precision(thresholds = 0.4),
                             tf.keras.metrics.Recall(thresholds = 0.4)])
    print(model.summary())

    return model


###################################################################################################
# Step 5: Training the graph neural network
###################################################################################################

print("Info: Training the graph neural network...")

model = SegmentClassifier()

datagen_time = 0
pad_time = 0

def data_generator():
    while True:
        start = t.time()

        random_batch = np.random.randint(0, NTrainingBatches - 1)

        # Initialize vectorization of training data
        max_train_hits, max_train_edges = 0, 0
        train_X = []
        train_Ri = []
        train_Ro = []
        train_y = []

        for e in range(BatchSize):

            # Prepare graph for a set of simulated events (training)
            event = TrainingDataSets[random_batch * BatchSize + e]
            graphRepresentation = GraphRepresentation.newGraphRepresentation(event, threshold=viz_threshold)
            graphData = graphRepresentation.graphData
            A, Ro, Ri, X, y = graphData
            max_train_hits = max(max_train_hits, len(X))
            max_train_edges = max(max_train_edges, len(y))
            train_X.append(X)
            train_Ri.append(Ri)
            train_Ro.append(Ro)
            train_y.append(y)

        global datagen_time
        datagen_time += (t.time() - start)
        #
        start = t.time()

        # Padding to maximum dimension
        for i in range(len(train_X)):
            train_X[i] = np.pad(train_X[i], [(0, max_train_hits - len(train_X[i])), (0, 0)], mode = 'constant')
            train_Ri[i] = np.pad(train_Ri[i], [(0, max_train_hits - len(train_Ri[i])), (0, max_train_edges - len(train_Ri[i][0]))], mode = 'constant')
            train_Ro[i] = np.pad(train_Ro[i], [(0, max_train_hits - len(train_Ro[i])), (0, max_train_edges - len(train_Ro[i][0]))], mode = 'constant')
            train_y[i] = np.pad(train_y[i], [(0, max_train_edges - len(train_y[i]))], mode = 'constant')

        global pad_time
        pad_time += (t.time() - start)

        yield ([np.array(train_X), np.array(train_Ri), np.array(train_Ro)], np.array(train_y))

test_datagen_time = 0
test_pad_time = 0

test_comp = []
test_type = []
pred_graph_ids = []

def predict_generator():
    global pred_graph_ids
    pred_graph_ids = []
    for batch_num in range(NTestingBatches):
        start = t.time()

        # Initialize vectorization of testing data
        max_test_hits, max_test_edges = 0, 0
        test_X = []
        test_Ri = []
        test_Ro = []
        test_y = []

        for e in range(BatchSize):

            # Prepare graph for a set of simulated events (testing)
            event = TestingDataSets[batch_num * BatchSize + e]
            graphRepresentation = GraphRepresentation.newGraphRepresentation(event, threshold=viz_threshold)
            pred_graph_ids.append(graphRepresentation.EventID)
            graphData = graphRepresentation.graphData
            A, Ro, Ri, X, y = graphData
            max_test_hits = max(max_test_hits, len(X))
            max_test_edges = max(max_test_edges, len(y))
            test_X.append(X)
            test_Ri.append(Ri)
            test_Ro.append(Ro)
            test_y.append(y)

            global test_comp
            test_comp.append(graphRepresentation.Compton)

            global test_type
            test_type.append(graphRepresentation.Tracks)

        global test_datagen_time
        test_datagen_time += (t.time() - start)

        start = t.time()

        # Padding to maximum dimension
        for i in range(len(test_X)):
            test_X[i] = np.pad(test_X[i], [(0, max_test_hits - len(test_X[i])), (0, 0)], mode = 'constant')
            test_Ri[i] = np.pad(test_Ri[i], [(0, max_test_hits - len(test_Ri[i])), (0, max_test_edges - len(test_Ri[i][0]))], mode = 'constant')
            test_Ro[i] = np.pad(test_Ro[i], [(0, max_test_hits - len(test_Ro[i])), (0, max_test_edges - len(test_Ro[i][0]))], mode = 'constant')
            test_y[i] = np.pad(test_y[i], [(0, max_test_edges - len(test_y[i]))], mode = 'constant')

        global test_pad_time
        test_pad_time += (t.time() - start)

        yield ([np.array(test_X), np.array(test_Ri), np.array(test_Ro)], np.array(test_y))



def evaluate_generator():
    while True:

        random_batch = np.random.randint(0, NTestingBatches - 1)

        # Initialize vectorization of testing data
        max_test_hits, max_test_edges = 0, 0
        test_X = []
        test_Ri = []
        test_Ro = []
        test_y = []

        for e in range(BatchSize):

            # Prepare graph for a set of simulated events (testing)
            event = TestingDataSets[random_batch * BatchSize + e]
            graphRepresentation = GraphRepresentation.newGraphRepresentation(event, threshold=viz_threshold)
            graphData = graphRepresentation.graphData
            A, Ro, Ri, X, y = graphData
            max_test_hits = max(max_test_hits, len(X))
            max_test_edges = max(max_test_edges, len(y))
            test_X.append(X)
            test_Ri.append(Ri)
            test_Ro.append(Ro)
            test_y.append(y)

        # Padding to maximum dimension
        for i in range(len(test_X)):
            test_X[i] = np.pad(test_X[i], [(0, max_test_hits - len(test_X[i])), (0, 0)], mode = 'constant')
            test_Ri[i] = np.pad(test_Ri[i], [(0, max_test_hits - len(test_Ri[i])), (0, max_test_edges - len(test_Ri[i][0]))], mode = 'constant')
            test_Ro[i] = np.pad(test_Ro[i], [(0, max_test_hits - len(test_Ro[i])), (0, max_test_edges - len(test_Ro[i][0]))], mode = 'constant')
            test_y[i] = np.pad(test_y[i], [(0, max_test_edges - len(test_y[i]))], mode = 'constant')

        yield ([np.array(test_X), np.array(test_Ri), np.array(test_Ro)], np.array(test_y))


###

class PrecisionRecallCallback(tf.keras.callbacks.Callback):
    best_train_recall = 0
    best_train_precision = 0
    best_train_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        best_train_accuracy = max(self.best_train_accuracy, logs[keys[1]])
        best_train_precision = max(self.best_train_precision, logs[keys[2]])
        best_train_recall = max(self.best_train_recall, logs[keys[3]])

        actual = []
        predictions = []

        for input, output in tqdm(predict_generator()):
            batch_pred = model.predict_on_batch(input)
            actual.extend(output)
            predictions.extend(batch_pred)

        assert len(pred_graph_ids) == len(predictions)

        for i in range(len(pred_graph_ids)):
            #print(i,sep="\r")
            GraphRepresentation.allGraphs[pred_graph_ids[i]].add_prediction(predictions[i])


callback = PrecisionRecallCallback()
# try different monitor values?
stopping = tf.keras.callbacks.EarlyStopping(monitor='precision', min_delta=0, patience=3, verbose=0, mode='auto',
                                            baseline=None, restore_best_weights=True)

train_start = t.time()
# Note: Not using stopping right now, to generate larger GIF.
hist = model.fit(data_generator(), steps_per_epoch = NTrainingBatches, epochs = epochs, callbacks=[callback, stopping])
train_time = t.time() - train_start

###################################################################################################
# Step 6: Evaluating the graph neural network
###################################################################################################

print("Info: Evaluating the graph neural network...")

# Generate predictions for a graph
start = t.time()

actual = []
predictions = []

for input, output in tqdm(predict_generator()):
    batch_pred = model.predict_on_batch(input)
    actual.extend(output)
    predictions.extend(batch_pred)

assert len(pred_graph_ids) == len(predictions)

for i in range(len(pred_graph_ids)):
    GraphRepresentation.allGraphs[pred_graph_ids[i]].add_prediction(predictions[i])

# GraphRepresentation.saveAllGraphs(OutputDirectory)

pred_time = t.time() - start

#test_graph = test_rep[0]
#test_graph.add_prediction(predictions[0])
#test_graph.visualize_last_prediction()

# for i in range(len(predictions)):
#     test_rep[i].add_prediction(predictions[i])
#     test_rep[i].visualize_last_prediction()

start = t.time()

evals = model.evaluate(evaluate_generator(), steps = NTestingBatches)

print(evals)

if Save:
    f = open(OutputDirectory + os.path.sep + "metrics.txt", "w+")
    keys = list(hist.history.keys())
    '''
    f.write("Num Events: {}\nAcceptance: {}\n\nTraining Metrics\nLoss: {}\nAccuracy: {}\nPrecision: {}\nRecall: {}\n\n".format(
            NumberOfDataSets,
            Acceptance,
            min(hist.history[keys[0]]),
            max(hist.history[keys[1]]),
            max(hist.history[keys[2]]),
            max(hist.history[keys[3]])))
    '''
    f.write("Num Events: {}\nAcceptance: {}\n\nTraining Metrics\nAccuracy: {}\nPrecision: {}\nRecall: {}\n\n".format(
            NumberOfDataSets,
            Acceptance,
            callback.best_train_accuracy,
            callback.best_train_precision,
            callback.best_train_recall))

    f.write("Eval Metrics\n{}".format(evals))
    f.close()

eval_time = t.time() - start

print("Time Elapsed for Data Loading: {} s".format(dataload_time))
print("Time Elapsed for Train/Test Split: {} s".format(traintestsplit_time))
print("Time Elapsed for Training Data Setup (Graph Representations): {} s".format(datagen_time))
print("Time Elapsed for Training Data Setup (Padding): {} s".format(pad_time))
print("Time Elapsed for Training: {} s".format(train_time))
print("Time Elapsed for Test Data Setup (Graph Representations): {} s".format(test_datagen_time))
print("Time Elapsed for Test Data Setup (Padding): {} s".format(test_pad_time))
print("Time Elapsed for Evaluation: {} s".format(eval_time))

precisions, recalls, thresholds = precision_recall_curve(np.hstack(actual), np.hstack(predictions))
data_dict = {'Precision' : precisions, 'Recall' : recalls, 'Thresholds' : thresholds}

np.save('Predictions', np.array(predictions, dtype = object))
np.save('Actual', np.array(actual, dtype = object))
np.save('Precision_Recall_Curve', data_dict)
np.save('Compton', np.array(test_comp, dtype = object))
np.save('Types', np.array(test_type, dtype = object))

