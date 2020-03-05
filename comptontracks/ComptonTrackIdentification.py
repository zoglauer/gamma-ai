###################################################################################################
#
# ComptonTrackIdentification.py
#
# Copyright (C) by Andreas Zoglauer, Simar Ganda, Jasper Gan, and Pranav Nagarajan.
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################



###################################################################################################


import tensorflow as tf
import numpy as np

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

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

# X, Y, Z bins
XBins = 32
YBins = 32
ZBins = 64

# File names
FileName = "ComptonTrackIdentification.p1.sim.gz"
GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup"

# Depends on GPU memory and layout
BatchSize = 128

# Split between training and testing data
TestingTrainingSplit = 0.1

MaxEvents = 100000


# Determine derived parameters

OutputDataSpaceSize = ZBins

XMin = -43
XMax = 43

# XMin = -5
# XMax = +5

YMin = -43
YMax = 43

# YMin = -5
# YMax = +5

ZMin = 13
ZMax = 45

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
NumberOfDataSets = 0
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


print("Info: Setting up neural network...")

# # Placeholders
# print("      ... placeholders ...")
# X = tf.placeholder(tf.float32, [None, XBins, YBins, ZBins, 1], name="X")
# Y = tf.placeholder(tf.float32, [None, OutputDataSpaceSize], name="Y")

# L = tf.layers.dense(X, 128)
# L = tf.nn.relu(L)
#
# L = tf.layers.dense(X, 128)
# L = tf.nn.relu(L)

input = tf.keras.layers.Input(batch_shape = (None, XBins, YBins, ZBins, 1))

conv_1 = tf.keras.layers.Conv3D(32, 5, 2, 'valid')(input)
batch_1 = tf.keras.layers.BatchNormalization()(conv_1)
max_1 = tf.keras.layers.LeakyReLU(alpha = 0.1)(batch_1)

conv_2 = tf.keras.layers.Conv3D(32, 3, 1, 'valid')(max_1)
batch_2 = tf.keras.layers.BatchNormalization()(conv_2)
max_2 = tf.keras.layers.LeakyReLU(alpha = 0.1)(batch_2)

max_pool_3d = tf.keras.layers.MaxPooling3D(pool_size = (2,2,2), strides = 2)(max_2)

"""conv_1 = tf.keras.layers.Conv3D(64, 5, 2, 'valid')(input)
batch_1 = tf.keras.layers.BatchNormalization()(conv_1)
max_1 = tf.keras.layers.LeakyReLU(alpha = 0.1)(batch_1)

conv_2 = tf.keras.layers.Conv3D(64, 3, 1, 'valid')(max_1)
batch_2 = tf.keras.layers.BatchNormalization()(conv_2)
max_2 = tf.keras.layers.LeakyReLU(alpha = 0.1)(batch_2)

conv_3 = tf.keras.layers.Conv3D(128, 2, 2, 'valid')(max_2)
batch_3 = tf.keras.layers.BatchNormalization()(conv_3)
max_3 = tf.keras.layers.LeakyReLU(alpha = 0.1)(batch_3)

conv_4 = tf.keras.layers.Conv3D(128, 2, 2, 'valid')(max_3)
batch_4 = tf.keras.layers.BatchNormalization()(conv_4)
max_4 = tf.keras.layers.LeakyReLU(alpha = 0.1)(batch_4)"""

#conv_reshape = tf.reshape(max_4, [-1, reduce(lambda a,b:a*b, max_4.shape.as_list()[1:])])
reshape = tf.keras.layers.Flatten()(max_pool_3d)

#reshape = tf.keras.layers.Flatten()(max_4)
dense_1 = tf.keras.layers.Dense(64)(reshape)
batch_5 = tf.keras.layers.BatchNormalization()(dense_1)
activation = tf.keras.layers.ReLU()(batch_5)

drop = tf.keras.layers.Dropout(0.2)(activation)
dense_2 = tf.keras.layers.Dense(64)(drop)

#dense_2 = tf.keras.layers.Dense(64)(activation)

print("      ... output layer ...")
output = tf.keras.layers.Softmax()(dense_2)

model = tf.keras.models.Model(inputs = input, outputs = output)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Session configuration
print("      ... configuration ...")
Config = tf.ConfigProto()
Config.gpu_options.allow_growth = True

# Create and initialize the session
print("      ... session ...")
Session = tf.Session(config=Config)
Session.run(tf.global_variables_initializer())

print("      ... listing uninitialized variables if there are any ...")
print(tf.report_uninitialized_variables())

print("      ... writer ...")
writer = tf.summary.FileWriter(OutputDirectory, Session.graph)
writer.close()

# Add ops to save and restore all the variables.
print("      ... saver ...")
Saver = tf.train.Saver()

K = tf.keras.backend
K.set_session(Session)


###################################################################################################
# Step 5: Training and evaluating the network
###################################################################################################


print("Info: Training and evaluating the network")

# Train the network
BestLoss = sys.float_info.max
IterationOutputInterval = 10
CheckPointNum = 0

print("Info: Creating configuration and progress file")

TestingRealLayer = np.array([])
TestingPredictedLayer = np.array([])
TestingUniqueZLayer = np.array([])
TrainingRealLayer = np.array([])
TrainingPredictedLayer = np.array([])
TrainingUniqueZLayer = np.array([])

# Helper method
def getRealAndPredictedLayers(OutputDataSpaceSize, OutputTensor, Result, e, Event):
    real = -1
    predicted = -1
    unique = Event.unique
    predicted = np.argmax(Result[e])
    for l in range(0, OutputDataSpaceSize):
        if OutputTensor[e][l] > 0.5:
            real = l
    return real, predicted, unique


BestPercentageGood = 0.0

def CheckPerformance():
  global BestPercentageGood

  Improvement = False

  TotalEvents = 0
  BadEvents = 0

  # Step run all the testing batches, and detrmine the percentage of correct identifications
  # Step 1: Loop over all Testing batches
  for Batch in range(0, NTestingBatches):

    # Step 1.1: Convert the data set into the input and output tensor
    InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1))
    OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))


    # Loop over all testing  data sets and add them to the tensor
    for e in range(0, BatchSize):
      Event = TestingDataSets[e + Batch*BatchSize]
      # Set the layer in which the event happened

      LayerBin = int ((Event.OriginPositionZ - ZMin) / ((ZMax- ZMin)/ ZBins) )
      #print("layer bin: {} {}".format(Event.OriginPositionZ, LayerBin))

      if LayerBin < 0 or LayerBin >= OutputDataSpaceSize:
        print("Error: The calculated layer bin ({}) is out of bounds [0, {}]".format(LayerBin, OutputDataSpaceSize-1))
      else:
        OutputTensor[e][LayerBin] = 1

      # Set all the hit locations and energies
      SomethingAdded = False
      for h in range(0, len(Event.X)):
        XBin = int( (Event.X[h] - XMin) / ((XMax - XMin) / XBins) )
        YBin = int( (Event.Y[h] - YMin) / ((YMax - YMin) / YBins) )
        ZBin = int( (Event.Z[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
        #print("hit z bin: {} {}".format(Event.Z[h], ZBin))
        if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
          InputTensor[e][XBin][YBin][ZBin][0] = Event.E[h]
          SomethingAdded = True

      if SomethingAdded == False:
        print("Nothing added for event {}".format(Event.ID))
        Event.print()


    # Step 2: Run it
    # Result = Session.run(Output, feed_dict={X: InputTensor})
    Result = model.predict(InputTensor)

    #print(Result[e])
    #print(OutputTensor[e])

    for e in range(0, BatchSize):
      Event = TestingDataSets[e + Batch*BatchSize]
      TotalEvents += 1
      IsBad = False
      LargestValueBin = 0
      LargestValue = OutputTensor[e][0]
      for c in range(1, OutputDataSpaceSize) :
        if Result[e][c] > LargestValue:
          LargestValue = Result[e][c]
          LargestValueBin = c

      if OutputTensor[e][LargestValueBin] < 0.99:
        BadEvents += 1
        IsBad = True

        #if math.fabs(Result[e][c] - OutputTensor[e][c]) > 0.1:
        #  BadEvents += 1
        #  IsBad = True
        #  break

      # Fetch real and predicted layers for testing data
      real, predicted, uniqueZ = getRealAndPredictedLayers(OutputDataSpaceSize, OutputTensor, Result, e, Event)
      global TestingRealLayer
      global TestingPredictedLayer
      global TestingUniqueZLayer
      TestingRealLayer = np.append(TestingRealLayer, real)
      TestingPredictedLayer = np.append(TestingPredictedLayer, predicted)
      TestingUniqueZLayer = np.append(TestingUniqueZLayer, uniqueZ)

      # Some debugging
      if Batch == 0 and e < 500:
        EventID = e + Batch*BatchSize + NTrainingBatches*BatchSize
        print("Event {}:".format(EventID))
        if IsBad == True:
          print("BAD")
        else:
          print("GOOD")
        DataSets[EventID].print()

        print("Results layer: {}".format(LargestValueBin))
        for l in range(0, OutputDataSpaceSize):
          if OutputTensor[e][l] > 0.5:
            print("Real layer: {}".format(l))
          #print(OutputTensor[e])
          #print(Result[e])

  PercentageGood = 100.0 * float(TotalEvents-BadEvents) / TotalEvents

  if PercentageGood > BestPercentageGood:
    BestPercentageGood = PercentageGood
    Improvement = True

  print("Percentage of good events: {:-6.2f}% (best so far: {:-6.2f}%)".format(PercentageGood, BestPercentageGood))

  return Improvement

# Main training and evaluation loop

TimeConverting = 0.0
TimeTraining = 0.0
TimeTesting = 0.0

Iteration = 0
MaxIterations = 50000
TimesNoImprovement = 0
MaxTimesNoImprovement = 50
while Iteration < MaxIterations:
  Iteration += 1
  print("\n\nStarting iteration {}".format(Iteration))

  # Step 1: Loop over all training batches
  for Batch in range(0, NTrainingBatches):

    # Step 1.1: Convert the data set into the input and output tensor
    TimerConverting = time.time()

    InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1))
    OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))

    # Loop over all training data sets and add them to the tensor
    for g in range(0, BatchSize):
      Event = TrainingDataSets[g + Batch*BatchSize]
      # Set the layer in which the event happened
      if Event.OriginPositionZ > ZMin and Event.OriginPositionZ < ZMax:
        LayerBin = int ((Event.OriginPositionZ - ZMin) / ((ZMax- ZMin)/ ZBins) )
        OutputTensor[g][LayerBin] = 1
      else:
        OutputTensor[g][OutputDataSpaceSize-1] = 1

      # Set all the hit locations and energies
      for h in range(0, len(Event.X)):
        XBin = int( (Event.X[h] - XMin) / ((XMax - XMin) / XBins) )
        YBin = int( (Event.Y[h] - YMin) / ((YMax - YMin) / YBins) )
        ZBin = int( (Event.Z[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
        if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
          InputTensor[g][XBin][YBin][ZBin][0] = Event.E[h]

    TimeConverting += time.time() - TimerConverting

    # Step 1.2: Perform the actual training
    TimerTraining = time.time()
    #print("\nStarting training for iteration {}, batch {}/{}".format(Iteration, Batch, NTrainingBatches))
    #_, Loss = Session.run([Trainer, LossFunction], feed_dict={X: InputTensor, Y: OutputTensor})
    History = model.fit(InputTensor, OutputTensor, validation_split=0.1)
    Loss = History.history['loss'][-1]
    TimeTraining += time.time() - TimerTraining

    Result = model.predict(InputTensor)

    for e in range(0, BatchSize):
        Event = TrainingDataSets[e + Batch*BatchSize]
        # Fetch real and predicted layers for training data
        real, predicted, uniqueZ = getRealAndPredictedLayers(OutputDataSpaceSize, OutputTensor, Result, e, Event)
        TrainingRealLayer = np.append(TrainingRealLayer, real)
        TrainingPredictedLayer = np.append(TrainingPredictedLayer, predicted)
        TrainingUniqueZLayer = np.append(TrainingUniqueZLayer, uniqueZ)

    if Interrupted == True: break

  # End for all batches

  # Step 2: Check current performance
  TimerTesting = time.time()
  print("\nCurrent loss: {}".format(Loss))
  Improvement = CheckPerformance()

  if Improvement == True:
    TimesNoImprovement = 0

    Saver.save(Session, "{}/Model_{}.ckpt".format(OutputDirectory, Iteration))

    with open(OutputDirectory + '/Progress.txt', 'a') as f:
      f.write(' '.join(map(str, (CheckPointNum, Iteration, Loss)))+'\n')

    print("\nSaved new best model and performance!")
    CheckPointNum += 1
  else:
    TimesNoImprovement += 1

  TimeTesting += time.time() - TimerTesting

  # Exit strategy
  if TimesNoImprovement == MaxTimesNoImprovement:
    print("\nNo improvement for {} iterations. Quitting!".format(MaxTimesNoImprovement))
    break;

  # Take care of Ctrl-C
  if Interrupted == True: break

  print("\n\nTotal time converting per Iteration: {} sec".format(TimeConverting/Iteration))
  print("Total time training per Iteration:   {} sec".format(TimeTraining/Iteration))
  print("Total time testing per Iteration:    {} sec".format(TimeTesting/Iteration))

# End: for all iterations

# Store Real & Predicted Layers in FileSystem
realPredictedLayersData = {
    'TestingRealLayer': TestingRealLayer,
    'TestingPredictedLayer': TestingPredictedLayer,
    'TestingUniqueZLayer': TestingUniqueZLayer,
    'TrainingRealLayer': TrainingRealLayer,
    'TrainingPredictedLayer': TrainingPredictedLayer,
    'TrainingUniqueZLayer': TrainingUniqueZLayer
}

np.save('realPredictedLayers', realPredictedLayersData)
# loadedRealPredictedLayersData = np.load('realPredictedLayers.npy')

#input("Press [enter] to EXIT")
sys.exit(0)
