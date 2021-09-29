###################################################################################################
#
# RecoilElectrons.py
#
# Copyright (C) by Andreas Zoglauer & contributors
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################



###################################################################################################


import tensorflow as tf
from tensorflow.keras import datasets, layers, models


import numpy as np

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

import random
import pickle

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
FileName = "RecoilElectrons.inc1.id1.data"

# Depends on GPU memory and layout
BatchSize = 64

# Split between training and testing data
TestingTrainingSplit = 0.1

MaxEvents = 100000


# Determine derived parameters

OutputDataSpaceSize = 6

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

#if os.path.exists(OutputDirectory):
#  Now = datetime.now()
#  OutputDirectory += Now.strftime("_%Y%m%d_%H%M%S")
#os.makedirs(OutputDirectory)



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
  print("You pressed Ctrl+C - waiting for graceful abort, or press Ctrl-C again, for quick exit.")
signal.signal(signal.SIGINT, signal_handler)


# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from EventData import EventData


###################################################################################################
# Step 3: Read the data
###################################################################################################


print("\n\nStarted reading data sets")

with open(FileName, "rb") as FileHandle:
  DataSets = pickle.load(FileHandle)

NumberOfDataSets = len(DataSets)

print("Info: Parsed {} events".format(NumberOfDataSets))


###################################################################################################
# Step 4: Split the data into training, test & verification data sets
###################################################################################################


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

print("Info: Number of training data sets: {}   Number of testing data sets: {} (vs. input: {} and split ratio: {})".format(NumberOfTrainingEvents, NumberOfTestingEvents, len(DataSets), TestingTrainingSplit))




###################################################################################################
# Step 5: Setting up the neural network
###################################################################################################


print("Info: Setting up neural network...")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)])
  except RuntimeError as e:
    print(e)

Model = models.Sequential()
Model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(XBins, YBins, ZBins, 1)))
Model.add(layers.MaxPooling3D((2, 2, 3)))
Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
Model.add(layers.MaxPooling3D((2, 2, 2)))
Model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))

Model.add(layers.Flatten())
Model.add(layers.Dense(64, activation='relu'))
Model.add(layers.Dense(OutputDataSpaceSize))
    
Model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08), loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    
Model.summary()



###################################################################################################
# Step 6: Training and evaluating the network
###################################################################################################

print("Info: Training and evaluating the network")

# Train the network
BestLoss = sys.float_info.max
IterationOutputInterval = 10
CheckPointNum = 0


BestLocation = 100000
BestAngle = 180

###################################################################################################

def CheckPerformance():
  global BestLocation
  global BestAngle

  Improvement = False

  TotalEvents = 0
  SumDistDiff = 0
  SumAngleDiff = 0

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

      # Set all the hit locations and energies
      for h in range(0, len(Event.X)):
        XBin = int( (Event.X[h] - XMin) / ((XMax - XMin) / XBins) )
        YBin = int( (Event.Y[h] - YMin) / ((YMax - YMin) / YBins) )
        ZBin = int( (Event.Z[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
        #print("hit z bin: {} {}".format(Event.Z[h], ZBin))
        if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
          InputTensor[e][XBin][YBin][ZBin][0] = Event.E[h]



    # Step 2: Run it
    # Result = Session.run(Output, feed_dict={X: InputTensor})
    Result = Model.predict(InputTensor)

    #print(Result[e])
    #print(OutputTensor[e])

    for e in range(0, BatchSize):
      Event = TestingDataSets[e + Batch*BatchSize]
      
      oPos = np.array([ Event.TrackStartX, Event.TrackStartY, Event.TrackStartZ ])
      rPos = np.array([ Result[e][0], Result[e][1], Result[e][2] ])
      
      oDir = np.array([ Event.TrackDirectionX, Event.TrackDirectionY, Event.TrackDirectionZ ])
      rDir = np.array([ Result[e][3], Result[e][4], Result[e][5] ])
      
      # Distance difference location:
      DistDiff = np.linalg.norm(oPos - rPos)
      
      # Angle difference direction
      Norm = np.linalg.norm(oDir)
      if Norm == 0:
        print("Warning: original direction is zero: {} {} {}".format(Event.DirectionStartX, Event.DirectionStartY, Event.DirectionStartZ))
        continue
      oDir /= Norm
      Norm = np.linalg.norm(rDir)
      if Norm == 0:
        print("Warning: estimated direction is zero: {} {} {}".format(Result[e][3], Result[e][4], Result[e][5]))
        continue
      rDir /= Norm
      AngleDiff = np.arccos(np.clip(np.dot(oDir, rDir), -1.0, 1.0)) * 180/math.pi

      if math.isnan(AngleDiff):
        continue
      
      SumDistDiff += DistDiff
      SumAngleDiff += AngleDiff
      TotalEvents += 1


      # Some debugging
      if Batch == 0 and e < 5:
        EventID = e + Batch*BatchSize + NTrainingBatches*BatchSize
        print("\nEvent {}:".format(EventID))
        DataSets[EventID].print()

        print("Positions: {} vs {} -> {} cm difference".format(oPos, rPos, DistDiff))
        print("Directions: {} vs {} -> {} degree difference".format(oDir, rDir, AngleDiff))

  if TotalEvents > 0:
    if SumDistDiff / TotalEvents < BestLocation and SumAngleDiff / TotalEvents < BestAngle:
      BestLocation = SumDistDiff / TotalEvents
      BestAngle = SumAngleDiff / TotalEvents
      Improvement = True

    print("Status: distance difference = {:-6.2f} cm, angle difference = {:-6.2f} deg".format(SumDistDiff / TotalEvents, SumAngleDiff / TotalEvents))

  return Improvement


###################################################################################################



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
    print("Batch {} / {}".format(Batch+1, NTrainingBatches))
    
    # Step 1.1: Convert the data set into the input and output tensor
    TimerConverting = time.time()

    InputTensor = np.zeros(shape=(BatchSize, XBins, YBins, ZBins, 1))
    OutputTensor = np.zeros(shape=(BatchSize, OutputDataSpaceSize))

    # Loop over all training data sets and add them to the tensor
    for g in range(0, BatchSize):
      Event = TrainingDataSets[g + Batch*BatchSize]

      # Set all the hit locations and energies
      for h in range(0, len(Event.X)):
        XBin = int( (Event.X[h] - XMin) / ((XMax - XMin) / XBins) )
        YBin = int( (Event.Y[h] - YMin) / ((YMax - YMin) / YBins) )
        ZBin = int( (Event.Z[h] - ZMin) / ((ZMax - ZMin) / ZBins) )
        if XBin >= 0 and YBin >= 0 and ZBin >= 0 and XBin < XBins and YBin < YBins and ZBin < ZBins:
          InputTensor[g][XBin][YBin][ZBin][0] = Event.E[h]
      
      # for us: output tensor will be 'expected gamma energy'
      # input: X, Y, Z, event energy measured, (and maybe electron type (pair/compton))

      #output tensor [g] = gamma energy
      
      OutputTensor[g][0] = Event.TrackStartX
      OutputTensor[g][1] = Event.TrackStartY
      OutputTensor[g][2] = Event.TrackStartZ

      OutputTensor[g][3] = Event.TrackDirectionX
      OutputTensor[g][4] = Event.TrackDirectionY
      OutputTensor[g][5] = Event.TrackDirectionZ

    TimeConverting += time.time() - TimerConverting



    # Step 1.2: Perform the actual training
    TimerTraining = time.time()
    History = Model.fit(InputTensor, OutputTensor, validation_split=0.1)
    Loss = History.history['loss'][-1]
    TimeTraining += time.time() - TimerTraining

    if Interrupted == True: break

  # End for all batches

  # Step 2: Check current performance
  TimerTesting = time.time()
  print("\nCurrent loss: {}".format(Loss))
  Improvement = CheckPerformance()

  if Improvement == True:
    TimesNoImprovement = 0

    print("\nFound new best model and performance!")
  else:
    TimesNoImprovement += 1

  TimeTesting += time.time() - TimerTesting

  # Exit strategy
  if TimesNoImprovement == MaxTimesNoImprovement:
    print("\nNo improvement for {} iterations. Quitting!".format(MaxTimesNoImprovement))
    break;

  print("\n\nTotal time converting per Iteration: {} sec".format(TimeConverting/Iteration))
  print("Total time training per Iteration:   {} sec".format(TimeTraining/Iteration))
  print("Total time testing per Iteration:    {} sec".format(TimeTesting/Iteration))

  # Take care of Ctrl-C
  if Interrupted == True: break


# End: for all iterations


#input("Press [enter] to EXIT")
sys.exit(0)
