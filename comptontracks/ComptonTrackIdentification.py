###################################################################################################
#
# ComptonTrackIdentification.py
#
# Copyright (C) by Andreas Zoglauer & Simar Ganda.
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



###################################################################################################
# Step 1: Input parameters
###################################################################################################


# Default parameters

# X, Y, Z bins
XBins = 11
YBins = 11
ZBins = 64

# File names
FileName = "ComptonTrackIdentification.inc1.id1.sim.gz"
GeometryName = "$(MEGALIB)/resource/examples/geomega/GRIPS/GRIPS.geo.setup"

# Depends on GPU memory and layout 
BatchSize = 256

# Split between training and testing data
TestingTrainingSplit = 0.25


# Determine derived parameters

OutputDataSpaceSize = 65

XMin = -43
YMin = -43
ZMin = 13
XMax = 43
YMax = 43
ZMax = 45

OutputDirectory = "Results"

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

while True: 
  Event = Reader.GetNextEvent()
  if not Event:
    break
  
  if Event.GetNIAs() > 0:
    Data = EventData()
    if Data.parse(Event) == True:
      DataSets.append(Data)

print("Info: Parsed {} events".format(len(DataSets)))


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
   TestingDataSets.append(DataSets[NTrainingBatches * BatchSize + 1])


NumberOfTrainingEvents = len(TrainingDataSets)
NumberOfTestingEvents = len(TestingDataSets)

print("Info: Number of training data sets: {}   Number of testing data sets: {} (vs. input: {} and split ratio: {})".format(NumberOfTrainingEvents, NumberOfTestingEvents, len(DataSets), TestingTrainingSplit))




###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################



print("Info: Setting up neural network...")

# Placeholders 
print("      ... placeholders ...")
X = tf.placeholder(tf.float32, [None, XBins, YBins, ZBins, 1], name="X")
Y = tf.placeholder(tf.float32, [None, OutputDataSpaceSize], name="Y")


L = tf.layers.conv3d(X, 8, 5, 2, 'VALID')
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(L, 0.1*L)

#L = tf.layers.conv3d(L, 8, 3, 1, 'VALID')
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(L, 0.1*L)

#L = tf.layers.conv3d(L, 16, 2, 2, 'VALID')
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(X, 0.1*X)

L = tf.layers.conv3d(L, 16, 2, 2, 'VALID')
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(L, 0.1*L)
 
#L = tf.layers.dense(tf.reshape(L, [-1, reduce(lambda a,b:a*b, L.shape.as_list()[1:])]), 128)
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.nn.relu(L)

L = tf.layers.dense(tf.reshape(L, [-1, reduce(lambda a,b:a*b, L.shape.as_list()[1:])]), OutputDataSpaceSize)
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
L = tf.nn.relu(L)

print("      ... output layer ...")
Output = tf.nn.softmax(L)


#tf.print("Y: ", Y, output_stream=sys.stdout)



# Loss function - simple linear distance between output and ideal results
print("      ... loss function ...")
LossFunction = tf.reduce_sum(np.abs(Output - Y)/NumberOfTestingEvents)
#LossFunction = tf.reduce_sum(tf.pow(Output - Y, 2))/NumberOfTestingEvents
#LossFunction = tf.losses.mean_squared_error(Output, Y)

# Minimizer
print("      ... minimizer ...")
Trainer = tf.train.AdamOptimizer().minimize(LossFunction)

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




###################################################################################################
# Step 5: Training and evaluating the network
###################################################################################################


print("Info: Training and evaluating the network")

# Train the network
MaxTimesNoImprovement = 1000
BestLoss = sys.float_info.max
IterationOutputInterval = 10
CheckPointNum = 0

print("Info: Creating configuration and progress file")
  
with open(OutputDirectory + '/Progress.txt', 'w') as f:
  f.write("Progress\n\n")



def CheckPerformance():
  Improvement = False
  
  # Step run all the testing batches, and detrmine the percentage of correct identifications
  
  return Improvement



# Main training and evaluation loop
Iteration = 0
MaxIterations = 50000
TimesNoImprovement = 0
while Iteration < MaxIterations:
  Iteration += 1
  print("Info: Starting iteration {}".format(Iteration))
  
  # Step 1: Loop over all training batches
  for Batch in range(0, NTrainingBatches):

    # Step 1.1: Convert the data set into the input and output tensor
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


    # Step 1.2: Perform the actual training
    _, Loss = Session.run([Trainer, LossFunction], feed_dict={X: InputTensor, Y: OutputTensor})

    if Interrupted == True: break

  # End for all batches

  

  # Step 2: Check current performance
  print("\n\nIteration: {}".format(Iteration))
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

  # Exit strategy
  if TimesNoImprovement == MaxTimesNoImprovement:
    print("\nNo improvement for {} iterations. Quitting!".format(MaxTimesNoImprovement))
    break;

  # Take care of Ctrl-C
  if Interrupted == True: break

# End: fo all iterations


#input("Press [enter] to EXIT")
sys.exit(0)

