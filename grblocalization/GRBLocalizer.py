###################################################################################################
#
# GRBLocalizer.py
#
# Copyright (C) by Andreas Zoglauer & Anna Shang.
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
from datetime import datetime
from functools import reduce

import ROOT as M

# Load MEGAlib into ROOT so that it is usable
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

from GRBData import GRBData
from GRBCreatorToyModel import GRBCreatorToyModel



print("\nGRB localization (tensorflow based) \n")



###################################################################################################
# Step 1: Input parameters
###################################################################################################


# User input parameters


NumberOfComptonEvents = 2000
NumberOfBackgroundEvents = 0

NumberOfTrainingLocations = 32*128
NumberOfTestLocations = 4*128

MaxBatchSize = 128

ResolutionInDegrees = 5

OneSigmaNoiseInDegrees = 0.0
OneSigmaNoiseInRadians = math.radians(OneSigmaNoiseInDegrees)

OutputDirectory = "Output"


# Set derived parameters
NumberOfTrainingBatches= (int) (NumberOfTrainingLocations / MaxBatchSize)
TrainingBatchSize = (int) (NumberOfTrainingLocations / NumberOfTrainingBatches)
if TrainingBatchSize > MaxBatchSize:
  print("Error: Training batch size larger than {}: {}".format(MaxBatchSize, TrainingBatchSize))
  sys.exit(0)


NumberOfTestingBatches= (int) (NumberOfTestLocations / MaxBatchSize)
TestingBatchSize = (int) (NumberOfTestLocations / NumberOfTestingBatches)
if TrainingBatchSize > MaxBatchSize:
  print("Error: Testing batch size larger than {}: {}".format(MaxBatchSize, TestingBatchSize))
  sys.exit(0)


PsiMin = -np.pi
PsiMax = +np.pi
PsiBins = int(360 / ResolutionInDegrees)

ChiMin = 0
ChiMax = np.pi
ChiBins = int(180 / ResolutionInDegrees)

PhiMin = 0
PhiMax = np.pi
PhiBins = int(180 / ResolutionInDegrees)

InputDataSpaceSize = PsiBins * ChiBins * PhiBins
OutputDataSpaceSize = 2

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




###################################################################################################
# Step 3: Create some training, test & verification data sets
###################################################################################################


print("Info: Creating {:,} Compton events".format((NumberOfTrainingLocations + NumberOfTestLocations) * (NumberOfComptonEvents + NumberOfBackgroundEvents)))


ToyModelCreator = GRBCreatorToyModel(ResolutionInDegrees, OneSigmaNoiseInDegrees)  


def generateOneDataSet(_):
  DataSet = GRBData()
  DataSet.create(ToyModelCreator, NumberOfComptonEvents, NumberOfBackgroundEvents)
  return DataSet
  
  
# Parallelizing using Pool.starmap()
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

# Create data sets
TimerCreation = time.time()

TrainingDataSets = pool.map(generateOneDataSet, range(0, NumberOfTrainingLocations))
print("Info: Created {:,} training data sets. ".format(NumberOfTrainingLocations))

TestingDataSets = pool.map(generateOneDataSet, range(0, NumberOfTestLocations))
print("Info: Created {:,} testing data sets. ".format(NumberOfTestLocations))

pool.close()

TimeCreation = time.time() - TimerCreation
print("Total time to create data sets: {:.1f} seconds (= {:,.0f} events/second)".format(TimeCreation, (NumberOfTrainingLocations + NumberOfTestLocations) * (NumberOfComptonEvents + NumberOfBackgroundEvents) / TimeCreation))


# Convert the data set into training and testing data
TimerConverting = time.time()
XTrain = np.zeros(shape=(TrainingBatchSize, PsiBins*ChiBins*PhiBins*1))
#XTrain = np.zeros(shape=(TrainingBatchSize, PsiBins, ChiBins, PhiBins, 1))
YTrain = np.zeros(shape=(TrainingBatchSize, OutputDataSpaceSize))

print("Total time for 1 initialization: {} seconds".format(time.time() - TimerConverting))

for g in range(0, TrainingBatchSize):
  GRB = TrainingDataSets[g]
  YTrain[g][0] = GRB.OriginLatitude
  YTrain[g][1] = GRB.OriginLongitude
  
  XSlice = XTrain[g,]
  for d in range(0, GRB.getNumberOfEntries()):
    #PsiBin, ChiBin, PhiBin = GRB.getEntry(d)
    #XTrain[g, PhiBin, ChiBin, PsiBin] += 1

    Index = GRB.getIndex(d)
    XSlice[Index] += 1

XTrain = XTrain.reshape((TrainingBatchSize, PsiBins, ChiBins, PhiBins, 1))

print("Total time for 1/{} conversion: {} seconds".format(NumberOfTrainingBatches, time.time() - TimerConverting))


# Plot the first test data point
'''
l = 0

print("Pos {}, {}".format(math.degrees(YTrain[l, 0]), math.degrees(YTrain[l, 1])))
  
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
  
fig = plt.figure()
ax = fig.gca(projection='3d')

adds = 0
for p in range(0, PsiBins):
  for c in range(0, ChiBins):
    for t in range(0, PhiBins):
      if XTrain[l, p,c, t] > 0:
        ax.scatter(math.degrees(PsiMin) + p * ResolutionInDegrees, math.degrees(ChiMin) + c * ResolutionInDegrees, math.degrees(PhiMin) + t * ResolutionInDegrees, XTrain[l, p,c, t])
        #print("{}, {}, {}".format(math.degrees(PsiMin) + p * ResolutionInDegrees, math.degrees(ChiMin) + c * ResolutionInDegrees, math.degrees(PhiMin) + t * ResolutionInDegrees))
        adds += XTrain[l, p,c, t]
    
print("Adds: {}".format(adds))

plt.show()
plt.pause(0.001)
    
input("Press [enter] to EXIT")
sys.exit()
'''


###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################



print("Info: Setting up neural network...")

# Placeholders 
print("      ... placeholders ...")
X = tf.placeholder(tf.float32, [None, PsiBins, ChiBins, PhiBins, 1], name="X")
Y = tf.placeholder(tf.float32, [None, OutputDataSpaceSize], name="Y")


L = tf.layers.conv3d(X, 64, 5, 2, 'VALID')
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(L, 0.1*L)

L = tf.layers.conv3d(L, 64, 3, 1, 'VALID')
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(L, 0.1*L)

L = tf.layers.conv3d(L, 128, 2, 2, 'VALID')
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(X, 0.1*X)

L = tf.layers.conv3d(L, 128, 2, 2, 'VALID')
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(L, 0.1*L)
 
L = tf.layers.dense(tf.reshape(L, [-1, reduce(lambda a,b:a*b, L.shape.as_list()[1:])]), 128)
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
L = tf.nn.relu(L)

print("      ... output layer ...")
Output = tf.layers.dense(tf.reshape(L, [-1, reduce(lambda a,b:a*b, L.shape.as_list()[1:])]), OutputDataSpaceSize)


#tf.print("Y: ", Y, output_stream=sys.stdout)



# Loss function - simple linear distance between output and ideal results
print("      ... loss function ...")
LossFunction = tf.reduce_sum(np.abs(Output - Y)/NumberOfTestLocations)
#LossFunction = tf.reduce_sum(tf.pow(Output - Y, 2))/NumberOfTestLocations
#LossFunction = tf.losses.mean_squared_error(Output, Y)

# Minimizer
print("      ... minimizer ...")
Trainer = tf.train.AdamOptimizer().minimize(LossFunction)

# Create and initialize the session
print("      ... session ...")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("      ... listing uninitialized variables if there are any ...")
print(tf.report_uninitialized_variables())


print("      ... writer ...")
writer = tf.summary.FileWriter(OutputDirectory, sess.graph)
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
TimesNoImprovement = 0
BestMeanSquaredError = sys.float_info.max
BestMeanAngularDeviation = sys.float_info.max
BestRMSAngularDeviation = sys.float_info.max
BestLoss = sys.float_info.max
IterationOutputInterval = 10
CheckPointNum = 0

print("Creating progress file")
with open(OutputDirectory + "/Progress.txt", 'w') as f:
  f.write('')



def CheckPerformance():
  global TimesNoImprovement
  global BestMeanSquaredError
  global BestMeanAngularDeviation
  global BestRMSAngularDeviation
  global IterationOutputInterval

  Improvement = False

  # Run the test data
  AngularDeviation = np.zeros(shape=(NumberOfTestingBatches*TestingBatchSize, 1))
  MeanAngularDeviation = 0
  RMSAngularDeviation = 0
  for Batch in range(0, NumberOfTestingBatches):
        
    # Step 1: Convert the data
    XTest = np.zeros(shape=(TestingBatchSize, PsiBins*ChiBins*PhiBins*1))
    YTest = np.zeros(shape=(TestingBatchSize, OutputDataSpaceSize))

    for g in range(0, TestingBatchSize):
      GRB = TrainingDataSets[g + Batch*TrainingBatchSize]
      YTest[g][0] = GRB.OriginLatitude
      YTest[g][1] = GRB.OriginLongitude
      
      XSlice = XTest[g,]
      for d in range(0, GRB.getNumberOfEntries()):
        Index = GRB.getIndex(d)
        XSlice[Index] += 1
    
    XTest = XTest.reshape((TestingBatchSize, PsiBins, ChiBins, PhiBins, 1))
    
    
    # Step 2: Run it
    YOut = sess.run(Output, feed_dict={X: XTest})
    

    # Step 3: Analyze it
    # Calculate the angular deviation
    for l in range(0, TestingBatchSize):

      Real = M.MVector()
      Real.SetMagThetaPhi(1.0, YTest[l, 0], YTest[l, 1])

      Reconstructed = M.MVector()
      Reconstructed.SetMagThetaPhi(1.0, YOut[l, 0], YOut[l, 1])
  
      AngularDeviation[Batch*TestingBatchSize + l] = math.degrees(Real.Angle(Reconstructed))
  
      MeanAngularDeviation += AngularDeviation[l].item()
      RMSAngularDeviation += math.pow(AngularDeviation[l].item(), 2)
      
      if Batch == NumberOfTestingBatches-1:
        print("  Cross-Check element: {:-7.3f} degrees difference: {:-6.3f} vs. {:-6.3f} & {:-6.3f} vs. {:-6.3f}".format(AngularDeviation[l].item(), YTest[l, 0].item(), YOut[l, 0].item(), YTest[l, 1].item(), YOut[l, 1].item()))
      
  # End: For each batch
      
  # Calculate the mean RMS
  MeanAngularDeviation /= NumberOfTestingBatches*TestingBatchSize
  RMSAngularDeviation /= NumberOfTestingBatches*TestingBatchSize
  RMSAngularDeviation = math.sqrt(RMSAngularDeviation)

  # Check for improvement mean
  if MeanAngularDeviation < BestMeanAngularDeviation:
    BestMeanAngularDeviation = MeanAngularDeviation
    BestRMSAngularDeviation = RMSAngularDeviation
    Improvement = True

  # Check for improvement RMS
  #if RMSAngularDeviation < BestRMSAngularDeviation:
  #  BestRMSAngularDeviation = RMSAngularDeviation
  #  Improvement = True

  print("\n")
  print("RMS Angular deviation:   {:-6.3f} deg  -- best: {:-6.3f} deg".format(RMSAngularDeviation, BestRMSAngularDeviation))
  print("Mean Angular deviation:  {:-6.3f} deg  -- best: {:-6.3f} deg".format(MeanAngularDeviation, BestMeanAngularDeviation))
  
  return Improvement



# Main training and evaluation loop
TimeConverting = 0.0
TimeTraining = 0.0
TimeTesting = 0.0
MaxIterations = 50000
for Iteration in range(1, MaxIterations+1):
  for Batch in range(0, NumberOfTrainingBatches):

    # Convert the data set into training and testing data
    TimerConverting = time.time()
    
    XTrain = np.zeros(shape=(TrainingBatchSize, PsiBins*ChiBins*PhiBins*1))
    YTrain = np.zeros(shape=(TrainingBatchSize, OutputDataSpaceSize))

    for g in range(0, TrainingBatchSize):
      GRB = TrainingDataSets[g + Batch*TrainingBatchSize]
      YTrain[g][0] = GRB.OriginLatitude
      YTrain[g][1] = GRB.OriginLongitude
      
      XSlice = XTrain[g,]
      for d in range(0, GRB.getNumberOfEntries()):
        Index = GRB.getIndex(d)
        XSlice[Index] += 1
        
    XTrain = XTrain.reshape((TrainingBatchSize, PsiBins, ChiBins, PhiBins, 1))
    
    TimeConverting += time.time() - TimerConverting


    # The actual training
    TimerTraining = time.time()
    _, Loss = sess.run([Trainer, LossFunction], feed_dict={X: XTrain, Y: YTrain})
    TimeTraining += time.time() - TimerTraining
    
  # End for all batches


  # Check performance
  TimerTesting = time.time()
  print("\n\nIteration {}".format(Iteration))
  Improvement = CheckPerformance()
  
  if Improvement == True:
    TimesNoImprovement = 0

    Saver.save(sess, "{}/Model_{}.ckpt".format(OutputDirectory, Iteration))

    with open(OutputDirectory + '/Progress.txt', 'a') as f:
      f.write(' '.join(map(str, (CheckPointNum, Iteration, BestMeanAngularDeviation, BestRMSAngularDeviation)))+'\n')
    
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

# End: fo all iterations

print("Total time converting: {} sec".format(TimeConverting))
print("Total time training:   {} sec".format(TimeTraining))
print("Total time testing:    {} sec".format(TimeTesting))


#input("Press [enter] to EXIT")
sys.exit(0)

