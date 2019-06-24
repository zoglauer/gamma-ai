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


NumberOfComptonEvents = 500
NumberOfBackgroundEvents = 100

NumberOfTrainingLocations = 5*128
NumberOfTestLocations = 128

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


ThetaMin = 0
ThetaMax = np.pi
ThetaBins = int(180 / ResolutionInDegrees)

ChiMin = 0
ChiMax = np.pi
ChiBins = int(180 / ResolutionInDegrees)

PsiMin = -np.pi
PsiMax = +np.pi
PsiBins = int(360 / ResolutionInDegrees)

InputDataSpaceSize = ThetaBins * ChiBins * PsiBins
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
  
TrainingDataSets = []

for g in range(0, NumberOfTrainingLocations):
  d = GRBData()
  d.create(ToyModelCreator, NumberOfComptonEvents, NumberOfBackgroundEvents)
  TrainingDataSets.append(d)
  
TestingDataSets = []

for g in range(0, NumberOfTestLocations):
  d = GRBData()
  d.create(ToyModelCreator, NumberOfComptonEvents, NumberOfBackgroundEvents)
  TestingDataSets.append(d)



###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################



print("Info: Setting up neural network...")

# Placeholders 
print("      ... placeholders ...")
X = tf.placeholder(tf.float32, [None, ThetaBins, ChiBins, PsiBins, 1], name="X")
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
    
    XTest = np.zeros(shape=(TestingBatchSize, ThetaBins, ChiBins, PsiBins, 1))
    YTest = np.zeros(shape=(TestingBatchSize, OutputDataSpaceSize))

    for g in range(0, TestingBatchSize):
      GRB = TrainingDataSets[g + Batch*TrainingBatchSize]
      YTest[g][0] = GRB.OriginLatitude
      YTest[g][1] = GRB.OriginLongitude
      for d in range(0, GRB.getNumberOfEntries()):
        PsiBin, ChiBin, PhiBin = GRB.getEntry(d)
        XTest[g, PhiBin, ChiBin, PsiBin] += 1
    
    YOut = sess.run(Output, feed_dict={X: XTest})
    
    print("Batch {}".format(Batch))
     
    # Calculate the angular deviation

    for l in range(0, TestingBatchSize):

      Real = M.MVector()
      Real.SetMagThetaPhi(1.0, YTest[l, 0], YTest[l, 1])

      Reconstructed = M.MVector()
      Reconstructed.SetMagThetaPhi(1.0, YOut[l, 0], YOut[l, 1])
  
      AngularDeviation[Batch*TestingBatchSize + l] = math.degrees(Real.Angle(Reconstructed))
  
      MeanAngularDeviation += AngularDeviation[l].item()
      RMSAngularDeviation += math.pow(AngularDeviation[l].item(), 2)
      
      print("  Cross-Check element: {:-7.3f} degrees difference: {:-6.3f} vs. {:-6.3f} & {:-6.3f} vs. {:-6.3f}".format(AngularDeviation[l].item(), YTest[l, 0].item(), YOut[l, 0].item(), YTest[l, 1].item(), YOut[l, 1].item()))
      
      
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
Timing = time.process_time()
MaxIterations = 50000
for Iteration in range(1, MaxIterations+1):
  for Batch in range(0, NumberOfTrainingBatches):
    # Take care of Ctrl-C
    if Interrupted == True: break

    # Convert the data set into training and testing data
    XTrain = np.zeros(shape=(TrainingBatchSize, ThetaBins, ChiBins, PsiBins, 1))
    YTrain = np.zeros(shape=(TrainingBatchSize, OutputDataSpaceSize))

    for g in range(0, TrainingBatchSize):
      GRB = TrainingDataSets[g + Batch*TrainingBatchSize]
      YTrain[g][0] = GRB.OriginLatitude
      YTrain[g][1] = GRB.OriginLongitude
      for d in range(0, GRB.getNumberOfEntries()):
        PsiBin, ChiBin, PhiBin = GRB.getEntry(d)
        XTrain[g, PhiBin, ChiBin, PsiBin] += 1

    # The actual training
    _, Loss = sess.run([Trainer, LossFunction], feed_dict={X: XTrain, Y: YTrain})
  
  # Take care of Ctrl-C
  if Interrupted == True: break

  # Check performance
  print("\n\nIteration {}".format(Iteration))
  Improvement = CheckPerformance()
    
  print("\nAverage time per training loop: ", (time.process_time() - Timing)/Iteration, " seconds")
  
  if Improvement == True:
    TimesNoImprovement = 0

    Saver.save(sess, "{}/Model_{}.ckpt".format(OutputDirectory, Iteration))

    with open(OutputDirectory + '/Progress.txt', 'a') as f:
      f.write(' '.join(map(str, (CheckPointNum, Iteration, BestMeanAngularDeviation, BestRMSAngularDeviation)))+'\n')
    
    print("\nSaved new best model and performance!")
    CheckPointNum += 1
  else:
    TimesNoImprovement += 1
    
  # Exit strategy
  if TimesNoImprovement == MaxTimesNoImprovement:
    print("\nNo improvement for {} iterations. Quitting!".format(MaxTimesNoImprovement))
    break;



#input("Press [enter] to EXIT")
sys.exit(0)

