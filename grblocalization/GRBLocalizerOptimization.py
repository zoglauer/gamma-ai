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
import argparse
from datetime import datetime
from functools import reduce


print("\nGRB localization (tensorflow based)")
print("===================================\n")



###################################################################################################
# Step 1: Input parameters
###################################################################################################


# Default parameters

NumberOfComptonEvents = 2000
NumberOfBackgroundEvents = 0

# Depends on GPU memory and layout 
MaxBatchSize = 256

NumberOfTrainingBatches = 32
NumberOfTestingBatches = 8

ResolutionInDegrees = 5

OneSigmaNoiseInDegrees = 0.0


OutputDirectory = "Output"


# Parse command line:

print("\nParsing the command line (if there is any)\n")

parser = argparse.ArgumentParser(description='Perform training and/or testing for gamma-ray burst localization')
parser.add_argument('-m', '--mode', default='toymodel', help='Choose an input data more: toymodel or simulations')
parser.add_argument('-t', '--toymodeloptions', default='2000:0:0.0:32:8', help='The toy-model options: source_events:background_events:one_sigma_noise_in_degrees:training_batches:testing_batches')
parser.add_argument('-s', '--simulationoptions', default='', help='')
parser.add_argument('-r', '--resolution', default='5.0', help='Resolution of the input grid in degrees')
parser.add_argument('-b', '--batchsize', default='256', help='The number of GRBs in one training batch (default: 256 corresponsing to 5 degree grid resolution (64 for 3 degrees))')
parser.add_argument('-o', '--outputdirectory', default='Output', help='Name of the output directory. If it exists, the current data and time will be appended.')
parser

args = parser.parse_args()

  
Mode = (args.mode).lower()
if Mode != 'toymodel' and Mode != 'simulation':
  print("Error: The mode must be either \'toymodel\' or \'simulation\'")
  sys.exit(0)


if Mode == 'toymodel':
  print("CMD-Line: Using toy model".format(NumberOfComptonEvents))

  ToyModelOptions = args.toymodeloptions.split(":")
  if len(ToyModelOptions) != 5:
    print("Error: You need to give 5 toy model options. You gave {}. Options: {}".format(len(ToyModelOptions), ToyModelOptions))
    sys.exit(0)
  
  NumberOfComptonEvents = int(ToyModelOptions[0])
  if NumberOfComptonEvents <= 10:
    print("Error: You need at least 10 source events and not {}".format(NumberOfComptonEvents))
    sys.exit(0)       
  print("CMD-Line: Toy model: Using {} source events per GRB".format(NumberOfComptonEvents))

  NumberOfBackgroundEvents = int(ToyModelOptions[1])
  if NumberOfBackgroundEvents < 0:
    print("Error: You need a non-negative number of background events and not {}".format(NumberOfBackgroundEvents))
    sys.exit(0)        
  print("CMD-Line: Toy model: Using {} background events per GRB".format(NumberOfBackgroundEvents))

  OneSigmaNoiseInDegrees = float(ToyModelOptions[2])
  if OneSigmaNoiseInDegrees < 0:
    print("Error: You need a non-negative number for the noise and not {}".format(OneSigmaNoiseInDegrees))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} degrees as 1-sigma resolution".format(OneSigmaNoiseInDegrees))

  NumberOfTrainingBatches = int(ToyModelOptions[3])
  if NumberOfTrainingBatches < 1:
    print("Error: You need a positive number for the number of traing batches and not {}".format(NumberOfTrainingBatches))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} training batches".format(NumberOfTrainingBatches))

  NumberOfTestingBatches = int(ToyModelOptions[4])
  if NumberOfTestingBatches < 1:
    print("Error: You need a positive number for the number of testing batches and not {}".format(NumberOfTestingBatches))
    sys.exit(0)
  print("CMD-Line: Toy model: Using {} testing batches".format(NumberOfTestingBatches))

elif Mode == 'simulation':
  print("Error: The simulation mode has not yet implemented")
  sys.exit(0)  


ResolutionInDegrees = float(args.resolution)
if ResolutionInDegrees > 10 or ResolutionInDegrees < 1:
  print("Error: The resolution must be between 1 & 10 degrees")
  sys.exit(0)
print("CMD-Line: Using {} degrees as input grid resolution".format(ResolutionInDegrees))
  
MaxBatchSize = int(args.batchsize)
if MaxBatchSize < 1 or MaxBatchSize > 1024:
  print("Error: The batch size must be between 1 && 1024")
  sys.exit(0)  
print("CMD-Line: Using {} as batch size".format(MaxBatchSize))
  
OutputDirectory = args.outputdirectory
# TODO: Add checks
print("CMD-Line: Using \"{}\" as output directory".format(OutputDirectory))

print("\n\n")


# Determine derived parameters

OneSigmaNoiseInRadians = math.radians(OneSigmaNoiseInDegrees)

NumberOfTrainingLocations = NumberOfTrainingBatches*MaxBatchSize
TrainingBatchSize = MaxBatchSize

NumberOfTestLocations = NumberOfTestingBatches*MaxBatchSize
TestingBatchSize = MaxBatchSize


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


# Everything ROOT related can only be loaded here otherwise it interferes with the argparse
from GRBData import GRBData
from GRBCreatorToyModel import GRBCreatorToyModel

# Load MEGAlib into ROOT so that it is usable
import ROOT as M
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")
M.PyConfig.IgnoreCommandLineOptions = True



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
print("Info: Total time to create data sets: {:.1f} seconds (= {:,.0f} events/second)".format(TimeCreation, (NumberOfTrainingLocations + NumberOfTestLocations) * (NumberOfComptonEvents + NumberOfBackgroundEvents) / TimeCreation))


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
      if XTrain[l, p, c, t] > 0:
        ax.scatter(math.degrees(PsiMin) + p * ResolutionInDegrees, math.degrees(ChiMin) + c * ResolutionInDegrees, math.degrees(PhiMin) + t * ResolutionInDegrees, XTrain[l, p, c, t])
        #print("{}, {}, {}".format(math.degrees(PsiMin) + p * ResolutionInDegrees, math.degrees(ChiMin) + c * ResolutionInDegrees, math.degrees(PhiMin) + t * ResolutionInDegrees))
        adds += XTrain[l, p, c, t]
    
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


L = tf.layers.conv3d(X, 64, 5, 2, 'VALID', activation = "relu")
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(L, 0.1*L)g


L = tf.layers.conv3d(L, 64, 3, 1, 'VALID', activation = "relu")
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(L, 0.1*L)

L = tf.layers.max_pooling3d(L, pool_size = [2,2,2], strides = 2)

L = tf.layers.conv3d(L, 128, 2, 2, 'VALID', activation = "relu")
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(X, 0.1*X)

L = tf.layers.conv3d(L, 128, 2, 2, 'VALID', activation = "relu")
#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
#L = tf.maximum(L, 0.1*L)

#L = tf.layers.max_pooling3d(L, pool_size = [2,2,2], strides = 2)
#L = tf.layers.conv3d(L, 128, 2, 2, 'VALID')
 
#L = tf.layers.dense(tf.reshape(L, [-1, reduce(lambda a,b:a*b, L.shape.as_list()[1:])]), 128)
L = tf.layers.dense(tf.reshape(L, [-1, reduce(lambda a,b:a*b, L.shape.as_list()[1:])]), 128)


#L = tf.layers.batch_normalization(L, training=tf.placeholder_with_default(True, shape=None))
L = tf.nn.relu(L)

print("      ... output layer ...")
Output = tf.layers.dense(tf.reshape(L, [-1, reduce(lambda a,b:a*b, L.shape.as_list()[1:])]), OutputDataSpaceSize)


#tf.print("Y: ", Y, output_stream=sys.stdout)


# Loss function - simple linear distance between output and ideal results
print("      ... loss function ...")
#LossFunction = tf.reduce_sum(np.abs(Output - Y)/NumberOfTestLocations)

for l in range(0, TrainingBatchSize):
  Real = M.MVector()
  Real.SetMagThetaPhi(1.0, Y[l, 0], Y[l, 1])
  Reconstructed = M.MVector()
  Reconstructed.SetMagThetaPhi(1.0, Output[l, 0], Output[l, 1])
  AngularDeviation = math.degrees(Real.Angle(Reconstructed))
  MeanAngularDeviation += AngularDeviation
  RMSAngularDeviation += math.pow(AngularDeviation, 2)

LossFunction = tf.reduce(MeanAngularDeviation / TrainingBatchSize)
#LossFunction = tf.reduce(MeanAngularDeviation / NumberOfTrainingBatches*TrainingBatchSize)

#LossFunction = tf.reduce_sum(tf.pow(Output - Y, 2))/NumberOfTestLocations
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
TimesNoImprovement = 0
BestMeanSquaredError = sys.float_info.max
BestMeanAngularDeviation = sys.float_info.max
BestRMSAngularDeviation = sys.float_info.max
BestLoss = sys.float_info.max
IterationOutputInterval = 10
CheckPointNum = 0

print("Info: Creating configuration and progress file")

with open(OutputDirectory + "/Configuration.txt", 'w') as f:
  f.write("Configuration\n\n")
  f.write("Mode: {}\n".format(Mode))
  if Mode == 'toymodel':
    f.write("NumberOfComptonEvents: {}\n".format(NumberOfComptonEvents))
    f.write("NumberOfBackgroundEvents: {}\n".format(NumberOfBackgroundEvents))
    f.write("Noise: {}\n".format(OneSigmaNoiseInDegrees))
    f.write("TrainingBatchSize: {}\n".format(TrainingBatchSize))
    f.write("TestingBatchSize: {}\n".format(TestingBatchSize))
  f.write("ResolutionInDegrees: {}\n".format(ResolutionInDegrees))
  f.write("MaxBatchSize: {}\n".format(MaxBatchSize))
  f.write("OutputDirectory: {}\n".format(OutputDirectory))
  
with open(OutputDirectory + '/Progress.txt', 'w') as f:
  f.write("Progress\n\n")



def CheckPerformance():
  global TimesNoImprovement
  global BestMeanSquaredError
  global BestMeanAngularDeviation
  global BestRMSAngularDeviation
  global IterationOutputInterval

  Improvement = False

  # Run the test data
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
      XSlice.put(GRB.getIndices(), GRB.getValues())
    
    XTest = XTest.reshape((TestingBatchSize, PsiBins, ChiBins, PhiBins, 1))
    
    
    # Step 2: Run it
    YOut = Session.run(Output, feed_dict={X: XTest})
    

    # Step 3: Analyze it
    # Calculate the angular deviation
    for l in range(0, TestingBatchSize):

      Real = M.MVector()
      Real.SetMagThetaPhi(1.0, YTest[l, 0], YTest[l, 1])

      Reconstructed = M.MVector()
      Reconstructed.SetMagThetaPhi(1.0, YOut[l, 0], YOut[l, 1])
  
      AngularDeviation = math.degrees(Real.Angle(Reconstructed))
  
      MeanAngularDeviation += AngularDeviation
      RMSAngularDeviation += math.pow(AngularDeviation, 2)
      
      if Batch == NumberOfTestingBatches-1:
        print("  Cross-Check element: {:-7.3f} degrees difference: {:-6.3f} vs. {:-6.3f} & {:-6.3f} vs. {:-6.3f}".format(AngularDeviation, YTest[l, 0].item(), YOut[l, 0].item(), YTest[l, 1].item(), YOut[l, 1].item()))
      
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
Iteration = 0
while Iteration < MaxIterations:
  Iteration += 1
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
      XSlice.put(GRB.getIndices(), GRB.getValues())
        
    XTrain = XTrain.reshape((TrainingBatchSize, PsiBins, ChiBins, PhiBins, 1))
    
    TimeConverting += time.time() - TimerConverting


    # The actual training
    TimerTraining = time.time()
    _, Loss = Session.run([Trainer, LossFunction], feed_dict={X: XTrain, Y: YTrain})
    TimeTraining += time.time() - TimerTraining
    
  # End for all batches


  # Check performance
  TimerTesting = time.time()
  print("\n\nIteration {}".format(Iteration))
  Improvement = CheckPerformance()
  
  if Improvement == True:
    TimesNoImprovement = 0

    Saver.save(Session, "{}/Model_{}.ckpt".format(OutputDirectory, Iteration))

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

print("\n\nTotal time converting per Iteration: {} sec".format(TimeConverting/Iteration))
print("Total time training per Iteration:   {} sec".format(TimeTraining/Iteration))
print("Total time testing per Iteration:    {} sec".format(TimeTesting/Iteration))


#input("Press [enter] to EXIT")
sys.exit(0)

