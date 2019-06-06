###################################################################################################
#
# GRBToyModel.py
#
# Copyright (C) by Andreas Zoglauer & Jasmine Singh.
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
from functools import reduce

import ROOT as M

# Load MEGAlib into ROOT so that it is usable
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")


print("\nGRB localization toy model (tensorflow based) \n")



###################################################################################################
# Step 1: Input parameters
###################################################################################################


# Input parameters
NumberOfComptonEvents = 500

NumberOfTrainingLocations = 16384
NumberOfTestLocations = 256

MaxBatchSize = 16

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


ResolutionInDegrees = 5

ThetaMin = 0
ThetaMax = np.pi
ThetaBins = int(180 / 5)

ChiMin = 0
ChiMax = np.pi
ChiBins = int(180 / 5)

PsiMin = 0
PsiMax = 2 * np.pi
PsiBins = int(360 / 5)


OneSigmaNoiseInRadians = math.radians(1.0)

# Set derived parameters
InputDataSpaceSize = ThetaBins * ChiBins * PsiBins
OutputDataSpaceSize = 2



###################################################################################################
# Step 2: Global functions
###################################################################################################


# Take care of Ctrl-C
Interrupted = False
NInterrupts = 0
def signal_handler(signal, frame):
  print("You pressed Ctrl+C!")
  global Interrupted
  Interrupted = True        
  global NInterrupts
  NInterrupts += 1
  if NInterrupts >= 3:
    print("Aborting!")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)




###################################################################################################
# Step 3: Create some training, test & verification data sets
###################################################################################################


print("Info: Creating %i Compton events" % ((NumberOfTrainingLocations + NumberOfTestLocations) * NumberOfComptonEvents))


def KleinNishina(Ei, phi):
  if Ei <= 0:
    #print("Error: Invalid input: Ei <= 0")
    return 0
  if phi < 0 or phi > math.pi:
    #print("Error: Invalid input: phi < 0 or phi > math.pi")
    return 0

  Radius = 2.8E-15
  E0 = 510.998910

  sinphi = math.sin(phi)
  Eg = -E0*Ei/(math.cos(phi)*Ei-Ei-E0)

  return 0.5*Radius*Radius*Eg*Eg/Ei/Ei*(Eg/Ei+Ei/Eg-sinphi*sinphi)*sinphi



def ComptonScatterAngle(Eg, Ee):
  E0 = 510.998910
  Value = 1 - E0 * (1.0/Eg - 1.0/(Ee + Eg))

  if Value <= -1 or Value >= 1:
    #print("Error: Invalid input: Value <= -1 or Value >= 1")
    return 0

  return math.acos(Value)



def Create(Ei, Rotation):

  # Simulate the gamma ray according to Butcher & Messel: Nuc Phys 20(1960), 15

  Ei_m = Ei / 510.998910

  Epsilon = 0.0
  EpsilonSquare = 0.0
  OneMinusCosTheta = 0.0
  SinThetaSquared = 0.0

  Epsilon0 = 1./(1. + 2.*Ei_m)
  Epsilon0Square = Epsilon0*Epsilon0
  Alpha1 = - math.log(Epsilon0)
  Alpha2 = 0.5*(1.- Epsilon0Square)

  Reject = 0.0

  while True:
    if Alpha1/(Alpha1+Alpha2) > random.random():
      Epsilon = math.exp(-Alpha1*random.random())
      EpsilonSquare = Epsilon*Epsilon
    else:
      EpsilonSquare = Epsilon0Square + (1.0 - Epsilon0Square)*random.random()
      Epsilon = math.sqrt(EpsilonSquare)
      
    OneMinusCosTheta = (1.- Epsilon)/(Epsilon*Ei_m)
    SinThetaSquared = OneMinusCosTheta*(2.-OneMinusCosTheta)
    Reject = 1.0 - Epsilon*SinThetaSquared/(1.0 + EpsilonSquare)

    if Reject < random.random():
      break
  
  CosTeta = 1.0 - OneMinusCosTheta; 

  # Set the new photon parameters --- direction is random since we didn't give a start direction

  Theta = np.arccos(1 - 2*random.random()) # Compton scatter angle since on axis
  Phi = 2.0 * np.pi * random.random();   

  Dg = M.MVector()
  Dg.SetMagThetaPhi(1.0, Theta, Phi) 
  Dg = Rotation * Dg

  Chi = Dg.Theta()
  Psi = Dg.Phi()

  Eg = Epsilon*Ei
  Ee = Ei - Eg
  
  #print(Theta, Chi, Psi, Eg+Ee)
  
  return Chi, Psi, Theta, Eg+Ee


# Dummy noising of the data
def Noise(Chi, Psi, Theta, NoiseOneSigmaInRadians):
  NoisedChi = sys.float_info.max
  while NoisedChi < 0 or NoisedChi > math.pi:
    NoisedChi = np.random.normal(Chi, NoiseOneSigmaInRadians)
    #print("Chi: {} {}".format(Chi, NoisedChi))

  NoisedPsi = sys.float_info.max
  while NoisedPsi < -math.pi or NoisedPsi > math.pi:
    NoisedPsi = np.random.normal(Psi, NoiseOneSigmaInRadians)
    #print("Psi {} {}".format(Psi, NoisedPsi))

  NoisedTheta = sys.float_info.max
  while NoisedTheta < 0 or NoisedTheta > math.pi:
    NoisedTheta = np.random.normal(Theta, NoiseOneSigmaInRadians)
    #print("Theta {} {}".format(Theta, NoisedTheta))

  return NoisedChi, NoisedPsi, NoisedTheta





# Set the toy training data
XTrain = np.zeros(shape=(NumberOfTrainingLocations, ThetaBins, ChiBins, PsiBins, 1))
YTrain = np.zeros(shape=(NumberOfTrainingLocations, OutputDataSpaceSize))

for l in range(0, NumberOfTrainingLocations):

  if l > 0 and l % 128 == 0:
    print("Training set creation: {}/{}".format(l, NumberOfTrainingLocations))

  # Create a random rotation matrix
  V = M.MVector()
  V.SetMagThetaPhi(1, np.arccos(1 - 2*random.random()), 2.0 * np.pi * random.random())
  Angle = 2.0 * np.pi * random.random()

  '''
  if random.random() < 0.25:
    V.SetMagThetaPhi(1, 0.4, 0.1)
    Angle = 0.6
  elif random.random() < 0.5:
    V.SetMagThetaPhi(1, 0.9, 0.3)
    Angle = 4.6
  elif random.random() < 0.75:
    V.SetMagThetaPhi(1, 0.4, 0.8)
    Angle = 2.6
  else:
    V.SetMagThetaPhi(1, 0.2, 0.6)
    Angle = 0.2 
  '''
    
  Rotation = M.MRotation(Angle, V)

  # Retrieve the origin of the gamma rays
  Origin = M.MVector(0, 0, 1)
  Origin = Rotation*Origin

  # Set the location data
  YTrain[l, 0] = Origin.Theta()
  YTrain[l, 1] = Origin.Phi()
  #print("theta={}, phi={}".format(Origin.Theta(), Origin.Phi()))
  
  # Create the input data
  for e in range(0, NumberOfComptonEvents):
    Chi, Psi, Theta, Energy = Create(511, Rotation)
  
    if OneSigmaNoiseInRadians > 0:
      Chi, Psi, Theta = Noise(Chi, Psi, Theta, OneSigmaNoiseInRadians)

    ChiBin = (int) (((Chi - ChiMin) / (ChiMax - ChiMin)) * ChiBins)
    PsiBin = (int) (((Psi - PsiMin) / (PsiMax - PsiMin)) * PsiBins)
    ThetaBin = (int) (((Theta - ThetaMin) / (ThetaMax - ThetaMin)) * ThetaBins)
    
    XTrain[l, ThetaBin, ChiBin, PsiBin] += 1

  '''
  # Plot the first test data point
  
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  print("Origin: {}, {}".format(YTest[0, 0], YTest[0, 1]))
  for e in range(0, NumberOfComptonEvents):
    ax.scatter(XTest[l, 4*e], XTest[l, 4*e+1], XTest[l, 4*e+2])  

  plt.show()
  plt.pause(0.001)

  input("Press [enter] to EXIT")
  sys.exit()
  '''
 
print("Training set creation: {}/{}".format(NumberOfTrainingLocations, NumberOfTrainingLocations))



# Set the toy testing data
XTest = np.zeros(shape=(NumberOfTestLocations, ThetaBins, ChiBins, PsiBins, 1))
YTest = np.zeros(shape=(NumberOfTestLocations, OutputDataSpaceSize))

for l in range(0, NumberOfTestLocations):

  if l > 0 and l % 128 == 0:
    print("Testing set creation: {}/{}".format(l, NumberOfTestLocations))

  # Create a random rotation matrix
  V = M.MVector()
  V.SetMagThetaPhi(1, np.arccos(1 - 2*random.random()), 2.0 * np.pi * random.random())
  Angle = 2.0 * np.pi * random.random()
  '''
  # Temp fixed:
  if random.random() < 0.25:
    V.SetMagThetaPhi(1, 0.4, 0.1)
    Angle = 0.6
  elif random.random() < 0.5:
    V.SetMagThetaPhi(1, 0.9, 0.3)
    Angle = 4.6
  elif random.random() < 0.75:
    V.SetMagThetaPhi(1, 0.4, 0.8)
    Angle = 2.6
  else:
    V.SetMagThetaPhi(1, 0.2, 0.6)
    Angle = 0.2 
  '''
  
  Rotation = M.MRotation(Angle, V)

  # Retrieve the origin of the gamma rays
  Origin = M.MVector(0, 0, 1)
  Origin = Rotation*Origin

  # Set the location data
  YTest[l, 0] = Origin.Theta()
  YTest[l, 1] = Origin.Phi()
  
  # Create the input data
  for e in range(0, NumberOfComptonEvents):
    Chi, Psi, Theta, Energy = Create(511, Rotation)
  
    if OneSigmaNoiseInRadians > 0:
      Chi, Psi, Theta = Noise(Chi, Psi, Theta, OneSigmaNoiseInRadians)


    ChiBin = (int) (((Chi - ChiMin) / (ChiMax - ChiMin)) * ChiBins)
    PsiBin = (int) (((Psi - PsiMin) / (PsiMax - PsiMin)) * PsiBins)
    ThetaBin = (int) (((Theta - ThetaMin) / (ThetaMax - ThetaMin)) * ThetaBins)
    
    XTest[l, ThetaBin, ChiBin, PsiBin] += 1

print("Testing set creation: {}/{}".format(NumberOfTestLocations, NumberOfTestLocations))





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
writer = tf.summary.FileWriter("OUT_ExampleCompton", sess.graph)
writer.close()

# Add ops to save and restore all the variables.
print("      ... saver ...")
Saver = tf.train.Saver()




###################################################################################################
# Step 5: Training and evaluating the network
###################################################################################################


print("Info: Training and evaluating the network")

# Train the network
Timing = time.process_time()

MaxTimesNoImprovement = 1000
TimesNoImprovement = 0
BestMeanSquaredError = sys.float_info.max
BestMeanAngularDeviation = sys.float_info.max
BestRMSAngularDeviation = sys.float_info.max
BestLoss = sys.float_info.max
IterationOutputInterval = 10

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
    YOut = sess.run(Output, feed_dict={X: XTest[Batch*TestingBatchSize:(Batch+1)*TestingBatchSize]})
    
    # Calculate the angular deviation

    for l in range(0, TestingBatchSize):
      Real = M.MVector()
      Real.SetMagThetaPhi(1.0, YTest[Batch*TestingBatchSize + l, 0], YTest[Batch*TestingBatchSize + l, 1])

      Reconstructed = M.MVector()
      Reconstructed.SetMagThetaPhi(1.0, YOut[l, 0], YOut[l, 1])
  
      AngularDeviation[Batch*TestingBatchSize + l] = math.degrees(Real.Angle(Reconstructed))
  
      MeanAngularDeviation += AngularDeviation[Batch*TestingBatchSize + l]
      RMSAngularDeviation += math.pow(AngularDeviation[Batch*TestingBatchSize + l], 2)
      
  # Calculate the mean RMS
  MeanAngularDeviation /= NumberOfTestingBatches*TestingBatchSize
  RMSAngularDeviation /= NumberOfTestingBatches*TestingBatchSize
  RMSAngularDeviation = math.sqrt(RMSAngularDeviation)


  #MeanSquaredError = sess.run(tf.nn.l2_loss(Output - YTest)/NumberOfTestLocations, feed_dict={X: XTest})

  print("  RMS Angular deviation:  {} (best: {})".format(round(RMSAngularDeviation, 5), round(BestRMSAngularDeviation, 5)))
  print("  Mean Angular deviation: {} (best: {})".format(MeanAngularDeviation, BestMeanAngularDeviation))
  #print("  MSE of test data:       {} (best: {})".format(round(MeanSquaredError, 5), round(BestMeanSquaredError, 5)))

  # Check for improvement mean
  if MeanAngularDeviation < BestMeanAngularDeviation:
    BestMeanAngularDeviation = MeanAngularDeviation
    Improvement = True

  # Check for improvement RMS
  if RMSAngularDeviation < BestRMSAngularDeviation:
    BestRMSAngularDeviation = RMSAngularDeviation
    Improvement = True
    
  # Check for improvement MSE
  #if MeanSquaredError < BestMeanSquaredError:   
   # BestMeanSquaredError = MeanSquaredError
   # Improvement = True

  # Look at the first few elements
  for i in range(10):
    XSingle = XTest[2*i:2*i+1]
    YSingle = YTest[2*i:2*i+1]
    YOutSingle = sess.run(Output, feed_dict={X: XSingle})

    VYTest = M.MVector()
    VYTest.SetMagThetaPhi(1, YSingle[0,0], YSingle[0,1])
    VYOut = M.MVector()
    VYOut.SetMagThetaPhi(1, YOutSingle[0,0], YOutSingle[0,1])
    Angle = math.degrees(VYTest.Angle(VYOut))


    print("  Cross-Check element {}: {} degrees difference: {} vs. {} & {} vs. {}".format(i, round(Angle, 5), round(YSingle[0,0], 5), round(YOutSingle[0,0], 5), round(YSingle[0,1], 5), round(YOutSingle[0,1], 5)))

  return Improvement

# Main training and evaluation loop
MaxIterations = 50000
for Iteration in range(0, MaxIterations):
  # Take care of Ctrl-C
  if Interrupted == True: break

  for Batch in range(0, NumberOfTrainingBatches):
    # Take care of Ctrl-C
    if Interrupted == True: break

    _, Loss = sess.run([Trainer, LossFunction], feed_dict={X: XTrain[Batch*TrainingBatchSize:(Batch+1)*TrainingBatchSize], Y: YTrain[Batch*TrainingBatchSize:(Batch+1)*TrainingBatchSize]})
  
    #if Batch == 0 and (Loss < BestLoss or Iteration == 1 or (Iteration > 0 and Iteration % IterationOutputInterval == 0)):
      #print("\nIteration {}".format(Iteration))
      #print("  Loss (training data): {} (best: {})".format(Loss, BestLoss))
  
  # Check performance
  print("\nIteration {}".format(Iteration))
  Improvement = CheckPerformance()
  
  if Improvement == True:
    TimesNoImprovement = 0
  else:
    TimesNoImprovement += 1

  # Exit strategy
  if TimesNoImprovement == MaxTimesNoImprovement:
    print("No improvement for {} iterations. Best RMS result: {}".format(MaxTimesNoImprovement, BestRMSAngularDeviation))
    break;
    
  #plt.pause(0.001)


Timing = time.process_time() - Timing
if Iteration > 0: 
  print("Time per training loop: ", Timing/Iteration, " seconds")


input("Press [enter] to EXIT")
sys.exit(0)

