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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import signal
import sys
import time
import math
import csv

import ROOT as M
# Load MEGAlib into ROOT so that it is usable
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")


print("\nGRB localization toy model (tensorflow based) \n")



###################################################################################################
# Step 1: Input parameters
###################################################################################################


# Input parameters
NumberOfComptonEvents = 500
NumberOfTrainingLocations = 1024
NumberOfTestLocations = 128


# Set derived parameters
InputDataSpaceSize = 4*NumberOfComptonEvents
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
    if Alpha1/(Alpha1+Alpha2) > np.random.random():
      Epsilon = math.exp(-Alpha1*np.random.random())
      EpsilonSquare = Epsilon*Epsilon
    else:
      EpsilonSquare = Epsilon0Square + (1.0 - Epsilon0Square)*np.random.random()
      Epsilon = math.sqrt(EpsilonSquare)
      
    OneMinusCosTheta = (1.- Epsilon)/(Epsilon*Ei_m)
    SinThetaSquared = OneMinusCosTheta*(2.-OneMinusCosTheta)
    Reject = 1.0 - Epsilon*SinThetaSquared/(1.0 + EpsilonSquare)

    if Reject < np.random.random():
      break
  
  CosTeta = 1.0 - OneMinusCosTheta; 

  # Set the new photon parameters --- direction is random since we didn't give a start direction

  Theta = np.arccos(1 - 2*np.random.random()) # Compton scatter angle since on axis
  Phi = 2.0 * np.pi * np.random.random();   

  Dg = M.MVector()
  Dg.SetMagThetaPhi(1.0, Theta, Phi) 
  Dg = Rotation * Dg

  Chi = Dg.Theta()
  Psi = Dg.Phi()

  Eg = Epsilon*Ei
  Ee = Ei - Eg
  
  #print(Theta, Chi, Psi, Eg+Ee)
  
  return np.array([Chi, Psi, Theta, Eg+Ee])




# Set the toy training data
XTrain = np.zeros(shape=(NumberOfTrainingLocations, InputDataSpaceSize))
YTrain = np.zeros(shape=(NumberOfTrainingLocations, OutputDataSpaceSize))

for l in range(0, NumberOfTrainingLocations):

  if l > 0 and l % 128 == 0:
    print("Training set creation: {}/{}".format(l, NumberOfTrainingLocations))

  # Create a random rotation matrix
  V = M.MVector()
  V.SetMagThetaPhi(1, np.arccos(1 - 2*np.random.random()), 2.0 * np.pi * np.random.random())
  Angle = 2.0 * np.pi * np.random.random()
  Rotation = M.MRotation(Angle, V)

  # Retrieve the origin of the gamma rays
  Origin = M.MVector(0, 0, 1)
  Origin = Rotation*Origin

  # Set the location data
  YTrain[l, 0] = Origin.Theta()
  YTrain[l, 1] = Origin.Phi()
  
  # Create the input data
  for e in range(0, NumberOfComptonEvents):
    XTrain[l, 4*e:4*e+4] = Create(511, Rotation)



# Set the toy testing data
XTest = np.zeros(shape=(NumberOfTestLocations, InputDataSpaceSize))
YTest = np.zeros(shape=(NumberOfTestLocations, OutputDataSpaceSize))

for l in range(0, NumberOfTestLocations):

  if l > 0 and l % 128 == 0:
    print("Testing set creation: {}/{}".format(l, NumberOfTestLocations))

  # Create a random rotation matrix
  V = M.MVector()
  V.SetMagThetaPhi(1, np.arccos(1 - 2*np.random.random()), 2.0 * np.pi * np.random.random())
  Angle = 2.0 * np.pi * np.random.random()
  Rotation = M.MRotation(Angle, V)

  # Retrieve the origin of the gamma rays
  Origin = M.MVector(0, 0, 1)
  Origin = Rotation*Origin

  # Set the location data
  YTest[l, 0] = Origin.Theta()
  YTest[l, 1] = Origin.Phi()
  
  # Create the input data
  for e in range(0, NumberOfComptonEvents):
    XTest[l, 4*e:4*e+4] = Create(511, Rotation)


'''
# Plot the first test data point
fig = plt.figure()
ax = fig.gca(projection='3d')

print("Origin: {}, {}".format(YTest[0, 0], YTest[0, 1]))
for e in range(0, NumberOfComptonEvents):
  ax.scatter(XTest[0, 4*e], XTest[0, 4*e+1], XTest[0, 4*e+2])  

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
X = tf.placeholder(tf.float32, [None, InputDataSpaceSize], name="X")
Y = tf.placeholder(tf.float32, [None, OutputDataSpaceSize], name="Y")


# Layers: 1st hidden layer X1, 2nd hidden layer X2, etc.
print("      ... hidden layers ...")
NNodes = 128
NHiddenLayers = 3
H = tf.contrib.layers.fully_connected(X, NNodes) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
for i in range(1, NHiddenLayers):
  NNodes = NNodes // 2
  H = tf.contrib.layers.fully_connected(H, NNodes) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))

  
print("      ... output layer ...")
Output = tf.contrib.layers.fully_connected(H, OutputDataSpaceSize, activation_fn=None)

# Loss function - simple linear distance between output and ideal results
print("      ... loss function ...")
#LossFunction = tf.reduce_sum(np.abs(Output - Y)/TestBatchSize)
LossFunction = tf.reduce_sum(tf.pow(Output - Y, 2))/NumberOfTestLocations

# Minimizer
print("      ... minimizer ...")
Trainer = tf.train.AdamOptimizer().minimize(LossFunction)

# Create and initialize the session
print("      ... session ...")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

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
BestRMSAngularDeviation = sys.float_info.max
BestLoss = sys.float_info.max
IterationOutputInterval = 10

def CheckPerformance():
  global TimesNoImprovement
  global BestMeanSquaredError
  global BestRMSAngularDeviation
  global IterationOutputInterval

  # Look at the first
  XSingle = XTest[0:1]
  YSingle = YTest[0:1]
  YOutSingle = sess.run(Output, feed_dict={X: XSingle})
  
  if Iteration == 1 or (Iteration > 0 and Iteration % IterationOutputInterval == 0):
    print("  Cross-Check first element: {} vs. {} & {} vs. {}".format(YSingle[0,0], YOutSingle[0,0], YSingle[0,1], YOutSingle[0,1]))

  # Run the test data
  YOut = sess.run(Output, feed_dict={X: XTest})
  #YOut = sess.run(Output, feed_dict={X: XTrain})

  # Calculate the angular deviation
  AngularDeviation = np.zeros(shape=(NumberOfTestLocations, 1))
  RMSAngularDeviation = 0
  for l in range(0, NumberOfTestLocations):
    Real = M.MVector()
    Real.SetMagThetaPhi(1.0, YTest[l, 0], YTest[l, 1])
    #Real.SetMagThetaPhi(1.0, YTrain[l, 0], YTrain[l, 1])
    Reconstructed = M.MVector()
    Reconstructed.SetMagThetaPhi(1.0, YOut[l, 0], YOut[l, 1])
  
    AngularDeviation[l] = Real.Angle(Reconstructed)
  
    RMSAngularDeviation += math.pow(AngularDeviation[l], 2)
      
  # Calculate the RMS
  RMSAngularDeviation /= NumberOfTestLocations
  RMSAngularDeviation = math.sqrt(RMSAngularDeviation)


  MeanSquaredError = sess.run(tf.nn.l2_loss(Output - YTest)/NumberOfTestLocations, feed_dict={X: XTest})

  if Iteration == 1 or (Iteration > 0 and Iteration % IterationOutputInterval == 0):
    print("  RMS of test data: {} (best: {})".format(RMSAngularDeviation, BestRMSAngularDeviation))
    print("  MSE of test data: {} (best: {})".format(MeanSquaredError, BestMeanSquaredError))

  # Check for improvement RMS
  if RMSAngularDeviation < BestRMSAngularDeviation:
    BestRMSAngularDeviation = RMSAngularDeviation

  # Check for improvement MSE
  if MeanSquaredError < BestMeanSquaredError:   
    BestMeanSquaredError = MeanSquaredError





# Main training and evaluation loop
MaxIterations = 50000
for Iteration in range(0, MaxIterations):
  # Take care of Ctrl-C
  if Interrupted == True: break

  _, Loss = sess.run([Trainer, LossFunction], feed_dict={X: XTrain, Y: YTrain})
  
  if Iteration == 1 or (Iteration > 0 and Iteration % IterationOutputInterval == 0):
    print("\nIteration {}".format(Iteration))
    print("  Loss: {} (best: {})".format(Loss, BestLoss))
  
  # Check performance
  CheckPerformance()
  
  if Loss < BestLoss:
    BestLoss = Loss
    TimesNoImprovement = 0
  else:
    TimesNoImprovement += 1

  # Exit strategy
  if TimesNoImprovement == MaxTimesNoImprovement:
    print("No improvement for {} rounds. Best RMS result: {}".format(MaxTimesNoImprovement, BestRMSAngularDeviation))
    break;
    
  #plt.pause(0.001)


Timing = time.process_time() - Timing
if Iteration > 0: 
  print("Time per training loop: ", Timing/Iteration, " seconds")


input("Press [enter] to EXIT")
sys.exit(0)

