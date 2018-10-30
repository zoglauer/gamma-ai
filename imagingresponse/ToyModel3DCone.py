###################################################################################################
#
# ToyModel2DGauss.py
#
# Copyright (C) by Andreas Zoglauer & contributors.
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice. 
#  
###################################################################################################

## TODO: There is an unknown memeory leak

###################################################################################################

import tensorflow as tf
import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import signal
import sys
import time
import math
import csv

#you might have to download this package
import statistics


###################################################################################################
# Step 1: Input parameters
###################################################################################################


print("\nToyModel: (x,y) --> exp(-(x-x0)^2/s0^2)*exp(-(y-y0)^2/s1^2), random) ∀ x, y ∈ [-1, 1]\n")

gMinXY = -1
gMaxXY = +1
gSigmaX = 0.1
gSigmaY = 0.2
gMinZ = 0
gMaxZ = 1


gTrainingGridXY = 30
gTrainingGridZ = 30
bin_size_xy = (gMaxXY - gMinXY)/gTrainingGridXY

gGridCenters_x = np.zeros([gTrainingGridXY])
#gGridCenters_y = np.zeros([gTrainingGridXY]) same as gGridCeters_x
gGridCenters_z = np.zeros([gTrainingGridZ])



for x in range(0, gTrainingGridXY):
  gGridCenters_x[x] = gMinXY + (x+0.5)*(gMaxXY-gMinXY)/gTrainingGridXY

for z in range(0, gTrainingGridZ):
  gGridCenters_z[z] = gMinZ + (z+0.5)*(gMaxZ-gMinZ)/gTrainingGridZ


# Set data parameters
InputDataSpaceSize = 2 
OutputDataSpaceSize = gTrainingGridXY*gTrainingGridXY*gTrainingGridZ

SubBatchSize = 1024

NTrainingBatches = 1
TrainingBatchSize = NTrainingBatches*SubBatchSize

NTestingBatches = 1
TestBatchSize = NTestingBatches*SubBatchSize


###################################################################################################
# Step 2: Global functions
###################################################################################################


# First take care of Ctrl-C
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
# Step 3: Create the training, test & verification data sets
###################################################################################################

print("Info: Creating %i data sets" % (TrainingBatchSize + TestBatchSize))


def CreateRandomResponsePoint(PosX, PosY):
  return (PosX + random.gauss(PosX, gSigma), random.uniform(gMinXY, gMaxXY))

def get_gauss(d, sigma = 1):
  return 1/(sigma*math.sqrt(2*np.pi)) * math.exp(-0.5*pow(d/sigma, 2))

def CreateFullResponse(PosX, PosY):
  Out = np.zeros(shape=(1, OutputDataSpaceSize))

  for x in range(0, gTrainingGridXY):
    for y in range(0, gTrainingGridXY):
      for z in range(0, gTrainingGridZ):
        r = math.sqrt((PosX - gGridCenters_x[x])**2 + (PosY - gGridCenters_x[y])**2 )
        #unsure of indexing of GridCenters
        Out[0, x + y*gTrainingGridXY + z*gTrainingGridXY*gTrainingGridXY] = get_gauss(math.fabs(r - gGridCenters_z[z]));  

  return Out
    


XTrain = np.zeros(shape=(TrainingBatchSize, InputDataSpaceSize))
YTrain = np.zeros(shape=(TrainingBatchSize, OutputDataSpaceSize))

for i in range(0, TrainingBatchSize):
  if i > 0 and i % 128 == 0:
    print("Training set creation: {}/{}".format(i, TrainingBatchSize))
    XTrain[i,0] = random.uniform(gMinXY, gMaxXY)
    XTrain[i,1] = random.uniform(gMinXY, gMaxXY)
    YTrain[i,] = CreateFullResponse(XTrain[i,0], XTrain[i,1])

XTest = np.zeros(shape=(TestBatchSize, InputDataSpaceSize)) # should this even be 3?
YTest = np.zeros(shape=(TestBatchSize, OutputDataSpaceSize)) #
for i in range(0, TestBatchSize):
  if i > 0 and i % 128 == 0:
    print("Testing set creation: {}/{}".format(i, TestBatchSize))
    XTest[i, 0] = random.uniform(gMinXY, gMaxXY)
    XTest[i, 1] = random.uniform(gMinXY, gMaxXY)
    YTest[i, ] = CreateFullResponse(XTest[i,0], XTest[i,1])





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
H = tf.contrib.layers.fully_connected(X, 10) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
H = tf.contrib.layers.fully_connected(H, 100) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
H = tf.contrib.layers.fully_connected(H, 1000) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))

  
print("      ... output layer ...")
Output = tf.contrib.layers.fully_connected(H, OutputDataSpaceSize, activation_fn=None)

# Loss function 
print("      ... loss function ...")
#LossFunction = tf.reduce_sum(np.abs(Output - Y)/TestBatchSize)
LossFunction = tf.reduce_sum(tf.pow(Output - Y, 2))/TestBatchSize

# Minimizer
print("      ... minimizer ...")
Trainer = tf.train.AdamOptimizer().minimize(LossFunction)

# Create and initialize the session
print("      ... session ...")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("      ... writer ...")
writer = tf.summary.FileWriter("OUT_ToyModel2DGauss", sess.graph)
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

TimesNoImprovement = 0
BestMeanSquaredError = sys.float_info.max

def CheckPerformance():
  global TimesNoImprovement
  global BestMeanSquaredError

  MeanSquaredError = sess.run(tf.nn.l2_loss(Output - YTest)/TestBatchSize,  feed_dict={X: XTest})
  
  print("Iteration {} - MSE of test data: {}".format(Iteration, MeanSquaredError))

  if MeanSquaredError <= BestMeanSquaredError:    # We need equal here since later ones are usually better distributed
    BestMeanSquaredError = MeanSquaredError
    TimesNoImprovement = 0
    
    #Saver.save(sess, "model.ckpt")

    
    # Test just the first test case:
    XSingle = XTest[0:1]
    YSingle = YTest[0:1]
    YOutSingle = sess.run(Output, feed_dict={X: XSingle})
    print("YOut: ", YOutSingle.shape)
    print("YSingle: ", YSingle.shape)
    XV, YV = np.meshgrid(gGridCenters_x, gGridCenters_x)

    z = 5
    Y_now = np.zeros(shape=(1, gTrainingGridXY*gTrainingGridXY))
    Y_OutS = np.zeros(shape=(1, gTrainingGridXY*gTrainingGridXY))

    for x in range(gTrainingGridXY):
      for y in range(gTrainingGridXY):
        Y_now[0, x + y*gTrainingGridXY] = YSingle[0, x + y*gTrainingGridXY + z*gTrainingGridXY*gTrainingGridXY]
        Y_OutS[0, x +  y*gTrainingGridXY] = YOutSingle[0, x + y*gTrainingGridXY + z*gTrainingGridXY*gTrainingGridXY]
    
    fig = plt.figure(1)
    plt.clf()
    ax = fig.gca(projection='3d')
    Out1 = Y_now.reshape(gTrainingGridXY , gTrainingGridXY)
    surf = ax.plot_surface(XV, YV, Out1, cmap=cm.coolwarm)  #, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
  
    fig = plt.figure(2)
    plt.clf()
    ax = fig.gca(projection='3d')
    Out1 = Y_OutS.reshape(gTrainingGridXY, gTrainingGridXY)
    surf = ax.plot_surface(XV, YV, Out1, cmap=cm.coolwarm)  #, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ###insersions###

    plt.ion()
    plt.show()
    plt.pause(0.001)

  else:
    TimesNoImprovement += 1


# Main training and evaluation loop
MaxIterations = 50000
for Iteration in range(0, MaxIterations):
  # Take care of Ctrl-C
  if Interrupted == True: break

  # Train
  for Batch in range(0, NTrainingBatches):
    if Interrupted == True: break

    #if Batch%8 == 0:
    #  print("Iteration %6d, Batch %4d)" % (Iteration, Batch))

    Start = Batch * SubBatchSize
    Stop = (Batch + 1) * SubBatchSize
    sess.run(Trainer, feed_dict={X: XTrain[Start:Stop], Y: YTrain[Start:Stop]})
    
  # Check performance: Mean squared error
  if Iteration > 0 and Iteration % 20 == 0:
    CheckPerformance()

  if TimesNoImprovement == 100:
    print("No improvement for 30 rounds")
    break;


Timing = time.process_time() - Timing
if Iteration > 0: 
  print("Time per training loop: ", Timing/Iteration, " seconds")


input("Press [enter] to EXIT")
sys.exit(0)


# END  
###################################################################################################
